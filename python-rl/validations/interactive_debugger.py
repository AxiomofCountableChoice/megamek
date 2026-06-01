import os
import sys
import time
import json
import torch
import threading
from flask import Flask, request, jsonify, render_template

# Add python-rl to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.agent import MegaMekAgent
from env import MegaMekEnvironment
from state_parser import StateParser

app = Flask(__name__, template_folder='templates')

device = torch.device('cpu')
agent = MegaMekAgent(hidden_dim=128, ensemble_size=8).to(device)
try:
    agent.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "..", "model_objects", "bc_agent.pt"), map_location=device))
    print("Loaded bc_agent.pt")
except Exception as e:
    print(f"Failed to load agent: {e}")
agent.eval()

# Delay initialization until thread starts
env = None

state_lock = threading.Lock()
current_state = None
step_counter = 0
action_event = threading.Event()
user_action = None

global_state_graph = None
global_mask = None
global_hetero_data = None

@app.route('/')
def index():
    return render_template('debugger.html')

@app.route('/api/state')
def get_state():
    with state_lock:
        if current_state is None:
            return jsonify({"status": "waiting"})
        return jsonify({"status": "ready", "step_id": step_counter, "data": current_state})

@app.route('/api/action', methods=['POST'])
def submit_action():
    global user_action
    user_action = request.json
    action_event.set()
    return jsonify({"status": "ok"})

@app.route('/api/score_weapon_tree', methods=['POST'])
def score_weapon_tree():
    partial_action = request.json
    with state_lock:
        if current_state is None or global_state_graph is None or global_mask is None:
            return jsonify({"status": "error", "message": "No active state"}), 400
        
        try:
            # We must use global_state_graph.to_homogeneous() if hetero_data is expected,
            # but wait, the agent in interactive_debugger uses state_graph directly!
            # Let's pass what we have.
            res = agent.score_partial_weapon_trajectory(
                global_state_graph, 
                global_mask, 
                global_hetero_data, 
                partial_action
            )
            return jsonify({"status": "ok", "probs": res})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

def run_env():
    global current_state, step_counter, user_action, env
    global global_state_graph, global_mask, global_hetero_data
    
    while True:
        print("Connecting to MegaMek environment on port 5001...")
        env = None
        state_graph = None
        mask = None
        raw_payload = None
        hetero_data = None
        
        # Wait until it connects successfully
        while env is None:
            try:
                temp_env = MegaMekEnvironment(port=5001, device=device)
                state_graph, mask, raw_payload = temp_env.reset()
                hetero_data = state_graph.to_homogeneous() if state_graph is not None else None
                env = temp_env
                break
            except Exception as e:
                print(f"Exception during env.reset(): {e}")
                print("MegaMek not connected on 5001. Waiting...")
                time.sleep(5)
                
        topology = getattr(env, 'topology_payload', {})
        
        parser = StateParser()
        parser.handle_topology(topology)
        
        try:
            while True:
                # compute forward pass
                action_dict, v_mean, probs, mu_log_prob = ({}, torch.tensor(0.0), None, None)
                if state_graph is not None and 'action' in state_graph.node_types and state_graph['action'].x is not None and state_graph['action'].x.size(0) > 0:
                    with torch.no_grad():
                        action_dict, v_mean, probs, mu_log_prob = agent.get_action(state_graph, mask, deterministic=True)
                
                # Format probabilities for the UI
                formatted_probs = None
                if probs is not None:
                    if isinstance(probs, list):
                        formatted_probs = [p.tolist() for p in probs]
                    elif isinstance(probs, dict):
                        formatted_probs = {k: (v.tolist() if torch.is_tensor(v) else v) for k,v in probs.items()}
                    elif torch.is_tensor(probs):
                        formatted_probs = probs.tolist()
                
                # Build step data
                step_dict = {
                    "context": raw_payload.get("context", "UNKNOWN") if raw_payload else "UNKNOWN",
                    "action": action_dict, # Suggested action
                    "reward": raw_payload.get("reward", 0.0) if raw_payload else 0.0,
                    "total_reward": 0.0, # Will track in frontend
                    "reports": raw_payload.get("reports", []) if raw_payload else [],
                    "entities": raw_payload.get("state", {}).get("entities", []) if raw_payload else [],
                    "weapons": raw_payload.get("state", {}).get("weapons", []) if raw_payload else [],
                    "los_threats": raw_payload.get("state", {}).get("los_threat_edges", []) if raw_payload else [],
                    "los_targets": raw_payload.get("state", {}).get("los_target_edges", []) if raw_payload else [],
                    "movement_threats": raw_payload.get("state", {}).get("movement_threat_edges", []) if raw_payload else [],
                    "move_tmm_0": raw_payload.get("state", {}).get("move_type_tmm_0_edges", []) if raw_payload else [],
                    "move_tmm_1": raw_payload.get("state", {}).get("move_type_tmm_1_edges", []) if raw_payload else [],
                    "move_tmm_2": raw_payload.get("state", {}).get("move_type_tmm_2_edges", []) if raw_payload else [],
                    "move_tmm_3": raw_payload.get("state", {}).get("move_type_tmm_3_edges", []) if raw_payload else [],
                    "move_tmm_4": raw_payload.get("state", {}).get("move_type_tmm_4_edges", []) if raw_payload else [],
                    "partial_covers": raw_payload.get("state", {}).get("partial_cover_edges", []) if raw_payload else [],
                    "entities_meta": raw_payload.get("state", {}).get("entities_meta", []) if raw_payload else [],
                    "mask": mask,
                    "state_info": {
                        "phase": raw_payload.get("state", {}).get("phase_main", "UNKNOWN") if raw_payload else "UNKNOWN",
                        "round": raw_payload.get("state", {}).get("round_number", 0) if raw_payload else 0,
                        "turn": raw_payload.get("state", {}).get("turn_number", 0) if raw_payload else 0
                    },
                    "v_mean": v_mean.item() if hasattr(v_mean, "item") else v_mean,
                    "probs": formatted_probs,
                    "topology": topology
                }
                
                with state_lock:
                    step_counter += 1
                    current_state = step_dict
                    global_state_graph = state_graph
                    global_mask = mask
                    global_hetero_data = hetero_data
                    
                print(f"Waiting for user action via UI... (Step {step_counter})")
                action_event.clear()
                action_event.wait()
                
                with state_lock:
                    current_state = None # set to waiting state
                    
                print(f"Applying action: {user_action}")
                
                # Handle the case where action isn't valid, but let MegaMek handle it
                is_valid = True
                if (state_graph is None) or ('action' not in state_graph.node_types) or (state_graph['action'].x is None) or (state_graph['action'].x.size(0) == 0):
                    is_valid = False
                    
                state_graph, mask, done, raw_payload = env.step(user_action)
                hetero_data = state_graph.to_homogeneous() if state_graph is not None else None
                    
                if done:
                    print("Game Over. Resetting...")
                    state_graph, mask, raw_payload = env.reset()
                    hetero_data = state_graph.to_homogeneous() if state_graph is not None else None
        except Exception as e:
            print(f"Connection lost! {e}")
            with state_lock:
                current_state = None
            try:
                if env is not None:
                    env.close()
            except:
                pass
            time.sleep(2)

if __name__ == '__main__':
    threading.Thread(target=run_env, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
