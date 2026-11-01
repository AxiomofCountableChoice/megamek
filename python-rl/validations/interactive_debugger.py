import argparse
import atexit
import os
import signal
import subprocess
import sys
import threading
import time
import torch
from flask import Flask, jsonify, render_template, request

# Add python-rl to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from env import MegaMekEnvironment
from models.agent import MegaMekAgent
from state_parser import StateParser

app = Flask(__name__, template_folder="templates")

device = torch.device("cpu")
agent = MegaMekAgent(hidden_dim=128, ensemble_size=8).to(device)
agent.eval()

# Runtime state
env = None
env_port = 5001
server_proc = None

state_lock = threading.Lock()
current_state = None
step_counter = 0
action_event = threading.Event()
user_action = None

global_state_graph = None
global_mask = None
cumulative_reports = []


def load_agent_checkpoint(model_path=None):
    base_dir = os.path.join(os.path.dirname(__file__), "..", "model_objects")
    candidates = (
        [model_path]
        if model_path
        else [
            os.path.join(base_dir, "impala_agent_latest.pt"),
            os.path.join(base_dir, "bc_agent.pt"),
        ]
    )
    for path in candidates:
        if path and os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=device, weights_only=False)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    agent.load_state_dict(ckpt["model_state_dict"])
                else:
                    agent.load_state_dict(ckpt)
                agent.eval()
                print(f"[Debugger] Loaded checkpoint: {path}")
                return True
            except Exception as e:
                print(f"[Debugger] Failed to load {path}: {e}")
    print("[Debugger] Using freshly initialized MegaMekAgent weights.")
    return False


def terminate_spawned_server():
    global server_proc
    if server_proc is not None and server_proc.poll() is None:
        try:
            pgid = os.getpgid(server_proc.pid)
            print(f"[Debugger] Terminating spawned MegaMek server process group (PGID {pgid})...")
            os.killpg(pgid, signal.SIGTERM)
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
        except Exception as e:
            print(f"[Debugger] Error terminating MegaMek server: {e}")
        server_proc = None


def spawn_headless_megamek(mm_port=24000, bv=4000, max_meks=4):
    global server_proc
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    mm_bin = os.path.join(repo_root, "build", "install", "MegaMek", "bin", "megamek")
    cmd = [
        mm_bin,
        "-rl_offline",
        "-port",
        str(mm_port),
        "-autogen",
        "-random_map",
        "-random_bv",
        str(bv),
        "-max_meks",
        str(max_meks),
    ]
    child_env = os.environ.copy()
    child_env["JAVA_OPTS"] = "-Djava.awt.headless=true -Xmx2G"
    child_env["RL_DEBUG_TIMEOUT"] = "600000"  # 10 minutes timeout per step for human debugging

    print(f"[Debugger] Spawning headless MegaMek match on server port {mm_port} (RL port {mm_port + 1000})...")
    server_proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=child_env,
        start_new_session=True,
    )
    atexit.register(terminate_spawned_server)


@app.route("/")
def index():
    return render_template("debugger.html")


@app.route("/api/state")
def get_state():
    with state_lock:
        if current_state is None:
            return jsonify({"status": "waiting"})
        return jsonify({"status": "ready", "step_id": step_counter, "data": current_state})


@app.route("/api/action", methods=["POST"])
def submit_action():
    global user_action
    user_action = request.json
    action_event.set()
    return jsonify({"status": "ok"})


@app.route("/api/score_weapon_tree", methods=["POST"])
def score_weapon_tree():
    partial_action = request.json
    with state_lock:
        if current_state is None or global_state_graph is None or global_mask is None:
            return jsonify({"status": "error", "message": "No active state"}), 400

        try:
            res = agent.score_partial_weapon_trajectory(
                global_state_graph,
                global_mask,
                global_state_graph,
                partial_action,
            )
            return jsonify({"status": "ok", "probs": res})
        except Exception as e:
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500


def run_env():
    global current_state, step_counter, user_action, env
    global global_state_graph, global_mask, cumulative_reports

    while True:
        print(f"[Debugger] Connecting to MegaMek environment on port {env_port}...")
        env = None
        state_graph = None
        mask = None
        raw_payload = None
        cumulative_reports = []

        while env is None:
            try:
                temp_env = MegaMekEnvironment(port=env_port, device=device)
                state_graph, mask, raw_payload = temp_env.reset()
                env = temp_env
                break
            except Exception as e:
                print(f"[Debugger] MegaMek not ready on port {env_port} ({e}). Retrying in 3s...")
                time.sleep(3)

        topology = getattr(env, "topology_payload", {})
        parser = StateParser()
        parser.handle_topology(topology)

        try:
            while True:
                action_dict, v_mean, probs, mu_log_prob = ({}, torch.tensor(0.0), None, None)
                ui_probs = []
                if (
                    state_graph is not None
                    and "action" in state_graph.node_types
                    and state_graph["action"].x is not None
                    and state_graph["action"].x.size(0) > 0
                ):
                    with torch.no_grad():
                        action_dict, v_mean, probs, mu_log_prob = agent.get_action(
                            state_graph, mask, deterministic=True
                        )
                        ui_probs = agent.compute_ui_action_probabilities(state_graph, mask)

                step_reports = raw_payload.get("reports", []) if raw_payload else []
                if step_reports:
                    cumulative_reports.extend(step_reports)

                step_dict = {
                    "context": raw_payload.get("context", "UNKNOWN") if raw_payload else "UNKNOWN",
                    "action": action_dict,
                    "reward": raw_payload.get("reward", 0.0) if raw_payload else 0.0,
                    "total_reward": 0.0,
                    "reports": step_reports,
                    "all_reports": list(cumulative_reports),
                    "entities": raw_payload.get("state", {}).get("entities", []) if raw_payload else [],
                    "weapons": raw_payload.get("state", {}).get("weapons", []) if raw_payload else [],
                    "los_threats": raw_payload.get("state", {}).get("los_threat_edges", []) if raw_payload else [],
                    "los_targets": raw_payload.get("state", {}).get("los_target_edges", []) if raw_payload else [],
                    "movement_threats": raw_payload.get("state", {}).get("movement_threat_edges", [])
                    if raw_payload
                    else [],
                    "move_tmm_0": raw_payload.get("state", {}).get("move_type_tmm_0_edges", []) if raw_payload else [],
                    "move_tmm_1": raw_payload.get("state", {}).get("move_type_tmm_1_edges", []) if raw_payload else [],
                    "move_tmm_2": raw_payload.get("state", {}).get("move_type_tmm_2_edges", []) if raw_payload else [],
                    "move_tmm_3": raw_payload.get("state", {}).get("move_type_tmm_3_edges", []) if raw_payload else [],
                    "move_tmm_4": raw_payload.get("state", {}).get("move_type_tmm_4_edges", []) if raw_payload else [],
                    "partial_covers": raw_payload.get("state", {}).get("partial_cover_edges", [])
                    if raw_payload
                    else [],
                    "entities_meta": raw_payload.get("state", {}).get("entities_meta", []) if raw_payload else [],
                    "mask": mask,
                    "state_info": {
                        "phase": raw_payload.get("state", {}).get("phase_main", "UNKNOWN")
                        if raw_payload
                        else "UNKNOWN",
                        "round": raw_payload.get("state", {}).get("round_number", 0) if raw_payload else 0,
                        "turn": raw_payload.get("state", {}).get("turn_number", 0) if raw_payload else 0,
                    },
                    "v_mean": float(v_mean.item()) if hasattr(v_mean, "item") else float(v_mean or 0.0),
                    "probs": ui_probs,
                    "topology": topology,
                }

                with state_lock:
                    step_counter += 1
                    current_state = step_dict
                    global_state_graph = state_graph
                    global_mask = mask

                print(f"[Debugger] Waiting for user action via UI at http://localhost:5000 (Step {step_counter})...")
                action_event.clear()
                action_event.wait()

                with state_lock:
                    current_state = None

                print(f"[Debugger] Applying action: {user_action}")
                state_graph, mask, done, raw_payload = env.step(user_action)

                if done:
                    print("[Debugger] Game Over! Resetting for next match...")
                    cumulative_reports = []
                    state_graph, mask, raw_payload = env.reset()
        except Exception as e:
            print(f"[Debugger] Connection lost or reset: {e}")
            with state_lock:
                current_state = None
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
            time.sleep(2)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="MegaMek RL Interactive Web Debugger")
    arg_parser.add_argument("--port", type=int, default=5001, help="MegaMek RLDataPipeline port (default: 5001)")
    arg_parser.add_argument("--ui-port", type=int, default=5000, help="Flask Web UI port (default: 5000)")
    arg_parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint (.pt)")
    arg_parser.add_argument(
        "--spawn-server",
        action="store_true",
        help="Automatically spawn a local headless MegaMek match and connect to it",
    )
    arg_parser.add_argument("--server-port", type=int, default=24000, help="MegaMek server port when --spawn-server is used")
    args = arg_parser.parse_args()

    load_agent_checkpoint(args.model)
    env_port = args.port

    if args.spawn_server:
        env_port = args.server_port + 1000
        spawn_headless_megamek(mm_port=args.server_port)

    def handle_exit(signum, frame):
        terminate_spawned_server()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    threading.Thread(target=run_env, daemon=True).start()
    app.run(host="0.0.0.0", port=args.ui_port, debug=False)
