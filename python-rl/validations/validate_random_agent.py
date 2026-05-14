import os
import sys
import time
import subprocess
import signal
import torch
import traceback
from env import MegaMekEnvironment
from models.agent import MegaMekAgent

PORT = 4051
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "megamek"))
mm_data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mm-data"))

def launch_server():
    print(f"Starting headless MegaMek on port {PORT} for random agent validation...")
    cmd = [
        "build/install/MegaMek/bin/megamek",
        "-rlexport",
        "-autogen",
        "-randomMap",
        "-p1meks", os.path.join(mm_data_root, "data", "mekfiles", "meks", "3075", "Pariah Prime.mtf"),
        "-p2meks", os.path.join(mm_data_root, "data", "mekfiles", "meks", "Rec Guides ilClan", "Vol 33", "Gyrfalcon 5.mtf")
    ]
    env = os.environ.copy()
    env["RL_SERVER_PORT"] = str(PORT)
    proc = subprocess.Popen(cmd, cwd=repo_root, env=env)
    return proc

def validate_random_agent():
    proc = launch_server()
    time.sleep(5)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Random Agent on {device}...")
        agent = MegaMekAgent(hidden_dim=128, ensemble_size=8).to(device)
        agent.eval()
        
        print("Connecting to environment...")
        env = MegaMekEnvironment(port=PORT + 1000, device=device, connect_on_init=True)
        graph, mask, payload = env.reset()
        
        print("\n=== STEPPING RANDOM AGENT (50 STEPS) ===")
        step_idx = 0
        while not getattr(env, "done", False) and step_idx < 50:
            if graph is None:
                print("Game Over or State None.")
                break
                
            print(f"Step {step_idx + 1}: Context = {graph.context}")
            
            with torch.no_grad():
                action_dict, v_mean, probs, mu_log_prob = agent.get_action(graph, mask, deterministic=False)
                
            print(f"  -> Agent Output: {action_dict}")
            print(f"  -> V_mean: {v_mean.item():.4f}, log_prob: {mu_log_prob:.4f}")
            
            graph, mask, done, payload = env.step(action_dict)
            if done:
                print("Done flag returned.")
                break
            
            step_idx += 1
            
        print(f"Random agent successfully executed {step_idx} steps without crashing.")
        
    except Exception as e:
        print("\n!!! EXCEPTION CAUGHT !!!")
        traceback.print_exc()
        raise e
    finally:
        print("\nShutting down server...")
        proc.send_signal(signal.SIGTERM)
        subprocess.run(["pkill", "-f", "java.*MegaMek"], stderr=subprocess.DEVNULL)
        proc.wait(timeout=2)
        print("Done.")

if __name__ == "__main__":
    validate_random_agent()
