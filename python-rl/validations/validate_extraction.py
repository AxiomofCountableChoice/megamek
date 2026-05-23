import os
import sys
import time
import subprocess
import signal
import torch
import traceback
from env import MegaMekEnvironment

PORT = 4050
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
mm_data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "mm-data"))

def launch_server():
    print(f"Starting headless MegaMek on port {PORT} for validation...")
    cmd = [
        "megamek/build/install/MegaMek/bin/megamek",
        "-rlexport",
        "-autogen",
        "-randomMap",
        "-p1meks", os.path.join(mm_data_root, "data", "mekfiles", "meks", "3075", "Pariah Prime.mtf"),
        "-p2meks", os.path.join(mm_data_root, "data", "mekfiles", "meks", "Rec Guides ilClan", "Vol 33", "Gyrfalcon 5.mtf")
    ]
    env = os.environ.copy()
    env["RL_SERVER_PORT"] = str(PORT)
    env["SENTRY_DSN"] = ""
    log_path = os.path.join(os.path.dirname(__file__), "..", "megamek_server.log")
    log_file = open(log_path, "w")
    
    install_dir = os.path.join(repo_root, "megamek", "build", "install", "MegaMek")
    cmd[0] = "./bin/megamek"
    proc = subprocess.Popen(cmd, cwd=install_dir, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return proc

def validate_extraction():
    proc = launch_server()
    time.sleep(5)  # Give Java time to boot
    
    try:
        print("Connecting to environment...")
        env = MegaMekEnvironment(port=PORT + 1000, device='cpu', connect_on_init=True)
        print("Waiting for initial reset...")
        graph, mask, payload = env.reset()
        
        print("\n=== TOPOLOGY CACHE ===")
        print(f"Board Width: {env.board_width}")
        print(f"Hex Feature Dim: {env.feature_dims['hex']}")
        print(f"Unit Feature Dim: {env.feature_dims['unit']}")
        print(f"Weapon Feature Dim: {env.feature_dims['weapon']}")
        assert env.static_hex_features is not None, "Failed to cache static_hex_features"
        print(f"Static Hex Tensor Shape: {env.static_hex_features.shape}")
        
        print("\n=== STEPPING & LOGGING (5 STEPS) ===")
        for step_idx in range(5):
            print(f"\n--- STEP {step_idx + 1} ---")
            
            if graph is None:
                print("Game Over or State None.")
                break
                
            print(f"Context: {graph.context}")
            print(f"Global Context Tensor: {graph.global_context}")
            print(f"Hex Nodes: {graph['hex'].x.shape}")
            print(f"Unit Nodes: {graph['unit'].x.shape}")
            print(f"Weapon Nodes: {graph['weapon'].x.shape}")
            print(f"Action Nodes: {graph['action'].x.shape}")
            
            # Print Edges
            for edge_type in graph.edge_index_dict.keys():
                edges = graph[edge_type].edge_index
                print(f"Edge {edge_type}: {edges.shape}")
                
            # Perform assertions
            assert graph['hex'].x.size(1) == env.feature_dims['hex'], f"Hex feature dimension mismatch. Expected {env.feature_dims['hex']}, got {graph['hex'].x.size(1)}"
            assert graph['unit'].x.size(1) == env.feature_dims['unit'], f"Unit feature dimension mismatch."
            assert graph['weapon'].x.size(1) == env.feature_dims['weapon'], f"Weapon feature dimension mismatch."
            assert graph['action'].x.size(1) == 8, f"Action feature dimension mismatch. Expected 8, got {graph['action'].x.size(1)}"
            
            # Select random action
            action_dict = {"selected_path_index": -1}
            if graph.context.startswith("MOVEMENT"):
                valid_paths = mask.get("valid_paths", [])
                if valid_paths:
                    import random
                    action_dict = {"selected_path_index": random.randint(0, len(valid_paths) - 1)}
            elif graph.context.startswith("WEAPON") or graph.context.startswith("PHYSICAL"):
                # Simplistic fallback for attack phases to just "pass" for now in extraction test
                pass
                
            print(f"Selected action: {action_dict}")
            graph, mask, done, payload = env.step(action_dict)
            if done:
                print("Done flag returned.")
                break
                
    except Exception as e:
        print("\n!!! EXCEPTION CAUGHT !!!")
        traceback.print_exc()
        if 'payload' in locals() and payload is not None:
            print("\nLAST PAYLOAD KEYS:", payload.keys())
            if "state" in payload:
                state = payload["state"]
                print("STATE KEYS:", state.keys())
                for k, v in state.items():
                    if isinstance(v, list):
                        print(f" - {k} list size: {len(v)}")
        raise e
    finally:
        print("\nShutting down server...")
        proc.send_signal(signal.SIGTERM)
        subprocess.run(["pkill", "-f", "java.*MegaMek"], stderr=subprocess.DEVNULL)
        proc.wait(timeout=2)
        print("Done.")

if __name__ == "__main__":
    validate_extraction()
