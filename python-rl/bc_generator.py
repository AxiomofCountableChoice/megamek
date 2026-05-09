import os
import time
import argparse
import subprocess
import threading
import torch
import glob
import random
import numpy as np
import concurrent.futures

from env import MegaMekEnvironment

# Set up relative bounds for meks
MEKFILES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mm-data", "data", "mekfiles", "meks"))

def get_random_meks(all_meks, num=1):
    if not all_meks:
        return ""
    return ",".join(random.choices(all_meks, k=num))

def run_megamek_episode(all_meks, server_port):
    p1_meks = get_random_meks(all_meks, num=4)
    p2_meks = get_random_meks(all_meks, num=4)
    
    print(f"[bc_generator] Starting Episode on Port {server_port} | Map=[Randomized] P1=[{p1_meks}] | P2=[{p2_meks}]")
    
    cmd = [
        "build/install/MegaMek/bin/megamek", 
        "-rlexport", "-bcdatagen", "-randomMap", 
        "-p1meks", p1_meks, 
        "-p2meks", p2_meks
    ]
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "megamek"))
    
    # Isolate log4j files so multiple parallel instances don't throw NoSuchFileException
    env = os.environ.copy()
    
    # Force Java 21 to prevent UnsupportedClassVersionError (Java 65.0)
    java_home = os.environ.get("JAVA_HOME", "/home/stuart_hatzioannou/jdk-21.0.2")
    env["JAVA_HOME"] = java_home
    env["PATH"] = f"{os.path.join(java_home, 'bin')}:{env.get('PATH', '')}"
    
    env["MEGAMEK_OPTS"] = f"-DlogPath=logs/server_{server_port}"
    env["RL_SERVER_PORT"] = str(server_port)
    
    # Launch subprocess. Wait for it to boot.
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Start a thread to stream server output to python stdout
    import threading
    def stream_output(pipe):
        for line in iter(pipe.readline, ''):
            print(f"[MegaMek Server] {line}", end='')
    threading.Thread(target=stream_output, args=(proc.stdout,), daemon=True).start()
    
    return proc, p1_meks, p2_meks

def collect_trajectories(port, target_trajectories, collected_dataset):
    env = MegaMekEnvironment(port=port)
    retries = 60
    
    # Simple Wait loop
    while retries > 0:
        try:
            env.connect()
            break
        except ConnectionRefusedError:
            time.sleep(2)
            retries -= 1
            
    if not env.sock:
        print(f"[Worker {port}] Failed to connect after retries.")
        return 0
        
    print(f"[Worker {port}] Connected to socket. Waiting for trajectories...")
    collected_this_episode = 0
    
    while True:
        try:
            payload = env._receive_payload()
            if payload is None:
                print(f"[Worker {port}] Disconnected or Match Over.")
                break
                
            if payload.get("context") == "START_GAME":
                continue
                
            if payload.get("context") == "TOPOLOGY":
                nodes = payload.get("hex_nodes", [])
                edges = payload.get("hex_edges", [])
                if nodes:
                    env.static_hex_features = torch.tensor(nodes, dtype=torch.float32)
                else:
                    env.static_hex_features = torch.empty((0, 5), dtype=torch.float32)
                if edges:
                    env.static_hex_adjacency_edges = np.array(edges, dtype=np.int64)
                else:
                    env.static_hex_adjacency_edges = torch.empty((2, 0), dtype=torch.long)
                continue
                
            if payload.get("context") in ["MOVEMENT_BC", "WEAPON_BC", "PHYSICAL_BC"]:
                # Ensure all parsing remains on CPU for dataset preparation
                state_graph, mask = env._parse_to_heterodata(payload)
                if state_graph is not None:
                    # state_graph is natively CPU in env.py
                    # env.py automatically computes target tree subset mappings
                    if getattr(state_graph, 'y_sequence', None) is not None:
                        collected_dataset.append(state_graph)
                        collected_this_episode += 1
                        
                        if payload.get("context") in ["WEAPON_BC", "PHYSICAL_BC"]:
                            print(f"[Worker {port}] Successfully extracted {payload.get('context')}! Nodes: {state_graph['action'].x.shape}")
                        
                        if len(collected_dataset) % 10 == 0:
                            print(f"[Worker {port}] Global Collection count: {len(collected_dataset)} valid actions...")
        except Exception as e:
            print(f"[Worker {port}] Error parsing: {e}")
            break
            
    return collected_this_episode

def run_single_episode(ep_idx, server_port, all_meks):
    import datetime
    import os
    import threading
    import torch
    print(f"\n=======================")
    print(f"Initiating Episode {ep_idx+1} on port {server_port}")
    
    proc, p1_meks, p2_meks = run_megamek_episode(all_meks, server_port)
    
    local_dataset = []
    t1 = threading.Thread(target=collect_trajectories, args=(server_port + 1000, 1, local_dataset), daemon=True)
    t2 = threading.Thread(target=collect_trajectories, args=(server_port + 1001, 1, local_dataset), daemon=True)
    
    t1.start()
    t2.start()
    
    try:
        # Let matches naturally conclude. No timeout.
        t1.join()
        t2.join()
    finally:
        print(f"[bc_generator] Terminating Headless Server on port {server_port}...")
        proc.terminate()
        proc.wait(timeout=5)
        if proc.poll() is None:
            proc.kill()
    
    if local_dataset:
        os.makedirs("bc_dataset", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bc_dataset/match_{ep_idx+1}_{timestamp}.pt"
        
        payload = {
            "metadata": {
                "p1_meks": p1_meks,
                "p2_meks": p2_meks,
                "timestamp": timestamp,
                "map": "Randomized"
            },
            "trajectories": local_dataset
        }
        torch.save(payload, filename)
        print(f"Episode {ep_idx+1} complete. Saved {len(local_dataset)} Trajectories to {filename}")
    else:
        print(f"Episode {ep_idx+1} complete, but no valid trajectories collected.")

if __name__ == "__main__":
            
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of random matches to generate")
    parser.add_argument("--save-path", type=str, default="bc_dataset_master.pt", help="File to serialize data")
    args = parser.parse_args()
    
    print("Pre-fetching all valid MTF Mek files...")
    all_mtf_files = glob.glob(os.path.join(MEKFILES_ROOT, "**", "*.mtf"), recursive=True)
    all_meks = [os.path.abspath(f) for f in all_mtf_files]
    print(f"Loaded {len(all_meks)} available mechs for procedural generation.")
    
    device = torch.device('cpu') # Always accumulate Dataset on CPU!
    print(f"Using compute device: {device} to avoid VRAM exhaustion")
    
    # Run in parallel using ProcessPoolExecutor
    max_workers = min(args.episodes, 4) # cap at 4 parallel matches
    base_port = 2346
    
    try:
        for ep in range(args.episodes):
            try:
                run_single_episode(ep, base_port + ep, all_meks)
            except Exception as e:
                print(f"Episode Exception: {e}")
    except KeyboardInterrupt:
        print("\n[bc_generator] Interrupted by user.")
    
    print(f"\n[Finished] Generation completed.")
