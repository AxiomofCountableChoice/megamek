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
    p1_meks = get_random_meks(all_meks, num=1)
    p2_meks = get_random_meks(all_meks, num=1)
    
    print(f"[bc_generator] Starting Episode on Port {server_port} | Map=[Randomized] P1=[{p1_meks}] | P2=[{p2_meks}]")
    
    cmd = [
        "build/install/MegaMek/bin/megamek", 
        "-rlexport", "-autogen", "-randomMap", 
        "-p1meks", p1_meks, 
        "-p2meks", p2_meks
    ]
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "megamek"))
    
    # Isolate log4j files so multiple parallel instances don't throw NoSuchFileException
    env = os.environ.copy()
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
    
    return proc

def collect_trajectories(port, target_trajectories, collected_dataset):
    env = MegaMekEnvironment(port=port)
    retries = 10
    
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
                
            if payload.get("context") == "TOPOLOGY":
                nodes = payload.get("hex_nodes", [])
                edges = payload.get("hex_edges", [])
                if nodes:
                    env.static_hex_features = torch.tensor(nodes, dtype=torch.float32)
                else:
                    env.static_hex_features = torch.empty((0, 5), dtype=torch.float32)
                if edges:
                    edge_array = np.array(edges, dtype=np.int64).T
                    env.static_hex_adjacency_edges = torch.tensor(edge_array, dtype=torch.long)
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
    
    master_dataset = []
    dataset_lock = threading.Lock()
    
    def run_single_episode(ep_idx, server_port):
        print(f"\n=======================")
        print(f"Initiating Episode {ep_idx+1}/{args.episodes} on port {server_port}")
        
        proc = run_megamek_episode(all_meks, server_port)
        
        local_dataset = []
        t1 = threading.Thread(target=collect_trajectories, args=(server_port + 1000, 1, local_dataset))
        t2 = threading.Thread(target=collect_trajectories, args=(server_port + 1001, 1, local_dataset))
        
        t1.start()
        t2.start()
        
        try:
            t1.join(timeout=180) # 3 min max
            t2.join(timeout=180)
        finally:
            print(f"[bc_generator] Terminating Headless Server on port {server_port}...")
            proc.terminate()
            proc.wait(timeout=5)
            if proc.poll() is None:
                proc.kill()
        
        with dataset_lock:
            master_dataset.extend(local_dataset)
            print(f"Episode {ep_idx+1} complete. Total Trajectories so far: {len(master_dataset)}")
            
    # Run in parallel using ThreadPoolExecutor
    max_workers = min(args.episodes, 4) # cap at 4 parallel matches
    base_port = 2346
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ep in range(args.episodes):
            futures.append(executor.submit(run_single_episode, ep, base_port + ep))
        concurrent.futures.wait(futures)
    
    # Save dataset natively
    if master_dataset:
        torch.save(master_dataset, args.save_path)
        print(f"\n[Finished] Serialized {len(master_dataset)} graphs to {args.save_path}.")
    else:
        print("[Warning] No valid trajectories collected.")
