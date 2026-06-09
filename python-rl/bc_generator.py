import os
import time
import argparse
import subprocess
import threading
import torch
import glob
import random
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

def collect_trajectories(port, target_trajectories, collected_dataset, dataset_lock):
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
        
    env.sock.settimeout(60.0)
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
                    env.static_hex_adjacency_edges = torch.tensor(edges, dtype=torch.long)
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
                        with dataset_lock:
                            collected_dataset.append(state_graph)
                        collected_this_episode += 1
                        
                        state_info = payload.get("state", {})
                        phase = state_info.get("phase_main", "UNKNOWN")
                        turn = state_info.get("turn_number", 0)
                        round_num = state_info.get("round_number", 0)
                        
                        if payload.get("context") in ["MOVEMENT_BC", "WEAPON_BC", "PHYSICAL_BC"]:
                            print(f"[Worker {port}] Round: {round_num} | Phase: {phase} | Turn: {turn} | Extracted {payload.get('context')}! Nodes: {state_graph['action'].x.shape}")
                        
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
    import uuid
    print(f"\n=======================")
    print(f"Initiating Episode {ep_idx+1} on port {server_port}")
    
    proc, p1_meks, p2_meks = run_megamek_episode(all_meks, server_port)
    
    dataset_lock = threading.Lock()
    local_dataset = []
    t1 = threading.Thread(target=collect_trajectories, args=(server_port + 1000, 1, local_dataset, dataset_lock), daemon=True)
    t2 = threading.Thread(target=collect_trajectories, args=(server_port + 1001, 1, local_dataset, dataset_lock), daemon=True)
    
    t1.start()
    t2.start()
    
    crashed = False
    try:
        import time
        while t1.is_alive() or t2.is_alive():
            if proc.poll() is not None:
                print(f"[bc_generator] Server crashed prematurely on port {server_port} (code {proc.poll()})")
                crashed = True
                break
            time.sleep(1)
    finally:
        print(f"[bc_generator] Terminating Headless Server on port {server_port}...")
        proc.terminate()
        proc.wait(timeout=5)
        if proc.poll() is None:
            proc.kill()
            
    if crashed:
        return False
    
    if local_dataset:
        os.makedirs("data/bc_trajectories", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"data/bc_trajectories/match_{ep_idx+1}_{timestamp}_{unique_id}.pt"
        
        gamelog_html = ""
        try:
            gamelog_path = os.path.join(os.path.dirname(__file__), "..", "megamek", "logs", f"server_{server_port}", "gamelog.html")
            if os.path.exists(gamelog_path):
                with open(gamelog_path, "r", encoding="utf-8") as f:
                    gamelog_html = f.read()
        except Exception as gle:
            print(f"Failed to read gamelog.html: {gle}")

        payload = {
            "metadata": {
                "p1_meks": p1_meks,
                "p2_meks": p2_meks,
                "timestamp": timestamp,
                "map": "Randomized",
                "gamelog_html": gamelog_html
            },
            "trajectories": local_dataset
        }
        torch.save(payload, filename)
        print(f"Episode {ep_idx+1} complete. Saved {len(local_dataset)} Trajectories to {filename}")
        return True
    else:
        print(f"Episode {ep_idx+1} complete, but no valid trajectories collected.")
        return False

if __name__ == "__main__":
            
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of random matches to generate")
    parser.add_argument("--save-path", type=str, default="bc_dataset_master.pt", help="File to serialize data")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel megamek processes")
    args = parser.parse_args()
    
    print("Pre-fetching all valid MTF Mek files...")
    all_mtf_files = glob.glob(os.path.join(MEKFILES_ROOT, "**", "*.mtf"), recursive=True)
    all_meks = [os.path.abspath(f) for f in all_mtf_files]
    print(f"Loaded {len(all_meks)} available mechs for procedural generation.")
    
    device = torch.device('cpu') # Always accumulate Dataset on CPU!
    print(f"Using compute device: {device} to avoid VRAM exhaustion")
    
    import concurrent.futures
    
    # Run in parallel using ProcessPoolExecutor
    max_workers = min(args.episodes, args.workers)
    base_port = 2346
    
    try:
        successful_ep = 0
        attempts = 0
        max_attempts = args.episodes * 5
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = set()
            
            # Initial batch
            while len(futures) < max_workers and attempts < max_attempts:
                futures.add(executor.submit(run_single_episode, attempts, base_port + 2 * attempts, all_meks))
                attempts += 1
                
            # Process as they complete
            while futures and successful_ep < args.episodes:
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                
                for future in done:
                    try:
                        success = future.result()
                        if success:
                            successful_ep += 1
                    except Exception as e:
                        print(f"Episode Exception: {e}")
                    
                    # Submit replacement task if needed
                    if successful_ep + len(futures) < args.episodes and attempts < max_attempts:
                        futures.add(executor.submit(run_single_episode, attempts, base_port + 2 * attempts, all_meks))
                        attempts += 1

    except KeyboardInterrupt:
        print("\n[bc_generator] Interrupted by user.")
    
    print(f"\n[Finished] Generation completed.")
