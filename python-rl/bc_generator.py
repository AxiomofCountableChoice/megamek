import os
import time
import signal
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

def run_megamek_episode(all_meks, server_port, max_meks=4, use_bv_balancer=True, min_bv=3000, max_bv=8000):
    p1_meks = ""
    p2_meks = ""
    
    cmd = [
        "build/install/MegaMek/bin/megamek", 
        "-rlexport", "-bcdatagen"
    ]
    
    if use_bv_balancer:
        target_bv = random.randint(min_bv, max_bv)
        print(f"[bc_generator] Starting Episode on Port {server_port} | Map=[Randomized] BV=[{target_bv}] MaxMeks=[{max_meks}]")
        cmd.extend([
            "-randomMap",
            "-randomBV", str(target_bv),
            "-maxMeks", str(max_meks)
        ])
        p1_meks = "procedural_bv_balanced"
        p2_meks = "procedural_bv_balanced"
    else:
        p1_num = random.randint(1, max_meks)
        p2_num = random.randint(1, max_meks)
        p1_meks = get_random_meks(all_meks, num=p1_num)
        p2_meks = get_random_meks(all_meks, num=p2_num)
        print(f"[bc_generator] Starting Episode on Port {server_port} | Map=[Randomized] P1=[{p1_meks}] | P2=[{p2_meks}]")
        cmd.extend([
            "-randomMap", 
            "-p1meks", p1_meks, 
            "-p2meks", p2_meks
        ])
        
    abs_midgame_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "midgame_saves"))
    os.makedirs(abs_midgame_dir, exist_ok=True)
    cmd.extend(["-autoSaveMidGame", abs_midgame_dir])
    
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "megamek"))
    
    # Isolate log4j files so multiple parallel instances don't throw NoSuchFileException
    env = os.environ.copy()
    
    # Force Java 21 to prevent UnsupportedClassVersionError (Java 65.0)
    java_home = os.environ.get("JAVA_HOME", os.path.expanduser("~/jdk-21.0.2"))
    env["JAVA_HOME"] = java_home
    env["PATH"] = f"{os.path.join(java_home, 'bin')}:{env.get('PATH', '')}"
    
    env["MEGAMEK_OPTS"] = f"-DlogPath=logs/server_{server_port}"
    env["RL_SERVER_PORT"] = str(server_port)
    
    # Launch subprocess in its own POSIX process group so JVM children can be terminated cleanly
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    
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
                env.state_parser.handle_topology(payload)
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

def run_single_episode(ep_idx, server_port, all_meks, max_meks=4, use_bv_balancer=True, min_bv=3000, max_bv=8000):
    import datetime
    import os
    import threading
    import torch
    import uuid
    print(f"\n=======================")
    print(f"Initiating Episode {ep_idx+1} on port {server_port}")
    
    proc, p1_meks, p2_meks = run_megamek_episode(all_meks, server_port, max_meks, use_bv_balancer, min_bv, max_bv)
    
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
        print(f"[bc_generator] Terminating Headless Server process group on port {server_port}...")
        if proc.poll() is None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    proc.kill()
            except OSError:
                proc.terminate()
            
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
        tmp_filename = filename + ".tmp"
        torch.save(payload, tmp_filename)
        os.replace(tmp_filename, filename)
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
    parser.add_argument("--max-meks", type=int, default=4, help="Maximum number of meks per side")
    parser.add_argument("--no-bv-balancer", action="store_true", help="Disable BV balancer (default: enabled)")
    parser.add_argument("--min-bv", type=int, default=None, help="Minimum BV for forces")
    parser.add_argument("--max-bv", type=int, default=None, help="Maximum BV for forces")
    args = parser.parse_args()
    
    if args.min_bv is None:
        args.min_bv = 1500 if args.max_meks <= 2 else 3000
    if args.max_bv is None:
        args.max_bv = 4500 if args.max_meks <= 2 else 8000
    
    use_bv_balancer = not args.no_bv_balancer
    
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
                futures.add(executor.submit(run_single_episode, attempts, base_port + 2 * attempts, all_meks, args.max_meks, use_bv_balancer, args.min_bv, args.max_bv))
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
                        futures.add(executor.submit(run_single_episode, attempts, base_port + 2 * attempts, all_meks, args.max_meks, use_bv_balancer, args.min_bv, args.max_bv))
                        attempts += 1

    except KeyboardInterrupt:
        print("\n[bc_generator] Interrupted by user.")
    
    print(f"\n[Finished] Generation completed.")
