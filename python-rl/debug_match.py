import os
import time
import subprocess
import threading
import torch

from env import MegaMekEnvironment

MEKFILES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "megamek", "data", "mekfiles", "meks"))

def run_headless():
    cmd = [
        "build/install/MegaMek/bin/megamek", 
        "-rlexport", "-port", "3456", "-autogen", "-randomMap", 
        "-p1meks", "testresources/megamek/common/units/Charger C.mtf",
        "-p2meks", "testresources/megamek/common/units/Sagittaire SGT-14D.mtf"
    ]
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "megamek"))
    
    print("[debug] Starting Headless MegaMek Server...")
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Start a thread to stream server output to python stdout
    def stream_output(pipe):
        for line in iter(pipe.readline, ''):
            print(f"[MegaMek] {line}", end='')
    threading.Thread(target=stream_output, args=(proc.stdout,), daemon=True).start()
    
    return proc

def debug_worker(port, worker_id):
    env = MegaMekEnvironment(port=port)
    retries = 15
    
    print(f"[Worker {port}] Attempting to connect...")
    while retries > 0:
        try:
            env.connect()
            break
        except ConnectionRefusedError:
            time.sleep(2)
            retries -= 1
            
    if not env.sock:
        print(f"[Worker {port}] Failed to connect.")
        return
        
    print(f"[Worker {port}] Connected! Awaiting graph data...")
    
    state_graph, mask = env.reset()
    if state_graph is None:
        print(f"[Worker {port}] Failed to get initial state.")
        return
        
    print(f"\n[Worker {port}] --- INITIAL STATE ---")
    print(f"Mask Size: {len(mask)}")
    print("Graph Composition:")
    for node_type in state_graph.node_types:
        if getattr(state_graph[node_type], 'x', None) is not None:
            print(f"  {node_type} nodes: {state_graph[node_type].x.size()}")
    for edge_type in state_graph.edge_types:
        if getattr(state_graph[edge_type], 'edge_index', None) is not None:
            count = state_graph[edge_type].edge_index.size(1)
            print(f"  {edge_type} edges: {count}")
            
    step_count = 0
    while step_count < 25:
        # Take a dummy action to trigger next step
        action_dict = {"selected_path_index": 0} if len(mask) > 0 else {"selected_path_index": -1}
        state_graph, mask, done = env.step(action_dict)
        
        if done:
            print(f"[Worker {port}] Match done.")
            break
            
        step_count += 1
        print(f"\n[Worker {port}] --- ACTION STEP {step_count} ---")
        print(f"Mask Size: {len(mask)}")
        for node_type in state_graph.node_types:
            if getattr(state_graph[node_type], 'x', None) is not None:
                print(f"  {node_type} nodes: {state_graph[node_type].x.size()}")
        for edge_type in state_graph.edge_types:
            if getattr(state_graph[edge_type], 'edge_index', None) is not None:
                count = state_graph[edge_type].edge_index.size(1)
                if count > 0:
                    print(f"  {edge_type} edges: {count}")

if __name__ == "__main__":
    proc = run_headless()
    try:
        t1 = threading.Thread(target=debug_worker, args=(8001, 1))
        t2 = threading.Thread(target=debug_worker, args=(8002, 2))
        t1.start()
        t2.start()
        t1.join(timeout=180)
        t2.join(timeout=180)
    finally:
        print("[debug] Terminating Server...")
        proc.terminate()
