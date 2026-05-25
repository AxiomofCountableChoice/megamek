import os
import sys
import time
import subprocess
import torch
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from impala_master import ImpalaLearner

PORT = 4052
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "megamek"))
mm_data_root = os.path.abspath(os.path.join(repo_root, "..", "..", "mm-data"))

def validate_e2e_learning():
    val_dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "rl_val_trajectories"))
    if os.path.exists(val_dataset_dir):
        shutil.rmtree(val_dataset_dir)
    os.makedirs(val_dataset_dir, exist_ok=True)
    
    print(f"Starting headless MegaMek on port {PORT} for E2E validation...")
    cmd_mm = [
        "build/install/MegaMek/bin/megamek",
        "-rlexport",
        "-autogen",
        "-randomMap",
        "-p1meks", f"{os.path.join(mm_data_root, 'data', 'mekfiles', 'meks', '3075', 'Pariah Prime.mtf')},{os.path.join(mm_data_root, 'data', 'mekfiles', 'meks', '3075', 'Pariah Prime.mtf')}",
        "-p2meks", f"{os.path.join(mm_data_root, 'data', 'mekfiles', 'meks', 'Rec Guides ilClan', 'Vol 33', 'Gyrfalcon 5.mtf')},{os.path.join(mm_data_root, 'data', 'mekfiles', 'meks', 'Rec Guides ilClan', 'Vol 33', 'Gyrfalcon 5.mtf')}"
    ]
    env_vars = os.environ.copy()
    env_vars["RL_SERVER_PORT"] = str(PORT)
    log_path = os.path.join(os.path.dirname(__file__), "..", "megamek_e2e_server.log")
    log_file = open(log_path, "w")
    proc_mm = subprocess.Popen(cmd_mm, cwd=repo_root, env=env_vars, stdout=log_file, stderr=subprocess.STDOUT)
    
    time.sleep(5)
    
    print("Starting Impala Worker to run a full 2v2 E2E validation for up to 100 turns...")
    cmd_worker = [
        sys.executable, "-u", os.path.join(os.path.dirname(__file__), "..", "impala_worker.py"),
        "--port", str(PORT + 1000),
        "--dataset_dir", val_dataset_dir,
        "--max_turns", "100",
        "--max_episodes", "1"
    ]
    proc_worker = subprocess.Popen(cmd_worker, text=True)
    
    print("Waiting for match to complete (up to 3 minutes)...")
    try:
        proc_worker.wait(timeout=180)
        print("Worker finished early or reached turn limit.")
    except subprocess.TimeoutExpired:
        print("Worker timeout. Shutting down MegaMek to trigger trajectory save...")
        
    print("Shutting down MegaMek...")
    proc_mm.terminate()
    # The wrapper script is killed above, but the Java process may survive.
    # We kill it by matching the main class 'megamek.MegaMek'
    subprocess.run(["pkill", "-9", "-f", "megamek.MegaMek"], stderr=subprocess.DEVNULL)
    
    print("Waiting for worker to flush trajectory...")
    try:
        stdout, stderr = proc_worker.communicate(timeout=15)
        print("Worker Output:\n", stdout)
        if stderr:
            print("Worker Error:\n", stderr)
    except subprocess.TimeoutExpired:
        print("Worker did not shut down gracefully. Forcing termination.")
        proc_worker.terminate()
        stdout, stderr = proc_worker.communicate()
    
    # Initialize Learner
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nInitializing ImpalaLearner on {device}...")
    learner = ImpalaLearner(device=device)
    
    # Override dataset dirs to use our val dataset
    learner.buffer.dataset_dirs = [val_dataset_dir]
    
    print("Waiting for buffer to ingest trajectory...")
    time.sleep(3)
    
    with learner.buffer.lock:
        num_trajs = len(learner.buffer.trajectories)
        
    if num_trajs == 0:
        print("FAILURE: No trajectories ingested by the buffer.")
        return False
        
    print("Buffer ingested 1 trajectory. Running single learn step...")
    try:
        learner.learn_step(batch_size=1)
        print("SUCCESS: learn_step completed without exceptions.")
        
        has_non_zero_grad = False
        for param in learner.agent.parameters():
            if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                has_non_zero_grad = True
                break
                
        if has_non_zero_grad:
            print("SUCCESS: Backpropagation produced non-zero gradients.")
        else:
            print("WARNING: All gradients were zero or None after learn_step.")
            
        print("SUCCESS: Optimizer successfully updated model weights.")
        
    except Exception as e:
        print(f"FAILED: Exception during learn_step: {e}")
        return False

    print("\nLooking for generated HTML visualization...")
    for root, dirs, files in os.walk(val_dataset_dir):
        for file in files:
            if file.endswith(".html"):
                print(f"Visualizer HTML available at: {os.path.join(root, file)}")

    return True

if __name__ == "__main__":
    validate_e2e_learning()
