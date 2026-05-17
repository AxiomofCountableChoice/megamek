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
    val_dataset_dir = "rl_val_dataset"
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
    proc_mm = subprocess.Popen(cmd_mm, cwd=repo_root, env=env_vars)
    
    time.sleep(5)
    
    print("Starting Impala Worker to run a full 2v2 E2E validation for up to 10 turns...")
    cmd_worker = [
        sys.executable, os.path.join(os.path.dirname(__file__), "..", "impala_worker.py"),
        "--port", str(PORT + 1000),
        "--dataset_dir", val_dataset_dir,
        "--max_turns", "10"
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
    subprocess.run(["pkill", "-f", "java.*MegaMek"], stderr=subprocess.DEVNULL)
    
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
        
    print(f"Buffer ingested {num_trajs} trajectory. Running single learn step...")
    
    # Save parameters before step to check if they change
    params_before = [p.clone().detach() for p in learner.agent.parameters()]
    
    success = learner.learn_step(batch_size=1, sequence_length=8)
    
    if success:
        print("SUCCESS: learn_step completed without exceptions.")
        
        # Check gradients
        has_grads = False
        for name, p in learner.agent.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                if grad_norm > 0:
                    has_grads = True
                    break
                    
        if has_grads:
            print("SUCCESS: Backpropagation produced non-zero gradients.")
        else:
            print("FAILURE: Gradients are all zero or None.")
            return False
            
        # Check if weights updated
        weights_changed = False
        params_after = list(learner.agent.parameters())
        for pb, pa in zip(params_before, params_after):
            if not torch.equal(pb, pa):
                weights_changed = True
                break
                
        if weights_changed:
            print("SUCCESS: Optimizer successfully updated model weights.")
        else:
            print("FAILURE: Model weights did not change after optimizer.step().")
            return False
            
        return True
    else:
        print("FAILURE: learn_step returned False.")
        return False

if __name__ == "__main__":
    validate_e2e_learning()
