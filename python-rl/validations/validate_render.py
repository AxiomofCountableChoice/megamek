import os
import glob
import subprocess
import sys

def validate_render():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rl_root = os.path.join(repo_root, "python-rl")
    
    # Try to find a trajectory
    candidate_dirs = [
        os.path.join(rl_root, "rl_sp_dataset"),
        os.path.join(rl_root, "rl_princess_dataset"),
        rl_root
    ]
    
    traj_file = None
    for d in candidate_dirs:
        if os.path.exists(d):
            # check directly or inside worker dirs
            pts = glob.glob(os.path.join(d, "*.pt"))
            if not pts:
                pts = glob.glob(os.path.join(d, "*", "*.pt"))
                
            for p in pts:
                if "agent" not in p and "master" not in p: # exclude model weights and master dataset block
                    traj_file = p
                    break
        if traj_file:
            break
            
    if not traj_file:
        # If we didn't find a worker trajectory, use master block if it exists
        master_file = os.path.join(rl_root, "bc_dataset_master.pt")
        if os.path.exists(master_file):
            traj_file = master_file
            
    if not traj_file:
        print("No trajectory files found to render. Run validate_random_agent.py first to generate some data.")
        return False
        
    print(f"Found trajectory file to test rendering: {traj_file}")
    
    # Run the render script
    render_script = os.path.join(rl_root, "render_game_state.py")
    output_png = os.path.join(rl_root, "render_val.png")
    
    cmd = [sys.executable, render_script, traj_file, "0"]
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        proc = subprocess.run(cmd, cwd=rl_root, capture_output=True, text=True, check=True)
        print("Render script completed successfully.")
        print("Stdout:", proc.stdout)
        
        if os.path.exists(output_png):
            print(f"SUCCESS: Render file created at {output_png}")
            return True
        else:
            print(f"FAILURE: Render file {output_png} not found despite successful execution.")
            return False
    except subprocess.CalledProcessError as e:
        print("Render script failed!")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        return False

if __name__ == "__main__":
    validate_render()
