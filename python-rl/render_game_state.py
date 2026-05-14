import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from env import MegaMekEnvironment

def render_tensor_state(traj_path, step_idx=0, output_path="render.png"):
    if not os.path.exists(traj_path):
        print(f"File not found: {traj_path}")
        return
        
    print(f"Loading trajectory: {traj_path}")
    traj_data = torch.load(traj_path, map_location='cpu', weights_only=False)
    
    topology = traj_data.get("topology_payload", {})
    width = topology.get("width", 16)
    height = topology.get("height", 16)
    
    steps = traj_data.get("trajectories", [])
    if step_idx >= len(steps):
        print(f"Step {step_idx} out of bounds. Trajectory has {len(steps)} steps.")
        return
        
    data = steps[step_idx]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    context = getattr(data, 'context', ["Unknown"])[0] if hasattr(data, 'context') else "Unknown"
    ax.set_title(f"MegaMek Tensor State Render - Step {step_idx} - {context}")
    
    # Draw Hexes
    for y in range(height):
        for x in range(width):
            # Hex staggered math
            offset = 0.5 if x % 2 != 0 else 0
            px = x * 0.866
            py = -(y + offset)
            
            # Simple circle for hex
            circle = patches.Circle((px, py), radius=0.4, fill=False, color='lightgray', alpha=0.5)
            ax.add_patch(circle)
            
    # Extract Unit Data from PyTorch Tensors
    unit_x = data['unit'].x
    num_units = unit_x.size(0)
    
    unit_coords = []
    
    for i in range(num_units):
        ux = unit_x[i, 1].item()
        uy = unit_x[i, 2].item()
        
        offset = 0.5 if ux % 2 != 0 else 0
        px = ux * 0.866
        py = -(uy + offset)
        unit_coords.append((px, py))
        
        # Determine team (assume feature 0 is team/is_enemy flag, just guessing for visualization)
        is_enemy = unit_x[i, 0].item() > 0.5
        color = 'red' if is_enemy else 'blue'
        
        circle = patches.Circle((px, py), radius=0.3, fill=True, color=color, zorder=5)
        ax.add_patch(circle)
        ax.text(px, py, str(i), color='white', ha='center', va='center', zorder=6, fontweight='bold')
        
    # Draw Ephemeral Edges (LOS Threats)
    if ('unit', 'LOSThreat', 'hex') in data.edge_index_dict:
        los_edges = data['unit', 'LOSThreat', 'hex'].edge_index
        for i in range(los_edges.size(1)):
            u_idx = los_edges[0, i].item()
            h_idx = los_edges[1, i].item()
            
            hx = h_idx % width
            hy = h_idx // width
            
            hoffset = 0.5 if hx % 2 != 0 else 0
            hpx = hx * 0.866
            hpy = -(hy + hoffset)
            
            upx, upy = unit_coords[u_idx]
            
            # Draw line
            ax.plot([upx, hpx], [upy, hpy], color='orange', linestyle='--', alpha=0.3, zorder=2)
            
    # Draw Target Edges
    if ('unit', 'LOSTarget', 'unit') in data.edge_index_dict:
        tgt_edges = data['unit', 'LOSTarget', 'unit'].edge_index
        for i in range(tgt_edges.size(1)):
            u1_idx = tgt_edges[0, i].item()
            u2_idx = tgt_edges[1, i].item()
            
            upx1, upy1 = unit_coords[u1_idx]
            upx2, upy2 = unit_coords[u2_idx]
            
            ax.plot([upx1, upx2], [upy1, upy2], color='red', linestyle='-', linewidth=2, alpha=0.8, zorder=4)
            
    ax.set_aspect('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Render saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render_game_state.py <path_to_trajectory.pt> [step_idx]")
        sys.exit(1)
        
    traj = sys.argv[1]
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    render_tensor_state(traj, step)
