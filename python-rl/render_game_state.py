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
    
    steps = traj_data.get("steps", [])
    if step_idx >= len(steps):
        print(f"Step {step_idx} out of bounds. Trajectory has {len(steps)} steps.")
        return
        
    data_dict = steps[step_idx]
    raw_payload = data_dict.get("raw_payload", {})
    
    # Initialize environment just to use its parser
    env = MegaMekEnvironment(connect_on_init=False)
    
    # Inject topology features manually to satisfy parser
    nodes = topology.get("hex_nodes", [])
    env.static_hex_features = torch.tensor(nodes, dtype=torch.float32) if nodes else torch.empty((0, 5), dtype=torch.float32)
    
    # Parse into PyG HeteroData
    data, _ = env._parse_to_heterodata(raw_payload)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    context = raw_payload.get('context', "Unknown")
    ax.set_title(f"MegaMek Tensor State Render - Step {step_idx} - {context}")
    
    # Draw Hexes
    hex_features = env.static_hex_features
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            color = 'lightgray'
            if hex_features is not None and idx < hex_features.size(0):
                woods = hex_features[idx, 1].item()
                water = hex_features[idx, 2].item()
                swamp = hex_features[idx, 7].item()
                
                if woods > 0:
                    color = 'green'
                elif water > 0:
                    color = 'lightblue'
                elif swamp > 0:
                    color = 'purple'
                    
            # Hex staggered math
            offset = 0.5 if x % 2 != 0 else 0
            px = x * 0.866
            py = -(y + offset)
            
            # Simple circle for hex
            circle = patches.Circle((px, py), radius=0.4, fill=True if color != 'lightgray' else False, color=color, alpha=0.5 if color == 'lightgray' else 0.7)
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
            
    # Legend for terrain and edges
    import matplotlib.lines as mlines
    legend_elements = [
        patches.Patch(facecolor='green', alpha=0.7, label='Woods'),
        patches.Patch(facecolor='lightblue', alpha=0.7, label='Water'),
        patches.Patch(facecolor='purple', alpha=0.7, label='Swamp'),
        mlines.Line2D([], [], color='orange', linestyle='--', alpha=0.5, label='LOS Threat'),
        mlines.Line2D([], [], color='red', linestyle='-', linewidth=2, alpha=0.8, label='Target Edge'),
        patches.Circle((0,0), radius=0.3, fill=True, color='red', label='Enemy Unit'),
        patches.Circle((0,0), radius=0.3, fill=True, color='blue', label='Friendly Unit')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

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
