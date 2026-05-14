import torch
import warnings

# Suppress PyG warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        data = torch.load("bc_dataset_master.pt", weights_only=False)
        print(f"Loaded {len(data)} graphs.")
        weapon_graphs = [g for g in data if g.get("context") == "WEAPON_BC"]
        movement_graphs = [g for g in data if g.get("context", "MOVEMENT_BC") == "MOVEMENT_BC"]
        print(f"Movement graphs: {len(movement_graphs)}")
        print(f"Weapon graphs: {len(weapon_graphs)}")
        
        for i, g in enumerate(weapon_graphs):
            print(f"\nWeapon Graph {i+1}:")
            print(f"  Action shape: {g['action'].x.shape}")
            print(f"  Target sequence: {g.y_sequence}")
            
    except Exception as e:
        print(f"Error: {e}")
