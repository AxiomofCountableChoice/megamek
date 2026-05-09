import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from models.agent import MegaMekAgent
import os

def train():
    device = torch.device('cpu')
    print(f"Using compute device: {device}")
    
    import glob
    
    dataset_dir = 'bc_dataset'
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print(f"Dataset directory not found or empty at {dataset_dir}. Please run bc_generator.py first.")
        return
        
    dataset = []
    for f in glob.glob(os.path.join(dataset_dir, '*.pt')):
        match_data = torch.load(f, weights_only=False, map_location='cpu')
        dataset.extend(match_data.get('trajectories', []))
        
    print(f"Loaded {len(dataset)} trajectories across all matches for Behavioral Cloning.")
    
    # Feature Inspection
    for key in ['hex', 'unit', 'weapon', 'action']:
        all_x = []
        for data in dataset:
            if 'weapon' in data.node_types and hasattr(data['weapon'], 'x') and data['weapon'].x.size(0) > 0:
                data['weapon'].x[data['weapon'].x < -1000.0] = 0.0
            
            if key in data.node_types and hasattr(data[key], 'x') and data[key].x.size(0) > 0:
                all_x.append(data[key].x)
        if all_x:
            all_x = torch.cat(all_x, dim=0)
            print(f'[{key}] shape: {all_x.shape}, min: {all_x.min().item():.4f}, max: {all_x.max().item():.4f}, mean: {all_x.mean().item():.4f}')
        else:
            print(f'[{key}] empty')
            
    # We use batch_size=1 initially due to the dynamic length of the autoregressive action trees 
    # per trajectory without complex custom padding collators.
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    agent = MegaMekAgent(hidden_dim=128).to(device)
    optimizer = Adam(agent.parameters(), lr=1e-3)
    
    epochs = 10
    
    for epoch in range(epochs):
        agent.train()
        total_loss = 0.0
        valid_batches = 0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Requires ground-truth sequence target
            if getattr(batch, 'y_sequence', None) is None:
                continue
                
            y_seq = batch.y_sequence  # e.g. [true_a0, true_a1]
            seq_len = y_seq.size(0)
            
            # 1. Forward Graph Component
            z, x_dict = agent.encoder(batch)
            
            # 2. Extract valid tree options
            e_actions = agent.actor_pointer.compute_action_embeddings(x_dict, batch)
            
            if e_actions.size(0) == 0:
                continue
                
            step_indices = batch['action'].step_idx # Denotes which tier the action belongs to
            
            seq_loss = 0
            chosen_embeddings = []
            
            # 3. Autoregressive sequence execution (Teacher Forcing)
            for k in range(seq_len):
                target_idx = y_seq[k].item()
                
                # Dynamic Tier Masking based on Phase and Ground Truth Type
                if getattr(batch, 'context', [""])[0] == "MOVEMENT_BC":
                    valid_mask = (step_indices == k)
                else:
                    # Use the 5th dimension (node_type_flag) of the target to isolate candidates
                    target_type = batch['action'].x[target_idx, 4].item()
                    valid_mask = (batch['action'].x[:, 4] == target_type)
                    
                if not valid_mask.any():
                    break
                    
                if not valid_mask[target_idx]:
                    # Target node isn't in this tier (structural mismatch)
                    break 
                    
                # Get sequence context s_k
                if len(chosen_embeddings) == 0:
                    history_tensor = None
                else:
                    # history_tensor: (1, seq_len, hidden_dim) since batch_size=1
                    history_tensor = torch.stack(chosen_embeddings).unsqueeze(0) 
                    
                s_k = agent.actor_pointer.decode_sequence(z, history_tensor) # (1, hidden_dim)
                
                # Score all valid candidates in this tier
                # Using only candidates in tier k
                e_tier = e_actions[valid_mask]
                tier_batch_idx = batch['action'].batch[valid_mask] if hasattr(batch['action'], 'batch') and getattr(batch['action'], 'batch') is not None else None
                
                # Score
                logits = agent.actor_pointer.score_actions(s_k, e_tier, tier_batch_idx) # (num_tier_actions,)
                
                # The target_idx is global for the action tensor. We need its relative index in the tier.
                # (find the position of the True value in the masked slice)
                global_indices = torch.where(valid_mask)[0]
                relative_target = (global_indices == target_idx).nonzero(as_tuple=True)[0]
                
                if relative_target.numel() == 0:
                    break
                    
                # Cross Entropy Loss
                # F.cross_entropy expects logits (1, C) and target (1,)
                loss_k = F.cross_entropy(logits.unsqueeze(0), relative_target)
                seq_loss += loss_k
                
                # Teacher forcing: append the TRUE action embedding for the next sequence step
                chosen_embeddings.append(e_actions[target_idx])
                
            if seq_loss > 0:
                seq_loss.backward()
                optimizer.step()
                total_loss += seq_loss.item()
                valid_batches += 1
                
        avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} | Avg BC Loss: {avg_loss:.4f} | Samples: {valid_batches}")

    # Save the bootstrapped weights
    os.makedirs('models', exist_ok=True)
    save_path = 'models/bc_agent.pt'
    torch.save(agent.state_dict(), save_path)
    print(f"Saved trained BC model to {save_path}")

if __name__ == "__main__":
    train()
