import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax
from models.encoder import MegaMekHANEncoder
from models.actor import ActionConditionedPointer

# Note: In a real environment, ensure torch_scatter is installed.

def train_bc():
    dataset_path = "bc_dataset_master.pt"
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found. Please run bc_generator.py first.")
        return
        
    print(f"Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path)
    print(f"Loaded {len(dataset)} trajectories.")
    
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    hidden_dim = 128
    encoder = MegaMekHANEncoder(hidden_dim=hidden_dim).to(device)
    actor = ActionConditionedPointer(hidden_dim=hidden_dim).to(device)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(actor.parameters()), lr=1e-3)
    
    epochs = 10
    
    for epoch in range(epochs):
        encoder.train()
        actor.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        total_graphs_processed = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass Encoder
            z, node_embeddings_dict = encoder(batch)
            
            e_action_all = actor.compute_action_embeddings(node_embeddings_dict, batch)
            
            ptr = batch['action'].ptr
            num_graphs = batch.num_graphs
            batch_indices = batch['action'].batch
            
            if not hasattr(batch, 'y_sequence') or batch.y_sequence.size(0) == 0:
                continue
                
            seq_len = batch.y_sequence.size(1)
            total_seq_loss = 0.0
            
            chosen_embeddings_history = None
            
            step_correct = 0
            step_total = 0
            
            for k in range(seq_len):
                # 1. Autoregressive Transformer Step
                s_k = actor.decode_sequence(z, chosen_embeddings_history)
                
                # 2. Mask candidates that belong to step k
                step_mask = (batch['action'].step_idx == k)
                valid_batch_indices = batch_indices[step_mask]
                valid_e_action = e_action_all[step_mask]
                
                if valid_e_action.size(0) == 0:
                    continue
                    
                # 3. Calculate logits for step k (Broadcast s_k to subset)
                s_k_expanded = s_k[valid_batch_indices] 
                logits_k = (s_k_expanded * valid_e_action).sum(dim=-1)
                
                # 4. Sub-graph Segmented Softmax
                action_probs_k = softmax(logits_k, valid_batch_indices, num_nodes=num_graphs)
                log_probs_k = torch.log(action_probs_k + 1e-8)
                
                # 5. Extract True Ground-Truth Action Target for k
                true_relative_idx_k = batch.y_sequence[:, k] 
                valid_graph_mask = (true_relative_idx_k != -1)
                
                if not valid_graph_mask.any():
                    continue
                    
                absolute_target_idx_k = ptr[:-1][valid_graph_mask] + true_relative_idx_k[valid_graph_mask]
                
                # Map absolute array index to the subset array index
                mapping = torch.cumsum(step_mask.long(), dim=0) - 1
                subset_target_idx_k = mapping[absolute_target_idx_k]
                
                chosen_log_probs = log_probs_k[subset_target_idx_k]
                loss_k = -chosen_log_probs.mean()
                total_seq_loss += loss_k
                
                # Stats
                preds = []
                # Reconstruct ptr for subset to safely argmax
                from torch_scatter import scatter_max
                max_logits, max_indices = scatter_max(logits_k, valid_batch_indices, dim=0, dim_size=num_graphs)
                
                for idx in range(num_graphs):
                    if valid_graph_mask[idx]:
                        pred_subset_idx = max_indices[idx].item()
                        if pred_subset_idx == subset_target_idx_k[valid_graph_mask[:idx].sum().item()]:
                            step_correct += 1
                        step_total += 1
                
                # 6. Accumulate Ground Truth representation for next step causal mask
                selected_e = torch.zeros(num_graphs, hidden_dim, device=device)
                selected_e[valid_graph_mask] = e_action_all[absolute_target_idx_k]
                
                selected_e = selected_e.unsqueeze(1)
                if chosen_embeddings_history is None:
                    chosen_embeddings_history = selected_e
                else:
                    chosen_embeddings_history = torch.cat([chosen_embeddings_history, selected_e], dim=1)

            if isinstance(total_seq_loss, float):
                continue
                
            total_seq_loss.backward()
            optimizer.step()
            
            total_loss += total_seq_loss.item() * num_graphs
            correct_predictions += step_correct
            total_samples += step_total
            total_graphs_processed += num_graphs
                
        avg_loss = total_loss / max(1, total_graphs_processed)
        acc = correct_predictions / max(1, total_samples) * 100.0
        
        print(f"Epoch {epoch+1}/{epochs} | Seq Loss: {avg_loss:.4f} | Sub-Action Acc: {acc:.2f}%")
        
    print("Training loop complete!")
    os.makedirs("weights", exist_ok=True)
    torch.save(encoder.state_dict(), "weights/encoder_bc.pt")
    torch.save(actor.state_dict(), "weights/actor_bc.pt")
    print("Saved weights.")

if __name__ == "__main__":
    train_bc()
