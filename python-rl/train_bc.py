import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from models.agent import MegaMekAgent
import os
import glob

def process_batch(agent, batch, device, optimizer=None):
    batch = batch.to(device)
    if optimizer:
        optimizer.zero_grad()
        
    if getattr(batch, 'y_sequence', None) is None:
        return 0.0, 0, 0, False
        
    y_seq = batch.y_sequence
    seq_len = y_seq.size(0)
    
    z, x_dict = agent.encoder(batch)
    e_actions = agent.actor_pointer.compute_action_embeddings(x_dict, batch)
    
    if e_actions.size(0) == 0:
        return 0.0, 0, 0, False
        
    step_indices = batch['action'].step_idx
    
    seq_loss = 0
    correct = 0
    steps = 0
    chosen_embeddings = []
    
    for k in range(seq_len):
        target_idx = y_seq[k].item()
        
        if getattr(batch, 'context', [""])[0] == "MOVEMENT_BC":
            valid_mask = (step_indices == k)
        else:
            target_type = batch['action'].x[target_idx, 4].item()
            valid_mask = (batch['action'].x[:, 4] == target_type)
            
        if not valid_mask.any() or not valid_mask[target_idx]:
            break
            
        history_tensor = None if len(chosen_embeddings) == 0 else torch.stack(chosen_embeddings).unsqueeze(0)
        s_k = agent.actor_pointer.decode_sequence(z, history_tensor)
        
        e_tier = e_actions[valid_mask]
        tier_batch_idx = batch['action'].batch[valid_mask] if hasattr(batch['action'], 'batch') and getattr(batch['action'], 'batch') is not None else None
        
        logits = agent.actor_pointer.score_actions(s_k, e_tier, tier_batch_idx)
        
        global_indices = torch.where(valid_mask)[0]
        relative_target = (global_indices == target_idx).nonzero(as_tuple=True)[0]
        
        if relative_target.numel() == 0:
            break
            
        loss_k = F.cross_entropy(logits.unsqueeze(0), relative_target)
        seq_loss += loss_k
        
        if logits.argmax(dim=-1) == relative_target[0]:
            correct += 1
        steps += 1
        
        chosen_embeddings.append(e_actions[target_idx])
        
    if type(seq_loss) != int and seq_loss > 0:
        if optimizer:
            seq_loss.backward()
            optimizer.step()
        return seq_loss.item(), correct, steps, True
        
    return 0.0, 0, 0, False

def train():
    device = torch.device('cpu')
    print(f"Using compute device: {device}")
    
    dataset_dir = 'bc_dataset'
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print(f"Dataset directory not found or empty at {dataset_dir}. Please run bc_generator.py first.")
        return
        
    dataset = []
    for f in glob.glob(os.path.join(dataset_dir, '*.pt')):
        match_data = torch.load(f, weights_only=False, map_location='cpu')
        dataset.extend(match_data.get('trajectories', []))
        
    print(f"Loaded {len(dataset)} trajectories across all matches for Behavioral Cloning.")
    
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
            
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples.")
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    writer = SummaryWriter(log_dir='runs/bc_training_logs')
    
    agent = MegaMekAgent(hidden_dim=128).to(device)
    optimizer = Adam(agent.parameters(), lr=1e-3)
    
    epochs = 15
    os.makedirs('models', exist_ok=True)
    save_path = 'models/bc_agent.pt'
    
    for epoch in range(epochs):
        agent.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_steps = 0
        valid_train_batches = 0
        
        for batch in train_loader:
            loss_val, correct, steps, valid = process_batch(agent, batch, device, optimizer)
            if valid:
                total_train_loss += loss_val
                total_train_correct += correct
                total_train_steps += steps
                valid_train_batches += 1
                
        avg_train_loss = total_train_loss / valid_train_batches if valid_train_batches > 0 else 0
        train_acc = total_train_correct / total_train_steps if total_train_steps > 0 else 0
        
        agent.eval()
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_steps = 0
        valid_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                loss_val, correct, steps, valid = process_batch(agent, batch, device, None)
                if valid:
                    total_val_loss += loss_val
                    total_val_correct += correct
                    total_val_steps += steps
                    valid_val_batches += 1
                    
        avg_val_loss = total_val_loss / valid_val_batches if valid_val_batches > 0 else 0
        val_acc = total_val_correct / total_val_steps if total_val_steps > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")
        
        writer.add_scalars('Loss', {'Train': avg_train_loss, 'Val': avg_val_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch)
        
        torch.save(agent.state_dict(), save_path)
        
    writer.close()
    print(f"Saved trained BC model to {save_path}")

if __name__ == "__main__":
    train()
