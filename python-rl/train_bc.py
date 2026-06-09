import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.tensorboard import SummaryWriter
from models.agent import MegaMekAgent
import os
import glob
import random
import math

class BCIterableDataset(IterableDataset):
    def __init__(self, file_paths, shuffle=False):
        self.file_paths = file_paths
        self.shuffle = shuffle
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files_to_process = self.file_paths
        else:
            per_worker = int(math.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            files_to_process = self.file_paths[worker_id * per_worker:(worker_id + 1) * per_worker]
            
        if self.shuffle:
            files_to_process = files_to_process.copy()
            random.shuffle(files_to_process)
            
        for f in files_to_process:
            try:
                match_data = torch.load(f, weights_only=False, map_location='cpu')
                trajectories = match_data.get('trajectories', [])
                if self.shuffle:
                    random.shuffle(trajectories)
                for data in trajectories:
                    if hasattr(data, 'y_sequence'):
                        data.y_len = torch.tensor([data.y_sequence.size(0)], dtype=torch.long)
                    else:
                        data.y_len = torch.tensor([0], dtype=torch.long)
                        
                    if hasattr(data, 'action_mask'):
                        delattr(data, 'action_mask')
                    if 'weapon' in data.node_types and hasattr(data['weapon'], 'x') and data['weapon'].x.size(0) > 0:
                        data['weapon'].x[data['weapon'].x < -1000.0] = 0.0
                        
                    yield data
            except Exception as e:
                print(f"Error loading {f}: {e}")

def process_batch(agent, batch, device, is_training=False, accumulation_steps=32, return_random_baseline=False):
    batch = batch.to(device)
        
    if return_random_baseline:
        pi_log_prob_total, entropy_total, v_mean, v_variance, correct_total, steps_total, random_correct_total = agent.evaluate_actions(batch, return_random_baseline=True)
    else:
        pi_log_prob_total, entropy_total, v_mean, v_variance, correct_total, steps_total = agent.evaluate_actions(batch, return_random_baseline=False)
        random_correct_total = None
        
    mask = (steps_total > 0)
    if not mask.any():
        if return_random_baseline:
            return 0.0, 0, 0, 0.0, False
        else:
            return 0.0, 0, 0, False
        
    loss = -pi_log_prob_total[mask].mean() / accumulation_steps
    
    if is_training:
        loss.backward()
        
    if return_random_baseline:
        return loss.item() * accumulation_steps, int(correct_total.sum().item()), int(steps_total.sum().item()), float(random_correct_total.sum().item()), True
    else:
        return loss.item() * accumulation_steps, int(correct_total.sum().item()), int(steps_total.sum().item()), True

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")
    
    dataset_dir = 'data/bc_trajectories'
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print(f"Dataset directory not found or empty at {dataset_dir}. Please run bc_generator.py first.")
        return
        
    all_files = glob.glob(os.path.join(dataset_dir, '*.pt'))
    all_files.sort()
    random.seed(42)
    random.shuffle(all_files)
    
    val_size = int(len(all_files) * 0.2)
    val_files = all_files[:val_size]
    train_files = all_files[val_size:]
    
    print(f"Dataset split by files: {len(train_files)} training files, {len(val_files)} validation files.")
    
    train_dataset = BCIterableDataset(train_files, shuffle=True)
    val_dataset = BCIterableDataset(val_files, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    writer = SummaryWriter(log_dir='runs/bc_training_logs')
    
    agent = MegaMekAgent(hidden_dim=128).to(device)
    optimizer = Adam(agent.parameters(), lr=3e-4) # Lowered LR
    
    accumulation_steps = 64
    
    epochs = 15
    os.makedirs('model_objects', exist_ok=True)
    save_path = 'model_objects/bc_agent.pt'
    
    for epoch in range(epochs):
        agent.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_steps = 0
        valid_train_batches = 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            loss_val, correct, steps, valid = process_batch(agent, batch, device, is_training=True, accumulation_steps=accumulation_steps)
            if valid:
                total_train_loss += loss_val
                total_train_correct += correct
                total_train_steps += steps
                valid_train_batches += 1
                
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
                
        avg_train_loss = total_train_loss / valid_train_batches if valid_train_batches > 0 else 0
        train_acc = total_train_correct / total_train_steps if total_train_steps > 0 else 0
        
        agent.eval()
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_random_correct = 0.0
        total_val_steps = 0
        valid_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                loss_val, correct, steps, rand_correct, valid = process_batch(agent, batch, device, is_training=False, return_random_baseline=True)
                if valid:
                    total_val_loss += loss_val
                    total_val_correct += correct
                    total_val_random_correct += rand_correct
                    total_val_steps += steps
                    valid_val_batches += 1
                    
        avg_val_loss = total_val_loss / valid_val_batches if valid_val_batches > 0 else 0
        val_acc = total_val_correct / total_val_steps if total_val_steps > 0 else 0
        val_rand_acc = total_val_random_correct / total_val_steps if total_val_steps > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} (Random Baseline Acc: {val_rand_acc:.4f})")
        
        writer.add_scalars('Loss', {'Train': avg_train_loss, 'Val': avg_val_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc, 'Val_Random': val_rand_acc}, epoch)
        
        torch.save(agent.state_dict(), save_path)
        
    writer.close()
    print(f"Saved trained BC model to {save_path}")

if __name__ == "__main__":
    train()
