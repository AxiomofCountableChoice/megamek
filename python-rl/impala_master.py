import os
import glob
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import random

from models.agent import MegaMekAgent
from env import MegaMekEnvironment

class ImpalaLearner:
    def __init__(self, device='cpu'):
        self.device = device
        self.agent = MegaMekAgent(hidden_dim=128, ensemble_size=8).to(self.device)
        self.optimizer = Adam(self.agent.parameters(), lr=1e-4)
        self.writer = SummaryWriter(log_dir="runs/impala_master")
        
        # Replay buffer: stores loaded trajectory dicts
        self.buffer = []
        self.max_buffer_size = 1000
        self.processed_files = set()
        
        self.global_step = 0
        
        # Load BC bootstrapping if impala_latest doesn't exist
        self.latest_model_path = os.path.join("models", "impala_agent_latest.pt")
        bc_model_path = os.path.join("models", "bc_agent.pt")
        
        os.makedirs("models", exist_ok=True)
        os.makedirs("data/trajectories", exist_ok=True)
        
        if os.path.exists(self.latest_model_path):
            print(f"Resuming from {self.latest_model_path}")
            self.agent.load_state_dict(torch.load(self.latest_model_path, map_location=device))
        elif os.path.exists(bc_model_path):
            print(f"Bootstrapping Learner from BC weights: {bc_model_path}")
            self.agent.load_state_dict(torch.load(bc_model_path, map_location=device))
            self.save_checkpoint()
            
        # Dummy environment to parse raw payloads
        self.env = MegaMekEnvironment(port=0, device=self.device, connect_on_init=False)
        self.env._connected = True # Bypass connection logic for offline parsing

    def save_checkpoint(self):
        torch.save(self.agent.state_dict(), self.latest_model_path)
        
    def load_new_trajectories(self):
        search_pattern = os.path.join("data", "trajectories", "*", "*.pt")
        files = glob.glob(search_pattern)
        
        new_files = [f for f in files if f not in self.processed_files]
        if not new_files:
            return 0
            
        # Sort by creation time to process sequentially
        new_files.sort(key=os.path.getctime)
        
        loaded = 0
        for file in new_files:
            try:
                traj_data = torch.load(file, map_location='cpu')
                self.buffer.append((file, traj_data))
                self.processed_files.add(file)
                loaded += 1
            except Exception as e:
                print(f"Failed to load {file}: {e}")
                
        # Trim buffer
        while len(self.buffer) > self.max_buffer_size:
            old_file, _ = self.buffer.pop(0)
            try:
                os.remove(old_file) # Optional: delete old files to save disk space
                self.processed_files.remove(old_file)
            except:
                pass
                
        return loaded
        
    def compute_vtrace_targets(self, rewards, v_means, v_vars, mu_probs, pi_probs, 
                               rho_bar=1.0, c_bar=1.0, gamma=0.99, lam_epistemic=1.0):
        seq_len = len(rewards)
        vs = torch.zeros(seq_len, device=self.device)
        
        rhos = torch.min(torch.tensor(rho_bar, device=self.device), pi_probs / mu_probs)
        cs = torch.min(torch.tensor(c_bar, device=self.device), pi_probs / mu_probs)
        
        sigma = torch.sqrt(v_vars + 1e-8)
        
        v_next = 0.0
        for t in reversed(range(seq_len)):
            v_mean_t = v_means[t]
            r_t = rewards[t]
            
            if t == seq_len - 1:
                v_mean_next = 0.0
            else:
                v_mean_next = v_means[t+1]
                
            delta_V = rhos[t] * (r_t + gamma * v_mean_next - v_mean_t)
            v_s = v_mean_t + delta_V + gamma * cs[t] * (v_next - v_mean_next)
            
            vs[t] = v_s
            v_next = v_s
            
        v_enhanced = vs + lam_epistemic * sigma
        return vs, v_enhanced, rhos

    def parse_state(self, raw_payload, topology_payload):
        # Inject topology if not already present
        if getattr(self.env, "static_hex_features", None) is None and topology_payload is not None:
            nodes = topology_payload.get("hex_nodes", [])
            edges = topology_payload.get("hex_edges", [])
            self.env.static_hex_features = torch.tensor(nodes, dtype=torch.float32).to(self.device) if nodes else torch.empty((0, 5), dtype=torch.float32, device=self.device)
            self.env.static_hex_adjacency_edges = torch.tensor(edges, dtype=torch.long).to(self.device) if edges else None

        state, mask = self.env._parse_to_heterodata(raw_payload)
        if state is not None:
            state = state.to(self.device)
        return state, mask

    def learn_step(self, batch_size=4, rho_bar=1.0, c_bar=1.0, gamma=0.99, lam_epistemic=1.0, entropy_coef=0.01):
        if len(self.buffer) < batch_size:
            return False
            
        self.agent.train()
        self.optimizer.zero_grad()
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        
        # Sample B trajectories (First coin-flip)
        sampled_trajs = random.sample(self.buffer, batch_size)
        
        for file_path, traj_data in sampled_trajs:
            topology = traj_data.get("topology_payload", None)
            steps = traj_data["steps"]
            seq_len = len(steps)
            if seq_len == 0: continue
            
            # Second coin-flip: sample a specific step_i to optimize
            # But to calculate V-Trace for step_i, we need forward pass from step_i to T
            # For simplicity, we run the forward pass for the whole trajectory, then slice it.
            step_i = random.randint(0, seq_len - 1)
            
            rewards = torch.tensor([s["reward"] for s in steps], device=self.device, dtype=torch.float32)
            mu_probs = torch.tensor([torch.exp(torch.tensor(s["mu_log_prob"])) for s in steps], device=self.device, dtype=torch.float32)
            
            pi_probs = torch.zeros(seq_len, device=self.device)
            v_means = torch.zeros(seq_len, device=self.device)
            v_vars = torch.zeros(seq_len, device=self.device)
            log_pis = torch.zeros(seq_len, device=self.device)
            entropies = torch.zeros(seq_len, device=self.device)
            v_preds_ensemble = []
            
            # Recompute graph forward pass using CURRENT weights
            for t, s in enumerate(steps):
                graph, mask = self.parse_state(s["raw_payload"], topology)
                idx = s["action_idx"]
                
                z, x_dict = self.agent.encoder(graph)
                v_preds = torch.stack([critic(z) for critic in self.agent.ensembles], dim=-1)
                v_preds_ensemble.append(v_preds.squeeze(0).squeeze(0))
                
                v_means[t] = v_preds.mean()
                v_vars[t] = v_preds.var(unbiased=False)
                
                e_actions = self.agent.actor_pointer.compute_action_embeddings(x_dict, graph)
                s_k = self.agent.actor_pointer.decode_sequence(z, None) 
                batch_idx = graph['action'].batch if hasattr(graph['action'], 'batch') and getattr(graph['action'], 'batch') is not None else None
                logits = self.agent.actor_pointer.score_actions(s_k, e_actions, batch_idx)
                
                probs = torch.softmax(logits, dim=-1)
                pi_probs[t] = probs[idx] + 1e-10
                log_pis[t] = torch.log(probs[idx] + 1e-10)
                entropies[t] = -(probs * torch.log(probs + 1e-10)).sum()
                
            vs, v_enhanced, rhos = self.compute_vtrace_targets(
                rewards, v_means.detach(), v_vars.detach(), mu_probs, pi_probs.detach(), 
                rho_bar, c_bar, gamma, lam_epistemic
            )
            
            # Compute loss only on the sampled step_i to satisfy the unbiased Double Coin-Flip strategy
            if step_i == seq_len - 1:
                v_enh_next = 0.0
            else:
                v_enh_next = v_enhanced[step_i+1]
                
            advantage = rewards[step_i] + gamma * v_enh_next - v_means[step_i].detach()
            actor_loss = -rhos[step_i] * log_pis[step_i] * advantage
            
            # Critic loss over the entire trajectory is generally fine for sample efficiency, 
            # but we can restrict to step_i for purity
            v_preds_tensor = v_preds_ensemble[step_i].unsqueeze(0) # [1, E]
            vs_target_expanded = vs[step_i].unsqueeze(0).expand(1, self.agent.ensembles.__len__()) # [1, E]
            critic_loss = nn.functional.mse_loss(v_preds_tensor, vs_target_expanded.detach())
            
            entropy_loss = -entropies[step_i]
            
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_entropy_loss += entropy_loss
            
            # Logging
            self.writer.add_scalar("VTrace/Advantage", advantage.item(), self.global_step)
            self.writer.add_scalar("VTrace/V_Mean", v_means[step_i].item(), self.global_step)
            self.writer.add_scalar("VTrace/Rho", rhos[step_i].item(), self.global_step)
            
        loss = (total_actor_loss + 0.5 * total_critic_loss + entropy_coef * total_entropy_loss) / batch_size
        loss.backward()
        self.optimizer.step()
        
        self.writer.add_scalar("Loss/Total", loss.item(), self.global_step)
        self.writer.add_scalar("Loss/Actor", (total_actor_loss/batch_size).item(), self.global_step)
        self.writer.add_scalar("Loss/Critic", (total_critic_loss/batch_size).item(), self.global_step)
        self.writer.add_scalar("Loss/Entropy", (total_entropy_loss/batch_size).item(), self.global_step)
        
        self.global_step += 1
        return True

    def run(self):
        print(f"Starting IMPALA Learner on {self.device}...")
        while True:
            loaded = self.load_new_trajectories()
            if loaded > 0:
                print(f"Loaded {loaded} new trajectories. Buffer size: {len(self.buffer)}")
                self.writer.add_scalar("System/Buffer_Size", len(self.buffer), self.global_step)
                
            # Perform optimization step if we have enough data
            if self.learn_step(batch_size=8):
                if self.global_step % 10 == 0:
                    self.save_checkpoint()
                    print(f"Step {self.global_step}: Saved checkpoint.")
            else:
                # Wait for workers to generate data
                time.sleep(2)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learner = ImpalaLearner(device=device)
    learner.run()
