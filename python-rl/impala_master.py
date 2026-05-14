import os
import glob
import time
import threading
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import random

from models.agent import MegaMekAgent
from env import MegaMekEnvironment

class ImpalaReplayBuffer:
    """
    Monitors dataset directories for new trajectories, loads them into memory,
    and removes the processed files.
    """
    def __init__(self, dataset_dirs=["rl_sp_dataset", "rl_princess_dataset"], max_trajectories=1000):
        self.dataset_dirs = dataset_dirs
        self.max_trajectories = max_trajectories
        self.trajectories = []
        self.lock = threading.Lock()
        
        # Start background worker to ingest trajectories
        self.ingest_thread = threading.Thread(target=self._ingest_loop, daemon=True)
        self.ingest_thread.start()
        
    def _ingest_loop(self):
        while True:
            for d in self.dataset_dirs:
                if not os.path.exists(d):
                    continue
                # Find all worker directories
                worker_dirs = glob.glob(os.path.join(d, "*"))
                for wd in worker_dirs:
                    if not os.path.isdir(wd): continue
                    
                    files = glob.glob(os.path.join(wd, "*.pt"))
                    for f in files:
                        try:
                            data = torch.load(f, weights_only=False, map_location='cpu')
                            with self.lock:
                                self.trajectories.append(data)
                                if len(self.trajectories) > self.max_trajectories:
                                    self.trajectories.pop(0) # FIFO
                            # Archive or delete
                            os.remove(f)
                        except Exception as e:
                            print(f"[ReplayBuffer] Error loading {f}: {e}")
            time.sleep(2)

    def sample_batch(self, batch_size=4, sequence_length=16):
        """
        Double Coin-Flip Strategy:
        1. Uniformly sample B trajectories.
        2. Uniformly sample a starting step to get a chunk of sequence_length.
        """
        with self.lock:
            if len(self.trajectories) < batch_size:
                return None
                
            # Coin-flip 1: sample trajectories
            sampled_trajs = random.sample(self.trajectories, batch_size)
            
        batch_data = []
        for traj in sampled_trajs:
            topology = traj.get("topology_payload", None)
            steps = traj.get("steps", [])
            total_reward = sum([s.get("reward", 0.0) for s in steps])
            
            if len(steps) == 0:
                continue
                
            if len(steps) <= sequence_length + 1:
                sampled_steps = steps
            else:
                # Coin-flip 2: sample start step
                max_start = len(steps) - (sequence_length + 1)
                start_idx = random.randint(0, max_start)
                sampled_steps = steps[start_idx : start_idx + sequence_length + 1]
                
            batch_data.append((topology, sampled_steps, total_reward))
            
        return batch_data

class ImpalaLearner:
    def __init__(self, device='cpu'):
        self.device = device
        self.agent = MegaMekAgent(hidden_dim=128, ensemble_size=8).to(self.device)
        self.optimizer = Adam(self.agent.parameters(), lr=1e-4)
        self.writer = SummaryWriter(log_dir="runs/impala_master")
        
        self.buffer = ImpalaReplayBuffer()
        
        self.global_step = 0
        
        # Load BC bootstrapping if impala_latest doesn't exist
        self.latest_model_path = os.path.join("models", "impala_agent_latest.pt")
        bc_model_path = os.path.join("models", "bc_agent.pt")
        
        os.makedirs("models", exist_ok=True)
        
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
        
    @torch.no_grad()
    def compute_vtrace_targets(self, rewards, v_means, v_vars, mu_probs, pi_probs, 
                               rho_bar=1.0, c_bar=1.0, gamma=0.99, lam_epistemic=1.0):
        seq_len = len(rewards)
        vs = torch.zeros(seq_len, device=self.device)
        
        rhos = torch.min(torch.tensor(rho_bar, device=self.device), pi_probs / mu_probs)
        cs = torch.min(torch.tensor(c_bar, device=self.device), pi_probs / mu_probs)
        
        sigma = torch.sqrt(v_vars[:-1] + 1e-8)
        
        # Precompute v_mean_next and delta_V using vectorized operations
        v_means_next = v_means[1:]
        delta_Vs = rhos * (rewards + gamma * v_means_next - v_means[:-1])
        
        # True bootstrap value from the N+1'th state
        v_next = v_means[-1]
        for t in reversed(range(seq_len)):
            v_s = v_means[t] + delta_Vs[t] + gamma * cs[t] * (v_next - v_means_next[t])
            
            vs[t] = v_s
            v_next = v_s
            
        v_enhanced = vs + lam_epistemic * sigma
        return vs, v_enhanced, rhos

    def parse_state(self, raw_payload, target_action, topology_payload):
        # Inject target_action back into payload so _parse_to_heterodata reconstructs y_sequence
        raw_payload["target_action"] = target_action
        
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

    def learn_step(self, batch_size=4, sequence_length=16, rho_bar=1.0, c_bar=1.0, gamma=0.99, lam_epistemic=1.0, entropy_coef=0.01):
        batch_data = self.buffer.sample_batch(batch_size=batch_size, sequence_length=sequence_length)
        if not batch_data:
            return False
            
        self.agent.train()
        self.optimizer.zero_grad()
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        valid_batches = 0
        total_game_rewards = []
        
        for topology, steps, total_reward in batch_data:
            actual_seq_len = len(steps) - 1
            if actual_seq_len <= 0: continue
            
            valid_batches += 1
            total_game_rewards.append(total_reward)
            
            transitions = steps[:-1]
            
            rewards = torch.tensor([s["reward"] for s in transitions], device=self.device, dtype=torch.float32)
            mu_probs = torch.tensor([torch.exp(torch.tensor(s["mu_log_prob"])) for s in transitions], device=self.device, dtype=torch.float32)
            
            pi_probs = torch.zeros(actual_seq_len, device=self.device)
            log_pis = torch.zeros(actual_seq_len, device=self.device)
            entropies = torch.zeros(actual_seq_len, device=self.device)
            
            v_means = torch.zeros(actual_seq_len + 1, device=self.device)
            v_vars = torch.zeros(actual_seq_len + 1, device=self.device)
            
            v_preds_ensemble = []
            
            # Recompute graph forward pass using CURRENT weights for ALL N+1 states
            for t, s in enumerate(steps):
                graph, mask = self.parse_state(s["raw_payload"], s.get("action_dict", {}), topology)
                
                pi_log_prob, entropy, v_mean, v_variance = self.agent.evaluate_actions(graph)
                
                v_means[t] = v_mean[0] if v_mean.numel() > 0 else 0.0
                v_vars[t] = v_variance[0] if v_variance.numel() > 0 else 0.0
                
                # Only extract actor targets for the N transitions
                if t < actual_seq_len:
                    pi_probs[t] = torch.exp(pi_log_prob[0]) if pi_log_prob.numel() > 0 else 1.0
                    log_pis[t] = pi_log_prob[0] if pi_log_prob.numel() > 0 else 0.0
                    entropies[t] = entropy[0] if entropy.numel() > 0 else 0.0
                    
                    z, _ = self.agent.encoder(graph)
                    v_preds = torch.stack([critic(z) for critic in self.agent.ensembles], dim=-1).squeeze(0).squeeze(0) # [E]
                    v_preds_ensemble.append(v_preds)
                
            vs, v_enhanced, rhos = self.compute_vtrace_targets(
                rewards, v_means.detach(), v_vars.detach(), mu_probs, pi_probs.detach(), 
                rho_bar, c_bar, gamma, lam_epistemic
            )
            
            # Vectorized sequence loss computation
            # Compute true bootstrap for Nth state including epistemic variance
            v_enh_last = v_means[-1:] + lam_epistemic * torch.sqrt(v_vars[-1:] + 1e-8)
            v_enh_next = torch.cat([v_enhanced[1:], v_enh_last.detach()])
            advantages = rewards + gamma * v_enh_next - v_means[:-1].detach()
            
            # Actor Loss: sum over sequence
            actor_losses = -rhos * log_pis * advantages
            total_actor_loss += actor_losses.sum()
            
            # Critic Loss: mean over ensemble dimension, sum over sequence
            v_preds_tensor = torch.stack(v_preds_ensemble, dim=0) # [actual_seq_len, E]
            vs_target_expanded = vs.unsqueeze(-1).expand(actual_seq_len, self.agent.ensembles.__len__()) # [actual_seq_len, E]
            critic_losses = nn.functional.mse_loss(v_preds_tensor, vs_target_expanded.detach(), reduction='none')
            total_critic_loss += critic_losses.mean(dim=-1).sum()
            
            # Entropy Loss: sum over sequence
            entropy_losses = -entropies
            total_entropy_loss += entropy_losses.sum()
                
            # Logging the average sequence advantages
            self.writer.add_scalar("VTrace/Advantage_Mean", (rewards + gamma * torch.cat([v_enhanced[1:], v_enh_last.detach()]) - v_means[:-1].detach()).mean().item(), self.global_step)
            self.writer.add_scalar("VTrace/Rho_Mean", rhos.mean().item(), self.global_step)
            
        if valid_batches == 0:
            return False
            
        # Normalize the loss by the number of valid batches
        loss = (total_actor_loss + 0.5 * total_critic_loss + entropy_coef * total_entropy_loss) / valid_batches
        loss.backward()
        self.optimizer.step()
        
        self.writer.add_scalar("Loss/Total", loss.item(), self.global_step)
        self.writer.add_scalar("Loss/Actor", (total_actor_loss/valid_batches).item(), self.global_step)
        self.writer.add_scalar("Loss/Critic", (total_critic_loss/valid_batches).item(), self.global_step)
        self.writer.add_scalar("Loss/Entropy", (total_entropy_loss/valid_batches).item(), self.global_step)
        
        if len(total_game_rewards) > 0:
            avg_game_reward = sum(total_game_rewards) / len(total_game_rewards)
            self.writer.add_scalar("System/Average_Game_Total_Reward", avg_game_reward, self.global_step)
        
        self.global_step += 1
        return True

    def run(self):
        print(f"Starting IMPALA Learner on {self.device}...")
        try:
            while True:
                with self.buffer.lock:
                    buf_size = len(self.buffer.trajectories)
                
                self.writer.add_scalar("System/Buffer_Size", buf_size, self.global_step)
                    
                # Perform optimization step if we have enough data
                if self.learn_step(batch_size=8):
                    if self.global_step % 10 == 0:
                        self.save_checkpoint()
                        print(f"Step {self.global_step}: Saved checkpoint.")
                else:
                    # Wait for workers to generate data
                    time.sleep(2)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Gracefully shutting down Master...")
            self.save_checkpoint()
            print("Final checkpoint saved. Exiting.")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learner = ImpalaLearner(device=device)
    learner.run()
