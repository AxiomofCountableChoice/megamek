import torch
import torch.nn as nn
from torch.optim import Adam
import time
import os

from models.agent import MegaMekAgent
from env import MegaMekEnvironment

class ImpalaWorker:
    """
    Batched procedural rollout worker for IMPALA using the MegaMek environment.
    Runs episodes synchronously, collects V-Trace targets, Epistemic Variances, and updates.
    """
    def __init__(self, agent, host='localhost', port=4000, device='cpu'):
        self.agent = agent
        self.device = device
        self.env = MegaMekEnvironment(port=port)
        self.optimizer = Adam(self.agent.parameters(), lr=1e-4)
        
    def collect_trajectory(self, max_steps=50):
        trajectory = []
        state_graph, mask = self.env.reset(device=self.device)
        self.agent.eval()
        
        step_idx = 0
        while not getattr(self.env, "done", False) and step_idx < max_steps:
            if state_graph is None or 'action' not in state_graph.node_types or state_graph['action'].x is None or state_graph['action'].x.size(0) == 0:
                action_dict = {"selected_path_index": -1}
                state_graph, mask, done = self.env.step(action_dict, device=self.device)
                if done: break
                continue

            with torch.no_grad():
                action_dict, v_mean, probs = self.agent.get_action(state_graph, mask)
                
            selected_action = action_dict.get("selected_path_index", -1)
            
            if selected_action != -1:
                log_prob = torch.log(probs[selected_action] + 1e-10)
                trajectory.append({
                    "state": state_graph.cpu(),
                    "mask": mask, # Save mask for re-evaluating target policy probs
                    "action_idx": selected_action,
                    "mu_log_prob": log_prob.item()
                })
                
            state_graph, mask, done = self.env.step(action_dict, device=self.device)
            if done: break
            step_idx += 1
            
        return trajectory

    def compute_vtrace_targets(self, rewards, v_means, v_vars, mu_probs, pi_probs, 
                               rho_bar=1.0, c_bar=1.0, gamma=0.99, lam_epistemic=1.0):
        """
        Calculates the V-Trace targets incorporating the Epistemic Uncertainty bonus.
        """
        seq_len = len(rewards)
        vs = torch.zeros(seq_len, device=self.device)
        
        # Importance sampling weights
        rhos = torch.min(torch.tensor(rho_bar, device=self.device), pi_probs / mu_probs)
        cs = torch.min(torch.tensor(c_bar, device=self.device), pi_probs / mu_probs)
        
        # Calculate epistemic standard deviation
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
            
        # v_enhanced targets for actor loss (Section 2c)
        v_enhanced = vs + lam_epistemic * sigma
        
        return vs, v_enhanced, rhos

    def impala_update(self, trajectories, rho_bar=1.0, c_bar=1.0, gamma=0.99, lam_epistemic=1.0, entropy_coef=0.01):
        self.agent.train()
        
        for traj in trajectories:
            seq_len = len(traj)
            if seq_len == 0: continue
            
            rewards = torch.zeros(seq_len, device=self.device) # Placeholder
            mu_probs = torch.zeros(seq_len, device=self.device)
            pi_probs = torch.zeros(seq_len, device=self.device)
            v_means = torch.zeros(seq_len, device=self.device)
            v_vars = torch.zeros(seq_len, device=self.device)
            v_preds_ensemble = []
            log_pis = torch.zeros(seq_len, device=self.device)
            entropies = torch.zeros(seq_len, device=self.device)
            
            self.optimizer.zero_grad()
            
            # Re-evaluate trajectory to get current target policy \pi and critic predictions
            for t, step_data in enumerate(traj):
                graph = step_data["state"].to(self.device)
                idx = step_data["action_idx"]
                mu_probs[t] = torch.exp(torch.tensor(step_data["mu_log_prob"], device=self.device))
                
                # 1. Forward Graph Component
                z, x_dict = self.agent.encoder(graph)
                
                # 2. Extract Value Ensembles
                v_preds = torch.stack([critic(z) for critic in self.agent.ensembles], dim=-1) # [1, 1, E]
                v_preds_ensemble.append(v_preds.squeeze(0).squeeze(0)) # [E]
                
                v_means[t] = v_preds.mean()
                v_vars[t] = v_preds.var(unbiased=False)
                
                # 3. Action Logic
                e_actions = self.agent.actor_pointer.compute_action_embeddings(x_dict, graph)
                s_k = self.agent.actor_pointer.decode_sequence(z, None) 
                batch_idx = graph['action'].batch if hasattr(graph['action'], 'batch') and getattr(graph['action'], 'batch') is not None else None
                logits = self.agent.actor_pointer.score_actions(s_k, e_actions, batch_idx)
                
                probs = torch.softmax(logits, dim=-1)
                pi_probs[t] = probs[idx] + 1e-10
                log_pis[t] = torch.log(probs[idx] + 1e-10)
                
                # Entropy
                entropies[t] = -(probs * torch.log(probs + 1e-10)).sum()
                
            # Compute V-Trace
            vs, v_enhanced, rhos = self.compute_vtrace_targets(
                rewards, v_means.detach(), v_vars.detach(), mu_probs, pi_probs.detach(), 
                rho_bar, c_bar, gamma, lam_epistemic
            )
            
            # Compute Losses
            # Actor Loss (Policy Gradient with V-Trace and Epistemic Bonus)
            # -rho_s * log(pi) * (r_s + gamma * v^{enhanced}_{s+1} - V_mean(z_s))
            actor_loss = 0
            for t in range(seq_len):
                if t == seq_len - 1:
                    v_enh_next = 0.0
                else:
                    v_enh_next = v_enhanced[t+1]
                
                advantage = rewards[t] + gamma * v_enh_next - v_means[t].detach()
                actor_loss += -rhos[t] * log_pis[t] * advantage
                
            actor_loss = actor_loss / seq_len
            
            # Critic Loss (MSE of Ensemble Heads vs vs_target)
            v_preds_tensor = torch.stack(v_preds_ensemble) # [seq_len, E]
            vs_target_expanded = vs.unsqueeze(1).expand(-1, self.agent.ensembles.__len__()) # [seq_len, E]
            critic_loss = nn.functional.mse_loss(v_preds_tensor, vs_target_expanded.detach())
            
            # Entropy Bonus
            entropy_loss = -entropies.mean()
            
            loss = actor_loss + 0.5 * critic_loss + entropy_coef * entropy_loss
            loss.backward()
            self.optimizer.step()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting IMPALA Rollout Worker on {device}...")
    
    # Using E=8 critic ensemble
    agent = MegaMekAgent(hidden_dim=128, ensemble_size=8).to(device)
    worker = ImpalaWorker(agent, device=device)
    
    print("Awaiting MegaMek Gym connection...")
    
    for iteration in range(5):
        print(f"--- Iteration {iteration+1} ---")
        traj = worker.collect_trajectory(max_steps=20)
        print(f"Collected contiguous trajectory of length {len(traj)}.")
        if len(traj) > 0:
            worker.impala_update([traj], rho_bar=1.0, c_bar=1.0)
            print("Model weights updated using V-Trace.")
            
    print("Run finished.")
