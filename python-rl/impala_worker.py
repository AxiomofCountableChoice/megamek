import os
import time
import torch
import uuid

from models.agent import MegaMekAgent
from env import MegaMekEnvironment

class ImpalaWorker:
    """
    Rollout worker for IMPALA using the MegaMek environment.
    Connects to MegaMek, syncs latest weights, collects raw trajectories,
    and writes them to disk for the Learner.
    """
    def __init__(self, agent_model, worker_id=None, host='localhost', port=4000, device='cpu'):
        self.agent = agent_model
        self.device = device
        self.env = MegaMekEnvironment(port=port, device=self.device)
        self.worker_id = worker_id or str(uuid.uuid4())[:8]
        self.traj_dir = os.path.join("data", "trajectories", self.worker_id)
        os.makedirs(self.traj_dir, exist_ok=True)
        self.latest_model_path = os.path.join("models", "impala_agent_latest.pt")
        
    def sync_weights(self):
        """Loads the latest policy weights from disk if available."""
        if os.path.exists(self.latest_model_path):
            try:
                state_dict = torch.load(self.latest_model_path, map_location=self.device)
                self.agent.load_state_dict(state_dict)
                print(f"Worker {self.worker_id}: Synced latest weights from {self.latest_model_path}")
            except Exception as e:
                print(f"Worker {self.worker_id}: Failed to load weights: {e}")

    def run(self, max_episodes=1000, max_steps_per_episode=50):
        print(f"Worker {self.worker_id} starting rollout loop...")
        
        for ep in range(max_episodes):
            self.sync_weights()
            self.agent.eval()
            
            try:
                state_graph, mask, current_payload = self.env.reset()
            except ConnectionError:
                print("Connection to MegaMek lost. Waiting for server restart...")
                time.sleep(5)
                self.env = MegaMekEnvironment(port=self.env.port, device=self.device)
                continue
            
            topology_payload = getattr(self.env, 'topology_payload', None)
            trajectory = []
            
            step_idx = 0
            while not getattr(self.env, "done", False) and step_idx < max_steps_per_episode:
                # Fallback action if no valid actions
                if state_graph is None or 'action' not in state_graph.node_types or state_graph['action'].x is None or state_graph['action'].x.size(0) == 0:
                    action_dict = {"selected_path_index": -1}
                    state_graph, mask, done, current_payload = self.env.step(action_dict)
                    if done: break
                    continue

                with torch.no_grad():
                    # Forward pass
                    action_dict, v_mean, probs = self.agent.get_action(state_graph, mask)
                    
                selected_action = action_dict.get("selected_path_index", -1)
                
                if selected_action != -1:
                    log_prob = torch.log(probs[selected_action] + 1e-10).item()
                    
                    trajectory.append({
                        "raw_payload": current_payload,
                        "action_idx": selected_action,
                        "mu_log_prob": log_prob,
                        "reward": 0.0 # Standard zero reward placeholder (can be updated post-episode)
                    })
                    
                state_graph, mask, done, current_payload = self.env.step(action_dict)
                if done: break
                step_idx += 1
                
            if len(trajectory) > 0:
                # Assign final reward if episode ends with a win/loss
                # For now, simplistic reward
                trajectory[-1]["reward"] = 1.0 if getattr(self.env, "done", False) else 0.0
                
                # Save to disk
                traj_data = {
                    "topology_payload": topology_payload,
                    "steps": trajectory
                }
                
                traj_file = os.path.join(self.traj_dir, f"traj_{ep}_{int(time.time())}.pt")
                torch.save(traj_data, traj_file)
                print(f"Worker {self.worker_id}: Saved trajectory of length {len(trajectory)} to {traj_file}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing IMPALA Worker on {device}...")
    
    agent = MegaMekAgent(hidden_dim=128, ensemble_size=8).to(device)
    
    # Try to bootstrap from BC initially if impala_latest doesn't exist
    bc_model_path = os.path.join("models", "bc_agent.pt")
    if not os.path.exists(os.path.join("models", "impala_agent_latest.pt")) and os.path.exists(bc_model_path):
        print(f"Bootstrapping worker from BC weights: {bc_model_path}")
        agent.load_state_dict(torch.load(bc_model_path, map_location=device))
        
    worker = ImpalaWorker(agent, device=device)
    worker.run()
