import os
import time
import torch
import uuid
import sys
import signal
import argparse
from models.agent import MegaMekAgent
from env import MegaMekEnvironment

class ImpalaWorker:
    """
    Rollout worker for IMPALA using the MegaMek environment.
    Connects to MegaMek, syncs latest weights, collects raw trajectories,
    and writes them to disk for the Learner.
    """
    def __init__(self, agent_model, worker_id=None, host='localhost', port=4000, device='cpu', dataset_dir='rl_sp_dataset'):
        self.agent = agent_model
        self.device = device
        self.env = MegaMekEnvironment(port=port, device=self.device)
        self.worker_id = worker_id or str(uuid.uuid4())[:8]
        self.traj_dir = os.path.join(dataset_dir, self.worker_id)
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
            
            # Tracking for dense rewards
            prev_bv1, prev_bv2 = None, None
            prev_tp1, prev_tp2 = 0, 0
            prev_vp1 = 0
            total_match_bv = 1.0 # fallback to prevent div by zero
            
            beta_bv = 1.0
            beta_tp = 0.01
            beta_vp = 1.0
            
            step_idx = 0
            while not getattr(self.env, "done", False) and step_idx < max_steps_per_episode:
                # Fallback action if no valid actions
                if (state_graph is None) or ('action' not in state_graph.node_types) or (state_graph['action'].x is None) or (state_graph['action'].x.size(0) == 0):
                    action_dict = {"selected_path_index": -1}

                with torch.no_grad():
                    # Forward pass (Training mode: deterministic=False)
                    action_dict, v_mean, probs, mu_log_prob = self.agent.get_action(state_graph, mask, deterministic=False)

                    
                # Check if it was a valid action by looking at the default fallback
                is_valid = True
                if "selected_path_index" in action_dict and action_dict["selected_path_index"] == -1:
                    is_valid = False
                    
                if is_valid:
                    # Calculate Dense Reward
                    reward = 0.0
                    if current_payload and "rewards" in current_payload:
                        rew_dict = current_payload["rewards"]
                        bv1 = rew_dict.get("bv1", 0)
                        bv2 = rew_dict.get("bv2", 0)
                        tp1 = rew_dict.get("tp1", 0)
                        tp2 = rew_dict.get("tp2", 0)
                        vp1 = rew_dict.get("vp1", 0) # Already zero-sum from Java
                        
                        if prev_bv1 is None:
                            prev_bv1, prev_bv2 = bv1, bv2
                            total_match_bv = max(bv1 + bv2, 1.0)
                            prev_vp1 = vp1
                            
                        delta_bv1 = bv1 - prev_bv1
                        delta_bv2 = bv2 - prev_bv2
                        delta_vp1 = vp1 - prev_vp1
                        
                        reward_bv = beta_bv * ((delta_bv1 - delta_bv2) / total_match_bv)
                        reward_tp = beta_tp * (tp2 - tp1)
                        reward_vp = beta_vp * delta_vp1
                        
                        reward = reward_bv + reward_tp + reward_vp
                        
                        prev_bv1, prev_bv2 = bv1, bv2
                        prev_tp1, prev_tp2 = tp1, tp2
                        prev_vp1 = vp1

                    trajectory.append({
                        "raw_payload": current_payload,
                        "action_dict": action_dict,
                        "mu_log_prob": mu_log_prob,
                        "reward": reward
                    })

                state_graph, mask, done, current_payload = self.env.step(action_dict)

                if done:
                    break
                
                step_idx += 1

            if len(trajectory) > 0:
                traj_data = {
                    "topology_payload": topology_payload,
                    "steps": trajectory
                }
                
                traj_file = os.path.join(self.traj_dir, f"traj_{ep}_{int(time.time())}.pt")
                torch.save(traj_data, traj_file)
                print(f"Worker {self.worker_id}: Saved trajectory of length {len(trajectory)} to {traj_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4000, help="Port to connect to MegaMek")
    parser.add_argument("--dataset_dir", type=str, default="rl_sp_dataset", help="Directory to save trajectories")
    parser.add_argument("--test", action="store_true", help="Run in test mode (small limits)")
    args = parser.parse_args()

    worker = None
    
    def signal_handler(sig, frame):
        print(f"\nWorker received signal {sig}, shutting down gracefully...")
        if worker is not None and getattr(worker.env, 'sock', None) is not None:
            try:
                worker.env.sock.close()
            except Exception:
                pass
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing IMPALA Worker on {device} (Port: {args.port})...")
    
    agent = MegaMekAgent(hidden_dim=128, ensemble_size=8).to(device)
    
    # Try to bootstrap from BC initially if impala_latest doesn't exist
    bc_model_path = os.path.join("models", "bc_agent.pt")
    if not os.path.exists(os.path.join("models", "impala_agent_latest.pt")) and os.path.exists(bc_model_path):
        print(f"Bootstrapping worker from BC weights: {bc_model_path}")
        agent.load_state_dict(torch.load(bc_model_path, map_location=device))
        
    worker = ImpalaWorker(agent, port=args.port, device=device, dataset_dir=args.dataset_dir)
    if args.test:
        worker.run(max_episodes=1, max_steps_per_episode=15)
    else:
        worker.run()
