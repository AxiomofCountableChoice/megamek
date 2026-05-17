import os
import time
import torch
import uuid
import sys
import signal
import argparse
from models.agent import MegaMekAgent
from env import MegaMekEnvironment
from logger_config import setup_logger

class ImpalaWorker:
    """
    Rollout worker for IMPALA using the MegaMek environment.
    Connects to MegaMek, syncs latest weights, collects raw trajectories,
    and writes them to disk for the Learner.
    """
    def __init__(self, agent_model, worker_id=None, host='localhost', port=4000, device='cpu', dataset_dir='rl_sp_dataset'):
        self.agent = agent_model
        self.device = device
        self.worker_id = worker_id or str(uuid.uuid4())[:8]
        self.logger = setup_logger(f"Worker {port}")
        
        self.env = MegaMekEnvironment(port=port, device=self.device, logger=self.logger)
        self.traj_dir = os.path.join(dataset_dir, self.worker_id)
        os.makedirs(self.traj_dir, exist_ok=True)
        self.latest_model_path = os.path.join("models", "impala_agent_latest.pt")
        
    def sync_weights(self):
        """Loads the latest policy weights from disk if available."""
        if os.path.exists(self.latest_model_path):
            try:
                state_dict = torch.load(self.latest_model_path, map_location=self.device)
                self.agent.load_state_dict(state_dict)
                self.logger.debug(f"Synced latest weights from {self.latest_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load weights: {e}")

    def run(self, max_episodes=1000, max_steps_per_episode=500, max_turns=0):
        self.logger.info("Starting rollout loop...")
        
        state_graph, mask, current_payload = None, None, None
        
        for ep in range(max_episodes):
            self.sync_weights()
            self.agent.eval()
            
            if getattr(self.env, "done", True) or state_graph is None:
                try:
                    state_graph, mask, current_payload = self.env.reset()
                except ConnectionError:
                    self.logger.warning("Connection to MegaMek lost. Waiting for server restart...")
                    time.sleep(5)
                    self.env = MegaMekEnvironment(port=self.env.port, device=self.device, logger=self.logger)
                    state_graph, mask, current_payload = None, None, None
                    continue
            
            topology_payload = getattr(self.env, 'topology_payload', None)
            trajectory = []
            
            # State graphs handled natively via environment
    
            step_idx = 0
            try:
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
                        # Reward is now calculated inside self.env.step() and bundled in current_payload
                        reward = current_payload.get("reward", 0.0) if current_payload else 0.0

                        trajectory.append({
                            "raw_payload": current_payload,
                            "action_dict": action_dict,
                            "mu_log_prob": mu_log_prob,
                            "reward": reward
                        })

                    state_graph, mask, done, current_payload = self.env.step(action_dict)

                    if done:
                        break
                    
                    if max_turns > 0 and current_payload and current_payload.get("turn_number", 0) >= max_turns:
                        self.logger.info(f"Max turns ({max_turns}) reached. Exiting episode.")
                        break
                    
                    step_idx += 1
            except Exception as e:
                self.logger.error(f"Episode terminated abruptly: {e}")
            finally:
                if len(trajectory) > 0:
                    traj_data = {
                        "topology_payload": topology_payload,
                        "steps": trajectory
                    }
                    
                    traj_file = os.path.join(self.traj_dir, f"traj_{ep}_{int(time.time())}.pt")
                    torch.save(traj_data, traj_file)
                    self.logger.info(f"Saved trajectory of length {len(trajectory)} to {traj_file}")
                    
                    try:
                        import subprocess
                        render_script = os.path.join(os.path.dirname(__file__), "validations", "render_game_state.py")
                        out_html = traj_file.replace(".pt", ".html")
                        subprocess.Popen([sys.executable, render_script, traj_file, out_html], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception as re:
                        self.logger.warning(f"Failed to trigger auto-render for {traj_file}: {re}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4000, help="Port to connect to MegaMek")
    parser.add_argument("--dataset_dir", type=str, default="rl_sp_dataset", help="Directory to save trajectories")
    parser.add_argument("--test", action="store_true", help="Run in test mode (small limits)")
    parser.add_argument("--max_turns", type=int, default=0, help="Max turns to run (0 for infinite)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on")
    args = parser.parse_args()

    worker = None
    
    def signal_handler(sig, frame):
        if worker is not None and hasattr(worker, 'logger'):
            worker.logger.info(f"Worker received signal {sig}, shutting down gracefully...")
        else:
            print(f"\nWorker received signal {sig}, shutting down gracefully...")
        if worker is not None and getattr(worker.env, 'sock', None) is not None:
            try:
                worker.env.sock.close()
            except Exception:
                pass
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    device = torch.device(args.device)
    temp_logger = setup_logger(f"Worker {args.port}")
    temp_logger.info(f"Initializing IMPALA Worker on {device}...")
    
    agent = MegaMekAgent(hidden_dim=128, ensemble_size=8).to(device)
    
    # Try to bootstrap from BC initially if impala_latest doesn't exist
    bc_model_path = os.path.join("models", "bc_agent.pt")
    if not os.path.exists(os.path.join("models", "impala_agent_latest.pt")) and os.path.exists(bc_model_path):
        temp_logger.info(f"Bootstrapping worker from BC weights: {bc_model_path}")
        agent.load_state_dict(torch.load(bc_model_path, map_location=device))
        
    worker = ImpalaWorker(agent, port=args.port, device=device, dataset_dir=args.dataset_dir)
    if args.test:
        worker.run(max_episodes=1, max_steps_per_episode=15)
    else:
        worker.run(max_turns=args.max_turns)
