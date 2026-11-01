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
    def __init__(self, agent_model, worker_id=None, host='localhost', port=4000, device='cpu', dataset_dir='data/rl_selfplay_trajectories'):
        self.agent = agent_model
        self.device = device
        self.worker_id = worker_id or str(uuid.uuid4())[:8]
        self.logger = setup_logger(f"Worker {port}")
        
        self.env = MegaMekEnvironment(port=port, device=self.device, logger=self.logger)
        self.traj_dir = os.path.join(dataset_dir, self.worker_id)
        os.makedirs(self.traj_dir, exist_ok=True)
        self.latest_model_path = os.path.join("model_objects", "impala_agent_latest.pt")
        self.run_id = int(time.time())
        self.stop_requested = False
        
    def sync_weights(self):
        """Loads the latest policy weights from disk if available."""
        if os.path.exists(self.latest_model_path):
            try:
                checkpoint = torch.load(self.latest_model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.agent.load_state_dict(checkpoint["model_state_dict"], strict=False)
                else:
                    self.agent.load_state_dict(checkpoint, strict=False)
                self.logger.debug(f"Synced latest weights from {self.latest_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load weights: {e}")

    def save_trajectory_chunk(self, trajectory, topology_payload, ep, win_status, gamelog_html="", is_done=False, episode_reward=0.0):
        traj_data = {
            "topology_payload": topology_payload,
            "steps": trajectory,
            "gamelog_html": gamelog_html,
            "win": win_status,
            "is_done": is_done,
            "episode_reward": episode_reward
        }
        
        chunk_id = int(time.time() * 1000)
        traj_file = os.path.join(self.traj_dir, f"traj_{self.run_id}_{ep}_{chunk_id}.pt")
        tmp_file = traj_file + ".tmp"
        torch.save(traj_data, tmp_file)
        os.replace(tmp_file, traj_file)
        self.logger.info(f"Saved trajectory chunk of length {len(trajectory)} to {traj_file} with win={win_status}")
        return traj_file

    def run(self, max_episodes=1000, max_steps_per_episode=500, max_turns=0):
        self.logger.info("Starting rollout loop...")
        
        state_graph, mask, current_payload = None, None, None
        
        for ep in range(max_episodes):
            if self.stop_requested:
                break
            self.sync_weights()
            self.agent.eval()
            
            if getattr(self.env, "done", True) or state_graph is None:
                try:
                    state_graph, mask, current_payload = self.env.reset()
                except Exception:
                    if self.stop_requested:
                        break
                    self.logger.warning("Connection to MegaMek lost. Waiting for server restart...")
                    time.sleep(5)
                    self.env = MegaMekEnvironment(port=self.env.port, device=self.device, logger=self.logger)
                    state_graph, mask, current_payload = None, None, None
                    continue
            
            topology_payload = getattr(self.env, 'topology_payload', None)
            trajectory = []
            
            # State graphs handled natively via environment
    
            step_idx = 0
            episode_reward = 0.0
            try:
                while not getattr(self.env, "done", False) and not self.stop_requested and step_idx < max_steps_per_episode:
                    # Fallback action if no valid actions
                    if (state_graph is None) or ('action' not in state_graph.node_types) or (state_graph['action'].x is None) or (state_graph['action'].x.size(0) == 0):
                        action_dict = {"selected_path_index": -1}

                    with torch.no_grad():
                        action_dict, v_mean, probs, mu_log_prob = self.agent.get_action(
                            state_graph, mask, deterministic=False
                        )

                    # Check if it was a valid action by looking at whether options were available
                    is_valid = True
                    if (state_graph is None) or ('action' not in state_graph.node_types) or (state_graph['action'].x is None) or (state_graph['action'].x.size(0) == 0):
                        is_valid = False
                        
                    if is_valid:
                        next_state, next_mask, done, next_payload = self.env.step(action_dict)
                        env_reward = next_payload.get("reward", 0.0) if next_payload else 0.0
                        rew_comps = next_payload.get("reward_components", {}) if next_payload else {}
                        reward_bv = rew_comps.get("bv", 0.0)
                        reward_vp = rew_comps.get("vp", 0.0)
                        reward_tp = rew_comps.get("tp", 0.0)

                        reward = env_reward
                        episode_reward += reward

                        trajectory.append({
                            "raw_payload": current_payload,
                            "action_mask": mask,
                            "action_dict": action_dict,
                            "mu_log_prob": mu_log_prob,
                            "reward": reward,
                            "env_reward": env_reward,
                            "reward_bv": reward_bv,
                            "reward_vp": reward_vp,
                            "reward_tp": reward_tp,
                        })

                        state_graph = next_state
                        mask = next_mask
                        current_payload = next_payload
                    else:
                        state_graph, mask, done, current_payload = self.env.step(action_dict)

                    if len(trajectory) >= 50:
                        self.save_trajectory_chunk(trajectory, topology_payload, ep, win_status=False, is_done=False, episode_reward=episode_reward)
                        trajectory = trajectory[-1:] # Retain the last step for V-trace bootstrapping!
                        self.sync_weights()

                    if done:
                        if current_payload and current_payload.get("game_over"):
                            win_status = current_payload.get("win", False)
                            self.logger.info(f"Game finished properly. Win: {win_status}")
                            # Append terminal bootstrap state so transitions = steps[:-1] retains the final action
                            trajectory.append({
                                "raw_payload": current_payload,
                                "action_mask": None,
                                "action_dict": {},
                                "mu_log_prob": 0.0,
                                "reward": 0.0,
                                "env_reward": 0.0,
                                "reward_bv": 0.0,
                                "reward_vp": 0.0,
                                "reward_tp": 0.0,
                                "is_terminal": True
                            })
                        break
                    
                    if max_turns > 0 and current_payload and current_payload.get("turn_number", 0) >= max_turns:
                        self.logger.info(f"Max turns ({max_turns}) reached. Exiting episode.")
                        break
                    
                    step_idx += 1
            except Exception as e:
                self.logger.error(f"Episode terminated abruptly: {e}")
                self.env.done = True # force reset on next episode to avoid infinite broken pipe loops
            finally:
                is_done = getattr(self.env, "done", False)
                if len(trajectory) > 1 or (is_done and len(trajectory) > 0):
                    win_status = False
                    if current_payload and isinstance(current_payload, dict) and current_payload.get("game_over"):
                        win_status = current_payload.get("win", False)
                    
                    gamelog_html = ""
                    try:
                        gamelog_path = os.path.join(self.traj_dir, "logs", "gamelog.html")
                        if os.path.exists(gamelog_path):
                            with open(gamelog_path, "r", encoding="utf-8") as f:
                                gamelog_html = f.read()
                    except Exception as gle:
                        self.logger.warning(f"Failed to read gamelog.html: {gle}")

                    self.save_trajectory_chunk(trajectory, topology_payload, ep, win_status, gamelog_html, is_done=is_done, episode_reward=episode_reward)
                    
                    if is_done:
                        try:
                            import subprocess
                            import glob
                            render_script = os.path.join(os.path.dirname(__file__), "validations", "render_game_state.py")
                            
                            chunk_files = glob.glob(os.path.join(self.traj_dir, f"traj_{self.run_id}_{ep}_*.pt"))
                            chunk_files.sort(key=os.path.getctime)
                            
                            if chunk_files:
                                out_html = os.path.join(self.traj_dir, f"traj_{self.run_id}_{ep}_full.html")
                                cmd = [sys.executable, render_script] + chunk_files + [out_html]
                                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        except Exception as re:
                            self.logger.warning(f"Failed to trigger auto-render: {re}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4000, help="Port to connect to MegaMek")
    parser.add_argument("--dataset_dir", type=str, default="data/rl_selfplay_trajectories", help="Directory to save trajectories")
    parser.add_argument("--test", action="store_true", help="Run in test mode (small limits)")
    parser.add_argument("--max_turns", type=int, default=0, help="Max turns to run (0 for infinite)")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Max episodes to run (0 for infinite)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on")
    parser.add_argument("--worker_id", type=str, default=None, help="Explicit worker ID for directory organization")
    args = parser.parse_args()

    worker = None
    
    def signal_handler(sig, frame):
        if worker is not None:
            worker.stop_requested = True
            if hasattr(worker, 'logger'):
                worker.logger.info(f"Worker received signal {sig}, flushing in-flight trajectory and shutting down...")
            if getattr(worker.env, 'sock', None) is not None:
                try:
                    worker.env.sock.close()
                except Exception:
                    pass
        else:
            print(f"\nWorker received signal {sig}, shutting down...")
            sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    device = torch.device(args.device)
    temp_logger = setup_logger(f"Worker {args.port}")
    temp_logger.info(f"Initializing IMPALA Worker on {device}...")
    
    agent = MegaMekAgent(hidden_dim=128, ensemble_size=8).to(device)
    
    # Try to bootstrap from BC initially if impala_latest doesn't exist
    bc_model_path = os.path.join("model_objects", "bc_agent.pt")
    if not os.path.exists(os.path.join("model_objects", "impala_agent_latest.pt")) and os.path.exists(bc_model_path):
        temp_logger.info(f"Bootstrapping worker from BC weights: {bc_model_path}")
        agent.load_state_dict(torch.load(bc_model_path, map_location=device), strict=False)
        
    worker = ImpalaWorker(agent, worker_id=args.worker_id, port=args.port, device=device, dataset_dir=args.dataset_dir)
    if args.test:
        worker.run(max_episodes=1, max_steps_per_episode=15)
    else:
        worker.run(max_turns=args.max_turns, max_episodes=args.max_episodes)
