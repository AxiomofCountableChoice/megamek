import os
import glob
import time
import math
import signal
import threading
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import atexit
import random
import argparse
from torch_geometric.data import Batch

from models.agent import MegaMekAgent
from env import MegaMekEnvironment
from logger_config import setup_logger

logger = setup_logger("ImpalaMaster")


def compute_annealed_weight(
    step: int,
    initial: float,
    final: float = 0.0,
    warmup_steps: int = 0,
    decay_steps: int = 5000,
    schedule: str = "cosine",
) -> float:
    """
    Computes an annealed scalar weight with an optional constant warmup period
    followed by a 'linear', 'cosine', or 'exponential' decay to `final`.
    """
    if step < warmup_steps:
        return float(initial)
    if decay_steps <= 0 or step >= warmup_steps + decay_steps:
        return float(final)
    progress = (step - warmup_steps) / float(decay_steps)
    if schedule == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif schedule == "exponential":
        if initial <= 0.0:
            return float(final)
        ratio = max(final, 1e-4) / max(initial, 1e-4)
        return float(initial * (ratio ** progress))
    else:  # "linear"
        factor = 1.0 - progress
    return float(final + (initial - final) * factor)


class ImpalaReplayBuffer:
    """
    Monitors dataset directories for new trajectories, loads them into memory,
    and removes the processed files.
    """

    def __init__(
        self,
        dataset_dirs=["data/rl_selfplay_trajectories", "data/rl_princess_trajectories"],
        max_trajectories=1000,
    ):
        self.dataset_dirs = dataset_dirs
        self.max_trajectories = max_trajectories
        self.trajectories = []
        self.ingested_files = set()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

        # Start background worker to ingest trajectories
        self.ingest_thread = threading.Thread(target=self._ingest_loop, daemon=True)
        self.ingest_thread.start()
        atexit.register(self.close)

    def close(self):
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if hasattr(self, "ingest_thread") and self.ingest_thread.is_alive():
            self.ingest_thread.join(timeout=1.0)

    def _ingest_loop(self):
        last_cleanup_time = time.time()
        while not self._stop_event.is_set():
            for d in self.dataset_dirs:
                if not os.path.exists(d):
                    continue
                worker_dirs = glob.glob(os.path.join(d, "*"))
                for wd in worker_dirs:
                    if not os.path.isdir(wd):
                        continue

                    files = glob.glob(os.path.join(wd, "*.pt"))
                    for f in files:
                        if f in self.ingested_files:
                            continue
                        if os.path.exists(f + ".consumed"):
                            continue
                        try:
                            data = torch.load(f, weights_only=False, map_location="cpu")
                            with self.lock:
                                self.trajectories.append((f, data))
                                self.ingested_files.add(f)
                                if len(self.trajectories) > self.max_trajectories:
                                    oldest_f, _ = self.trajectories.pop(0)  # FIFO
                                    self.ingested_files.discard(oldest_f)
                                    try:
                                        open(oldest_f + ".consumed", "w").close()
                                    except OSError:
                                        pass
                        except Exception as e:
                            logger.error(f"[ReplayBuffer] Error loading {f}: {e}")

            if time.time() - last_cleanup_time > 60:
                self._cleanup_old_trajectories()
                last_cleanup_time = time.time()

            self._stop_event.wait(2.0)

    def _cleanup_old_trajectories(self, max_age_seconds=1800):
        """Deletes .consumed files and their corresponding .pt files if they are older than max_age_seconds."""
        now = time.time()
        for d in self.dataset_dirs:
            if not os.path.exists(d):
                continue
            worker_dirs = glob.glob(os.path.join(d, "*"))
            for wd in worker_dirs:
                if not os.path.isdir(wd):
                    continue

                consumed_files = glob.glob(os.path.join(wd, "*.pt.consumed"))
                for cf in consumed_files:
                    try:
                        mtime = os.path.getmtime(cf)
                        if now - mtime > max_age_seconds:
                            pt_file = cf.replace(".consumed", "")
                            os.remove(cf)
                            if os.path.exists(pt_file):
                                os.remove(pt_file)
                    except Exception:
                        pass

    def sample_batch(self, batch_size=4, sequence_length=16):
        """
        Double Coin-Flip Strategy:
        1. Uniformly sample B trajectories.
        2. Uniformly sample a starting step to get a chunk of sequence_length.
        """
        with self.lock:
            if len(self.trajectories) < batch_size:
                return None

            sampled_trajs = random.sample(self.trajectories, batch_size)

        batch_data = []
        for f, traj in sampled_trajs:
            topology = traj.get("topology_payload", None)
            steps = traj.get("steps", [])
            total_reward = sum([s.get("reward", 0.0) for s in steps])
            win = traj.get("win", False)

            if len(steps) == 0:
                continue

            if len(steps) <= sequence_length + 1:
                sampled_steps = steps
            else:
                max_start = len(steps) - (sequence_length + 1)
                start_idx = random.randint(0, max_start)
                sampled_steps = steps[start_idx : start_idx + sequence_length + 1]

            batch_data.append((topology, sampled_steps, total_reward, win))

        return batch_data


class ExpertReplayBuffer:
    """
    Maintains an in-memory pool of expert state-action graphs loaded from
    the static BC trajectory dataset (data/bc_trajectories/*.pt).
    """

    def __init__(self, expert_dir="data/bc_trajectories", max_graphs=15000):
        self.expert_dir = expert_dir
        self.max_graphs = max_graphs
        self.graphs = []
        self.lock = threading.Lock()
        self.files = glob.glob(os.path.join(expert_dir, "*.pt"))
        logger.info(f"[ExpertBuffer] Found {len(self.files)} expert trajectory files in {expert_dir}")
        self._load_initial()

        self._stop_event = threading.Event()
        self.refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self.refresh_thread.start()
        atexit.register(self.close)

    def close(self):
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if hasattr(self, "refresh_thread") and self.refresh_thread.is_alive():
            self.refresh_thread.join(timeout=1.0)

    def _load_initial(self):
        if not self.files:
            return
        files_shuffled = list(self.files)
        random.shuffle(files_shuffled)
        loaded = 0
        sync_target = min(500, self.max_graphs)
        self.remaining_files = []
        for idx, f in enumerate(files_shuffled):
            try:
                data = torch.load(f, weights_only=False, map_location="cpu")
                trajs = (
                    data.get("trajectories", [])
                    if isinstance(data, dict)
                    else (data if isinstance(data, list) else [])
                )
                for g in trajs:
                    if getattr(g, "y_sequence", None) is not None and g.y_sequence.numel() > 0:
                        self.graphs.append(g)
                        loaded += 1
                        if loaded >= sync_target:
                            self.remaining_files = files_shuffled[idx + 1 :]
                            break
            except Exception as e:
                logger.error(f"[ExpertBuffer] Error loading {f}: {e}")
            if loaded >= sync_target:
                break
        logger.info(
            f"[ExpertBuffer] Initialized pool with {len(self.graphs)} expert graphs (sync). Remaining will load in background."
        )

    def _refresh_loop(self):
        if hasattr(self, "remaining_files") and self.remaining_files:
            for f in self.remaining_files:
                if self._stop_event.is_set():
                    return
                try:
                    data = torch.load(f, weights_only=False, map_location="cpu")
                    trajs = (
                        data.get("trajectories", [])
                        if isinstance(data, dict)
                        else (data if isinstance(data, list) else [])
                    )
                    new_g = [
                        g
                        for g in trajs
                        if getattr(g, "y_sequence", None) is not None and g.y_sequence.numel() > 0
                    ]
                    if new_g:
                        with self.lock:
                            self.graphs.extend(new_g)
                            if len(self.graphs) >= self.max_graphs:
                                break
                except Exception:
                    pass
            logger.info(f"[ExpertBuffer] Background loading complete. Pool size: {len(self.graphs)}.")

        while not self._stop_event.is_set():
            if self._stop_event.wait(300.0):
                return
            new_files = glob.glob(os.path.join(self.expert_dir, "*.pt"))
            if not new_files:
                continue
            sample_files = random.sample(new_files, min(len(new_files), 50))
            new_graphs = []
            for f in sample_files:
                if self._stop_event.is_set():
                    return
                try:
                    data = torch.load(f, weights_only=False, map_location="cpu")
                    trajs = (
                        data.get("trajectories", [])
                        if isinstance(data, dict)
                        else (data if isinstance(data, list) else [])
                    )
                    for g in trajs:
                        if getattr(g, "y_sequence", None) is not None and g.y_sequence.numel() > 0:
                            new_graphs.append(g)
                except Exception:
                    pass
            if new_graphs:
                with self.lock:
                    if len(self.graphs) + len(new_graphs) > self.max_graphs:
                        keep_count = self.max_graphs - len(new_graphs)
                        self.graphs = self.graphs[:keep_count] + new_graphs
                    else:
                        self.graphs.extend(new_graphs)

    def sample_batch(self, batch_size=16):
        with self.lock:
            if len(self.graphs) < batch_size:
                if len(self.graphs) > 0:
                    sampled = random.choices(self.graphs, k=batch_size)
                else:
                    return None
            else:
                sampled = random.sample(self.graphs, batch_size)
        return Batch.from_data_list(sampled)


class ImpalaLearner:
    def __init__(
        self,
        device="cpu",
        dataset_dirs=None,
        batch_size=4,
        expert_batch_size=16,
        expert_dir="data/bc_trajectories",
        force_bootstrap=False,
        lambda_bc_initial=1.0,
        lambda_bc_final=0.0,
        bc_warmup_steps=500,
        bc_decay_steps=5000,
        bc_schedule="cosine",
        lam_epistemic_initial=0.5,
        lam_epistemic_final=0.0,
        epistemic_decay_steps=10000,
        bootstrap_prob=0.63,
    ):
        self.device = device
        self.batch_size = batch_size
        self.expert_batch_size = expert_batch_size
        self.agent = MegaMekAgent(hidden_dim=128, ensemble_size=8).to(self.device)
        self.optimizer = Adam(self.agent.parameters(), lr=1e-4)
        self.writer = SummaryWriter(log_dir="runs/impala_master")

        self.buffer = ImpalaReplayBuffer(dataset_dirs=dataset_dirs) if dataset_dirs else ImpalaReplayBuffer()
        self.expert_buffer = ExpertReplayBuffer(expert_dir=expert_dir)

        self.global_step = 0
        self.lambda_bc_initial = lambda_bc_initial
        self.lambda_bc_final = lambda_bc_final
        self.bc_warmup_steps = bc_warmup_steps
        self.bc_decay_steps = bc_decay_steps
        self.bc_schedule = bc_schedule
        self.lambda_bc = lambda_bc_initial

        self.lam_epistemic_initial = lam_epistemic_initial
        self.lam_epistemic_final = lam_epistemic_final
        self.epistemic_decay_steps = epistemic_decay_steps
        self.lam_epistemic = lam_epistemic_initial
        self.bootstrap_prob = bootstrap_prob
        self.stop_requested = False

        # Load BC bootstrapping if impala_latest doesn't exist or if forced
        self.latest_model_path = os.path.join("model_objects", "impala_agent_latest.pt")
        bc_model_path = os.path.join("model_objects", "bc_agent.pt")

        os.makedirs("model_objects", exist_ok=True)

        if not force_bootstrap and os.path.exists(self.latest_model_path):
            logger.info(f"Resuming from {self.latest_model_path}")
            checkpoint = torch.load(self.latest_model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.agent.load_state_dict(checkpoint["model_state_dict"], strict=False)
                if "optimizer_state_dict" in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        logger.info("Restored Adam optimizer state from checkpoint.")
                    except Exception as oe:
                        logger.warning(f"Could not restore optimizer state: {oe}")
                self.global_step = checkpoint.get("global_step", 0)
                if "lambda_bc" in checkpoint:
                    self.lambda_bc = checkpoint["lambda_bc"]
                if "lam_epistemic" in checkpoint:
                    self.lam_epistemic = checkpoint["lam_epistemic"]
            else:
                self.agent.load_state_dict(checkpoint, strict=False)
        elif os.path.exists(bc_model_path):
            logger.info(f"Bootstrapping Learner from BC weights: {bc_model_path}")
            checkpoint = torch.load(bc_model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.agent.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                self.agent.load_state_dict(checkpoint, strict=False)
            self.save_checkpoint()
        else:
            logger.info("No existing weights found. Starting from scratch.")

        # Offline parser environment
        self.env = MegaMekEnvironment(port=0, device=self.device, connect_on_init=False)
        self.env._connected = True
        atexit.register(self.close)

    def close(self):
        if hasattr(self, "buffer"):
            self.buffer.close()
        if hasattr(self, "expert_buffer"):
            self.expert_buffer.close()

    def save_checkpoint(self):
        payload = {
            "model_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lambda_bc": self.lambda_bc,
            "lam_epistemic": self.lam_epistemic,
            "global_step": self.global_step,
        }
        tmp_path = self.latest_model_path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, self.latest_model_path)

    @torch.no_grad()
    def compute_vtrace_targets(
        self,
        rewards,
        v_preds_detached,
        mu_probs,
        pi_probs,
        rho_bar=1.0,
        c_bar=1.0,
        gamma=0.99,
    ):
        """
        Vectorized per-head V-Trace target computation across all E Critic heads.
        Epistemic bonus is strictly excluded from V-Trace targets to prevent recursive bias.
        Args:
            rewards: [seq_len]
            v_preds_detached: [seq_len + 1, E]
            mu_probs: [seq_len]
            pi_probs: [seq_len]
        Returns:
            vs: [seq_len, E] per-head V-Trace value targets
            rhos: [seq_len] clipped importance sampling ratios
        """
        seq_len = rewards.size(0)
        num_heads = v_preds_detached.size(-1)
        vs = torch.zeros((seq_len, num_heads), device=self.device)

        ratios = pi_probs / (mu_probs + 1e-8)
        rhos = torch.clamp(ratios, max=rho_bar)
        cs = torch.clamp(ratios, max=c_bar)

        v_next_preds = v_preds_detached[1:]  # [seq_len, E]
        delta_Vs = rhos.unsqueeze(-1) * (
            rewards.unsqueeze(-1) + gamma * v_next_preds - v_preds_detached[:-1]
        )

        v_next = v_preds_detached[-1]  # [E]
        for t in reversed(range(seq_len)):
            v_s = v_preds_detached[t] + delta_Vs[t] + gamma * cs[t] * (v_next - v_next_preds[t])
            vs[t] = v_s
            v_next = v_s

        return vs, rhos

    def parse_state(self, raw_payload, target_action, topology_payload):
        raw_payload["target_action"] = target_action

        # Always refresh topology for the current trajectory's board (fixes stale map cache & board_width=0)
        if topology_payload is not None:
            self.env.state_parser.handle_topology(topology_payload)

        state, mask = self.env._parse_to_heterodata(raw_payload)
        if state is not None:
            state = state.to(self.device)
        return state, mask

    def learn_step(
        self,
        batch_size=4,
        sequence_length=16,
        rho_bar=1.0,
        c_bar=1.0,
        gamma=0.99,
        entropy_coef=0.01,
    ):
        batch_data = self.buffer.sample_batch(batch_size=batch_size, sequence_length=sequence_length)
        if not batch_data:
            return False

        # Annealing schedules for lambda_bc and lam_epistemic
        self.lambda_bc = compute_annealed_weight(
            self.global_step,
            initial=self.lambda_bc_initial,
            final=self.lambda_bc_final,
            warmup_steps=self.bc_warmup_steps,
            decay_steps=self.bc_decay_steps,
            schedule=self.bc_schedule,
        )
        self.lam_epistemic = compute_annealed_weight(
            self.global_step,
            initial=self.lam_epistemic_initial,
            final=self.lam_epistemic_final,
            warmup_steps=0,
            decay_steps=self.epistemic_decay_steps,
            schedule="linear",
        )

        self.agent.train()
        self.optimizer.zero_grad()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        valid_batches = 0
        total_game_rewards = []
        total_wins = []
        batch_env_rewards = []
        batch_bv_rewards = []
        batch_vp_rewards = []
        batch_total_rewards = []

        num_heads = len(self.agent.ensembles)

        for topology, steps, total_reward, win in batch_data:
            actual_seq_len = len(steps) - 1
            if actual_seq_len <= 0:
                continue

            valid_batches += 1
            total_game_rewards.append(total_reward)
            total_wins.append(1.0 if win else 0.0)

            transitions = steps[:-1]
            for s in transitions:
                batch_total_rewards.append(s.get("reward", 0.0))
                batch_env_rewards.append(s.get("env_reward", s.get("reward", 0.0)))
                batch_bv_rewards.append(s.get("reward_bv", 0.0))
                batch_vp_rewards.append(s.get("reward_vp", 0.0))

            rewards = torch.tensor(
                [s["reward"] for s in transitions], device=self.device, dtype=torch.float32
            )
            mu_probs = torch.tensor(
                [math.exp(float(s["mu_log_prob"])) for s in transitions],
                device=self.device,
                dtype=torch.float32,
            )

            graphs = []
            for s in steps:
                graph, _ = self.parse_state(s["raw_payload"], s.get("action_dict", {}), topology)
                graphs.append(graph)

            batched_graphs = Batch.from_data_list(graphs).to(self.device)

            # Single forward pass for Actor & Critic Ensemble over the sequence
            pi_log_probs, entropies_all, v_means, v_vars, v_preds_all, _, _ = (
                self.agent.evaluate_actions(batched_graphs)
            )

            log_pis = pi_log_probs[:actual_seq_len]
            pi_probs = torch.exp(log_pis)
            entropies = entropies_all[:actual_seq_len]

            is_last_terminal = steps[-1].get("is_terminal", False)
            v_preds_for_vtrace = v_preds_all.detach().clone()
            v_vars_for_vtrace = v_vars.detach().clone()
            if is_last_terminal:
                v_preds_for_vtrace[-1] = 0.0
                v_vars_for_vtrace[-1] = 0.0

            # Per-head V-Trace targets vs: [actual_seq_len, E]
            vs, rhos = self.compute_vtrace_targets(
                rewards,
                v_preds_for_vtrace,
                mu_probs,
                pi_probs.detach(),
                rho_bar=rho_bar,
                c_bar=c_bar,
                gamma=gamma,
            )

            # Actor Advantage with annealed 1-step epistemic exploration bonus
            vs_mean = vs.mean(dim=-1)  # [actual_seq_len]
            v_boot_last = v_preds_for_vtrace[-1:].mean(dim=-1)  # [1]
            v_next_mean = torch.cat([vs_mean[1:], v_boot_last])  # [actual_seq_len]
            sigma_next = torch.sqrt(v_vars_for_vtrace[1:] + 1e-8)  # [actual_seq_len]
            v_curr_mean = v_preds_for_vtrace[:-1].mean(dim=-1)  # [actual_seq_len]

            advantages = (
                rewards + gamma * (v_next_mean + self.lam_epistemic * sigma_next) - v_curr_mean
            )

            # Actor Loss
            actor_losses = -rhos * log_pis * advantages.detach()
            total_actor_loss += actor_losses.sum()

            # Critic Loss with independent Bernoulli bootstrap mask per trajectory/head
            bootstrap_mask = (
                torch.rand((1, num_heads), device=self.device) < self.bootstrap_prob
            ).float()
            if bootstrap_mask.sum() == 0:
                bootstrap_mask[0, random.randrange(num_heads)] = 1.0

            critic_losses = nn.functional.mse_loss(
                v_preds_all[:actual_seq_len], vs.detach(), reduction="none"
            )  # [actual_seq_len, E]
            masked_critic = (critic_losses * bootstrap_mask).sum(dim=-1) / bootstrap_mask.sum()
            total_critic_loss += masked_critic.sum()

            # Entropy Loss
            entropy_losses = -entropies
            total_entropy_loss += entropy_losses.sum()

            self.writer.add_scalar(
                "VTrace/Advantage_Mean", advantages.mean().item(), self.global_step
            )
            self.writer.add_scalar("VTrace/Rho_Mean", rhos.mean().item(), self.global_step)
            self.writer.add_scalar(
                "VTrace/Epistemic_Sigma_Mean", sigma_next.mean().item(), self.global_step
            )

        if valid_batches == 0:
            return False

        loss_rl = (
            total_actor_loss + 0.5 * total_critic_loss + entropy_coef * total_entropy_loss
        ) / valid_batches

        # Annealed Supervised Expert BC Co-Training Pass (skipped once lambda_bc -> 0)
        loss_bc = torch.tensor(0.0, device=self.device)
        if self.lambda_bc > 1e-5:
            expert_batch = self.expert_buffer.sample_batch(batch_size=self.expert_batch_size)
            if expert_batch is not None:
                expert_batch = expert_batch.to(self.device)
                pi_log_probs_exp, _, _, _, _, _, steps_exp = self.agent.evaluate_actions(
                    expert_batch
                )
                mask_exp = steps_exp > 0
                if mask_exp.any():
                    loss_bc = -pi_log_probs_exp[mask_exp].mean()

        loss_total = loss_rl + self.lambda_bc * loss_bc
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Logging
        self.writer.add_scalar("Loss/Total", loss_total.item(), self.global_step)
        self.writer.add_scalar("Loss/RL_Total", loss_rl.item(), self.global_step)
        self.writer.add_scalar(
            "Loss/Actor", (total_actor_loss / valid_batches).item(), self.global_step
        )
        self.writer.add_scalar(
            "Loss/Critic", (total_critic_loss / valid_batches).item(), self.global_step
        )
        self.writer.add_scalar(
            "Loss/Entropy", (total_entropy_loss / valid_batches).item(), self.global_step
        )
        self.writer.add_scalar("Loss/BC", loss_bc.item(), self.global_step)
        self.writer.add_scalar("Hyperparameters/Lambda_BC", self.lambda_bc, self.global_step)
        self.writer.add_scalar(
            "Hyperparameters/Lam_Epistemic", self.lam_epistemic, self.global_step
        )

        if len(total_game_rewards) > 0:
            avg_game_reward = sum(total_game_rewards) / len(total_game_rewards)
            self.writer.add_scalar(
                "System/Average_Game_Total_Reward", avg_game_reward, self.global_step
            )

        if len(total_wins) > 0:
            win_rate = sum(total_wins) / len(total_wins)
            self.writer.add_scalar("System/Win_Rate", win_rate, self.global_step)

        if len(batch_total_rewards) > 0:
            self.writer.add_scalar(
                "Reward/Total_Mean",
                sum(batch_total_rewards) / len(batch_total_rewards),
                self.global_step,
            )
            self.writer.add_scalar(
                "Reward/Env_Mean",
                sum(batch_env_rewards) / len(batch_env_rewards),
                self.global_step,
            )
            self.writer.add_scalar(
                "Reward/BV_Mean",
                sum(batch_bv_rewards) / len(batch_bv_rewards),
                self.global_step,
            )
            self.writer.add_scalar(
                "Reward/VP_Mean",
                sum(batch_vp_rewards) / len(batch_vp_rewards),
                self.global_step,
            )

        self.global_step += 1
        return True

    def run(self):
        logger.info(f"Starting IMPALA Learner on {self.device}...")
        try:
            while not self.stop_requested:
                with self.buffer.lock:
                    buf_size = len(self.buffer.trajectories)
                with self.expert_buffer.lock:
                    expert_buf_size = len(self.expert_buffer.graphs)

                self.writer.add_scalar("System/Buffer_Size", buf_size, self.global_step)
                self.writer.add_scalar(
                    "System/Expert_Buffer_Size", expert_buf_size, self.global_step
                )

                if self.learn_step(batch_size=self.batch_size):
                    if self.global_step % 10 == 0:
                        self.save_checkpoint()
                        logger.info(
                            f"Step {self.global_step}: Saved checkpoint (lambda_bc={self.lambda_bc:.4f}, lam_epistemic={self.lam_epistemic:.4f})."
                        )
                else:
                    time.sleep(2)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Gracefully shutting down Master...")
        finally:
            self.save_checkpoint()
            with self.buffer.lock:
                preserved_count = len(self.buffer.trajectories)
            logger.info(
                f"Final checkpoint saved at global_step={self.global_step}. "
                f"Preserved {preserved_count} unconsumed trajectory files on disk for next run."
            )
            self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Directory or comma-separated directories of agent trajectories",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run master trainer on")
    parser.add_argument(
        "--force-bootstrap",
        action="store_true",
        help="Force bootstrap from BC weights, overwriting impala latest weights",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="RL batch size (number of trajectory chunks)"
    )
    parser.add_argument(
        "--expert-batch-size",
        type=int,
        default=16,
        help="Expert BC batch size (number of state-action graphs)",
    )
    parser.add_argument(
        "--expert-dir",
        type=str,
        default="data/bc_trajectories",
        help="Directory of expert BC trajectories",
    )
    parser.add_argument("--lambda-bc", type=float, default=1.0, help="Initial weight for BC loss")
    parser.add_argument("--lambda-bc-final", type=float, default=0.0, help="Final weight for BC loss")
    parser.add_argument(
        "--bc-warmup-steps", type=int, default=500, help="Warmup steps before decaying lambda_bc"
    )
    parser.add_argument(
        "--bc-decay-steps", type=int, default=5000, help="Steps over which to decay lambda_bc"
    )
    parser.add_argument(
        "--bc-schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "exponential"],
        help="Annealing schedule for lambda_bc",
    )
    parser.add_argument(
        "--lam-epistemic", type=float, default=0.5, help="Initial weight for epistemic bonus"
    )
    parser.add_argument(
        "--lam-epistemic-final", type=float, default=0.0, help="Final weight for epistemic bonus"
    )
    parser.add_argument(
        "--epistemic-decay-steps",
        type=int,
        default=10000,
        help="Steps over which to decay lam_epistemic",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset_dirs = [d.strip() for d in args.dataset_dir.split(",")] if args.dataset_dir else None
    learner = ImpalaLearner(
        device=device,
        dataset_dirs=dataset_dirs,
        batch_size=args.batch_size,
        expert_batch_size=args.expert_batch_size,
        expert_dir=args.expert_dir,
        force_bootstrap=args.force_bootstrap,
        lambda_bc_initial=args.lambda_bc,
        lambda_bc_final=args.lambda_bc_final,
        bc_warmup_steps=args.bc_warmup_steps,
        bc_decay_steps=args.bc_decay_steps,
        bc_schedule=args.bc_schedule,
        lam_epistemic_initial=args.lam_epistemic,
        lam_epistemic_final=args.lam_epistemic_final,
        epistemic_decay_steps=args.epistemic_decay_steps,
    )
    def handle_master_signal(sig, frame):
        logger.info(f"Master received signal {sig}. Finishing current step and saving checkpoint...")
        learner.stop_requested = True

    signal.signal(signal.SIGINT, handle_master_signal)
    signal.signal(signal.SIGTERM, handle_master_signal)

    learner.run()
