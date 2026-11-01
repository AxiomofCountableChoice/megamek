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
            files_to_process = self.file_paths[
                worker_id * per_worker : (worker_id + 1) * per_worker
            ]

        if self.shuffle:
            files_to_process = files_to_process.copy()
            random.shuffle(files_to_process)

        for f in files_to_process:
            try:
                match_data = torch.load(f, weights_only=False, map_location="cpu")
                trajectories = match_data.get("trajectories", [])
                if len(trajectories) == 0:
                    continue

                # Compute returns chronologically in a canonical Player-0 frame
                step_rewards_p0 = []
                step_players = []
                prev_bv_p0, prev_bv_p1 = None, None
                prev_vp_p0 = 0.0
                total_initial_bv = None

                beta_bv = 0.1
                beta_vp = 1.0
                beta_tp = 0.0

                for data in trajectories:
                    if hasattr(data, "raw_rewards") and data.raw_rewards is not None:
                        rr = data.raw_rewards[0]
                        bv1 = float(rr[0].item())
                        bv2 = float(rr[1].item())
                        init_bv1 = float(rr[2].item())
                        init_bv2 = float(rr[3].item())
                        vp1 = float(rr[4].item())
                        tp1 = float(rr[6].item())
                        tp2 = float(rr[7].item())
                        player_id = int(round(float(rr[8].item())))
                    else:
                        ctx = data.global_context[0]
                        player_id = int(round(float(ctx[3].item())))
                        bv1 = float(ctx[6].item())
                        bv2 = float(ctx[7].item())
                        init_bv1 = 0.0
                        init_bv2 = 0.0
                        vp1 = float(ctx[8].item())
                        tp1 = float(ctx[10].item())
                        tp2 = float(ctx[11].item())

                    # Canonicalize to Player 0 perspective so alternating turns do not flip bv1/bv2
                    if player_id == 0:
                        bv_p0, bv_p1 = bv1, bv2
                        vp_p0 = vp1
                        tp_p0, tp_p1 = tp1, tp2
                    else:
                        bv_p0, bv_p1 = bv2, bv1
                        vp_p0 = -vp1
                        tp_p0, tp_p1 = tp2, tp1

                    if prev_bv_p0 is None:
                        prev_bv_p0, prev_bv_p1 = bv_p0, bv_p1
                        prev_vp_p0 = vp_p0
                        if (init_bv1 + init_bv2) > 0:
                            total_initial_bv = init_bv1 + init_bv2
                        else:
                            total_initial_bv = max(bv_p0 + bv_p1, 1.0)

                    delta_bv_p0 = bv_p0 - prev_bv_p0
                    delta_bv_p1 = bv_p1 - prev_bv_p1
                    delta_vp_p0 = vp_p0 - prev_vp_p0

                    r_p0 = (
                        beta_bv * 2.0 * (delta_bv_p0 - delta_bv_p1) / total_initial_bv
                        + beta_vp * delta_vp_p0
                        + beta_tp * (tp_p0 - tp_p1)
                    )
                    step_rewards_p0.append(r_p0)
                    step_players.append(player_id)

                    prev_bv_p0, prev_bv_p1 = bv_p0, bv_p1
                    prev_vp_p0 = vp_p0

                # Compute discounted returns G_t for each step using O(T) backward pass
                T = len(trajectories)
                gamma = 0.99
                g0 = 0.0
                g1 = 0.0
                returns_p0 = [0.0] * T
                returns_p1 = [0.0] * T
                for t in reversed(range(T)):
                    r_t_p0 = step_rewards_p0[t]
                    r_t_p1 = -step_rewards_p0[t]
                    g0 = r_t_p0 + gamma * g0
                    g1 = r_t_p1 + gamma * g1
                    returns_p0[t] = g0
                    returns_p1[t] = g1

                for t in range(T):
                    active_player = step_players[t]
                    g_t = returns_p0[t] if active_player == 0 else returns_p1[t]
                    trajectories[t].y_value = torch.tensor([g_t], dtype=torch.float32)

                if self.shuffle:
                    random.shuffle(trajectories)

                for data in trajectories:
                    if hasattr(data, "y_sequence"):
                        data.y_len = torch.tensor([data.y_sequence.size(0)], dtype=torch.long)
                    else:
                        data.y_len = torch.tensor([0], dtype=torch.long)

                    if hasattr(data, "action_mask"):
                        delattr(data, "action_mask")
                    if (
                        "weapon" in data.node_types
                        and hasattr(data["weapon"], "x")
                        and data["weapon"].x.size(0) > 0
                    ):
                        data["weapon"].x[data["weapon"].x < -1000.0] = 0.0

                    yield data
            except Exception as e:
                print(f"Error loading {f}: {e}")


def process_batch(
    agent, batch, device, is_training=False, accumulation_steps=32, return_random_baseline=False
):
    batch = batch.to(device)

    if return_random_baseline:
        (
            pi_log_prob_total,
            entropy_total,
            v_mean,
            v_variance,
            v_preds,
            correct_total,
            steps_total,
            random_correct_total,
        ) = agent.evaluate_actions(batch, return_random_baseline=True)
    else:
        (
            pi_log_prob_total,
            entropy_total,
            v_mean,
            v_variance,
            v_preds,
            correct_total,
            steps_total,
        ) = agent.evaluate_actions(batch, return_random_baseline=False)
        random_correct_total = None

    mask = steps_total > 0
    if not mask.any():
        if return_random_baseline:
            return 0.0, 0.0, 0.0, 0, 0, 0.0, False
        else:
            return 0.0, 0.0, 0.0, 0, 0, False

    loss_actor = -pi_log_prob_total[mask].mean()
    if hasattr(batch, "y_value") and batch.y_value is not None:
        v_preds_masked = v_preds[mask]  # [B_valid, E]
        y_target = batch.y_value[mask].view(-1, 1).expand_as(v_preds_masked)
        per_head_mse = F.mse_loss(v_preds_masked, y_target, reduction="none")
        if is_training:
            # Independent Bernoulli(0.63) bootstrap mask per sample and head
            boot_mask = (torch.rand_like(v_preds_masked) < 0.63).float()
            if boot_mask.sum() == 0:
                boot_mask[:, 0] = 1.0
            loss_critic = (per_head_mse * boot_mask).sum() / boot_mask.sum().clamp(min=1.0)
        else:
            loss_critic = per_head_mse.mean()
    else:
        loss_critic = torch.tensor(0.0, device=device)

    loss_total = (loss_actor + 0.5 * loss_critic) / accumulation_steps

    if is_training:
        loss_total.backward()

    if return_random_baseline:
        return (
            loss_total.item() * accumulation_steps,
            loss_actor.item(),
            loss_critic.item(),
            int(correct_total.sum().item()),
            int(steps_total.sum().item()),
            float(random_correct_total.sum().item()),
            True,
        )
    else:
        return (
            loss_total.item() * accumulation_steps,
            loss_actor.item(),
            loss_critic.item(),
            int(correct_total.sum().item()),
            int(steps_total.sum().item()),
            True,
        )


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    dataset_dir = "data/bc_trajectories"
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        print(
            f"Dataset directory not found or empty at {dataset_dir}. Please run bc_generator.py first."
        )
        return

    all_files = glob.glob(os.path.join(dataset_dir, "*.pt"))
    all_files.sort()
    random.seed(42)
    random.shuffle(all_files)

    val_size = int(len(all_files) * 0.2)
    val_files = all_files[:val_size]
    train_files = all_files[val_size:]

    print(
        f"Dataset split by files: {len(train_files)} training files, {len(val_files)} validation files."
    )

    train_dataset = BCIterableDataset(train_files, shuffle=True)
    val_dataset = BCIterableDataset(val_files, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=16)
    val_loader = DataLoader(val_dataset, batch_size=16)

    writer = SummaryWriter(log_dir="runs/bc_training_logs")

    agent = MegaMekAgent(hidden_dim=128).to(device)
    optimizer = Adam(agent.parameters(), lr=3e-4)

    accumulation_steps = 64

    epochs = 15
    os.makedirs("model_objects", exist_ok=True)
    save_path = "model_objects/bc_agent.pt"

    start_epoch = 0
    epoch_checkpoints = glob.glob("model_objects/bc_agent_epoch_*.pt")
    if len(epoch_checkpoints) > 0:
        epoch_nums = []
        for cp in epoch_checkpoints:
            try:
                base = os.path.basename(cp)
                num = int(base.replace("bc_agent_epoch_", "").replace(".pt", ""))
                epoch_nums.append((num, cp))
            except ValueError:
                pass
        if len(epoch_nums) > 0:
            epoch_nums.sort(key=lambda x: x[0], reverse=True)
            last_epoch, last_cp = epoch_nums[0]
            print(
                f"Found existing epoch checkpoint: {last_cp}. Auto-resuming from epoch {last_epoch + 1}...",
                flush=True,
            )
            agent.load_state_dict(torch.load(last_cp, map_location=device), strict=False)
            start_epoch = last_epoch

    for epoch in range(start_epoch, epochs):
        agent.train()
        total_train_loss = 0.0
        total_train_actor_loss = 0.0
        total_train_critic_loss = 0.0
        total_train_correct = 0
        total_train_steps = 0
        valid_train_batches = 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            loss_val, actor_l, critic_l, correct, steps, valid = process_batch(
                agent, batch, device, is_training=True, accumulation_steps=accumulation_steps
            )
            if valid:
                total_train_loss += loss_val
                total_train_actor_loss += actor_l
                total_train_critic_loss += critic_l
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
        avg_train_actor_loss = (
            total_train_actor_loss / valid_train_batches if valid_train_batches > 0 else 0
        )
        avg_train_critic_loss = (
            total_train_critic_loss / valid_train_batches if valid_train_batches > 0 else 0
        )
        train_acc = total_train_correct / total_train_steps if total_train_steps > 0 else 0

        agent.eval()
        total_val_loss = 0.0
        total_val_actor_loss = 0.0
        total_val_critic_loss = 0.0
        total_val_correct = 0
        total_val_random_correct = 0.0
        total_val_steps = 0
        valid_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                loss_val, actor_l, critic_l, correct, steps, rand_correct, valid = process_batch(
                    agent, batch, device, is_training=False, return_random_baseline=True
                )
                if valid:
                    total_val_loss += loss_val
                    total_val_actor_loss += actor_l
                    total_val_critic_loss += critic_l
                    total_val_correct += correct
                    total_val_random_correct += rand_correct
                    total_val_steps += steps
                    valid_val_batches += 1

        avg_val_loss = total_val_loss / valid_val_batches if valid_val_batches > 0 else 0
        avg_val_actor_loss = (
            total_val_actor_loss / valid_val_batches if valid_val_batches > 0 else 0
        )
        avg_val_critic_loss = (
            total_val_critic_loss / valid_val_batches if valid_val_batches > 0 else 0
        )
        val_acc = total_val_correct / total_val_steps if total_val_steps > 0 else 0
        val_rand_acc = total_val_random_correct / total_val_steps if total_val_steps > 0 else 0

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} (Actor: {avg_train_actor_loss:.4f}, Critic: {avg_train_critic_loss:.4f}) Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} (Actor: {avg_val_actor_loss:.4f}, Critic: {avg_val_critic_loss:.4f}) Acc: {val_acc:.4f} (Random Baseline Acc: {val_rand_acc:.4f})"
        )

        writer.add_scalars(
            "Loss",
            {
                "Train_Total": avg_train_loss,
                "Train_Actor": avg_train_actor_loss,
                "Train_Critic": avg_train_critic_loss,
                "Val_Total": avg_val_loss,
                "Val_Actor": avg_val_actor_loss,
                "Val_Critic": avg_val_critic_loss,
            },
            epoch,
        )
        writer.add_scalars(
            "Accuracy", {"Train": train_acc, "Val": val_acc, "Val_Random": val_rand_acc}, epoch
        )

        torch.save(agent.state_dict(), f"model_objects/bc_agent_epoch_{epoch+1}.pt")
        torch.save(agent.state_dict(), save_path)

    writer.close()
    print(f"Saved trained BC model to {save_path}")


if __name__ == "__main__":
    train()
