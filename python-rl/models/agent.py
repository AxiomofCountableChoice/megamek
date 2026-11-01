import torch
import torch.nn as nn
from models.encoder import MegaMekHGTEncoder
from models.actor import ActionConditionedPointer
from state_parser import StateParser


class MegaMekAgent(nn.Module):
    """
    Main IMPALA Actor-Critic Module defined in ARCHITECTURE.md (Sections 3c & 3d).
    Contains:
      - The HGT + Perceiver Spatial Encoder.
      - The Value Ensemble (E=8 by default for Epistemic Exploration).
      - The Autoregressive Pointer Decoder (Action head) operating over the
        Unified Hierarchical Action Tree on HeteroData['action'].
    """

    def __init__(self, hidden_dim=128, ensemble_size=8, action_feature_dim=12):
        super().__init__()
        num_latent_queries = 32
        self.encoder = MegaMekHGTEncoder(
            hidden_dim=hidden_dim, num_latent_queries=num_latent_queries, num_layers=4
        )

        # 1. Epistemic Value Ensemble (Critic)
        # Branching from z (dim = (num_latent_queries * hidden_dim) + hidden_dim)
        latent_dim = (num_latent_queries * hidden_dim) + hidden_dim

        self.ensembles = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(ensemble_size)
        ])

        # 2. Autoregressive Transformer Decoder (Teacher Forcing & Action Scoring)
        self.actor_pointer = ActionConditionedPointer(
            hidden_dim=hidden_dim,
            action_feature_dim=action_feature_dim,
            latent_dim=latent_dim,
        )

    def compute_values(self, z, return_all=False):
        """
        Computes the V-Trace Value Ensembles (Section 2c).
        """
        v_preds = torch.stack([critic(z) for critic in self.ensembles], dim=-1)
        # v_preds shape: [batch, 1, E]
        v_mean = v_preds.mean(dim=-1)
        v_var = v_preds.var(dim=-1, unbiased=False)
        if return_all:
            return v_mean, v_var, v_preds.squeeze(1)
        return v_mean, v_var

    @staticmethod
    def _get_valid_step_mask(parent_idx, next_parent_idx, exclusivity_group, chosen_seq_indices):
        """
        Phase-agnostic topological mask computation over the Unified Hierarchical Action Tree.
        Returns a boolean tensor of shape (num_actions,) indicating which action nodes are
        valid candidates at step k = len(chosen_seq_indices).
        """
        num_actions = parent_idx.size(0)
        device = parent_idx.device
        if num_actions == 0:
            return torch.zeros(0, dtype=torch.bool, device=device)

        if not chosen_seq_indices:
            return parent_idx == -1

        last_idx = int(chosen_seq_indices[-1])
        if last_idx < 0 or last_idx >= num_actions:
            return torch.zeros(num_actions, dtype=torch.bool, device=device)

        active_parent = int(next_parent_idx[last_idx].item())
        if active_parent == -2:
            # Terminal leaf reached
            return torch.zeros(num_actions, dtype=torch.bool, device=device)

        valid_mask = (parent_idx == active_parent).clone()
        if not valid_mask.any():
            return valid_mask

        used_groups = set()
        for idx in chosen_seq_indices:
            eg = int(exclusivity_group[int(idx)].item())
            if eg >= 0:
                used_groups.add(eg)

        if used_groups:
            candidate_indices = torch.where(valid_mask)[0].tolist()
            for c in candidate_indices:
                eg_c = int(exclusivity_group[c].item())
                if eg_c >= 0 and eg_c in used_groups:
                    valid_mask[c] = False
                elif int(next_parent_idx[c].item()) == c:
                    # Internal branch node (e.g. SELECT_TARGET): prune if all children are exhausted
                    child_mask = parent_idx == c
                    if child_mask.any():
                        child_egs = exclusivity_group[child_mask].tolist()
                        if all(eg >= 0 and eg in used_groups for eg in child_egs):
                            valid_mask[c] = False

        return valid_mask

    def evaluate_actions(self, batch, return_random_baseline=False):
        """
        Evaluates a HeteroData batch containing teacher-forcing y_sequence target actions.
        Uses the exact same _get_valid_step_mask tree traversal as get_action().
        Returns:
            pi_log_prob_total: [batch_size]
            entropy_total: [batch_size]
            v_mean: [batch_size]
            v_variance: [batch_size]
            v_preds: [batch_size, E]
            correct_total: [batch_size]
            steps_total: [batch_size]
            [random_correct_total]: [batch_size] (if return_random_baseline=True)
        """
        z, x_dict = self.encoder(batch)

        # Critic ensemble values
        v_preds_raw = torch.stack([critic(z) for critic in self.ensembles], dim=-1)  # [B, 1, E]
        v_preds = v_preds_raw.squeeze(1)  # [B, E]
        v_mean = v_preds.mean(dim=-1)  # [B]
        v_variance = v_preds.var(dim=-1, unbiased=False)  # [B]

        batch_size = z.size(0)
        pi_log_prob_total = torch.zeros(batch_size, device=z.device)
        entropy_total = torch.zeros(batch_size, device=z.device)
        correct_total = torch.zeros(batch_size, device=z.device)
        steps_total = torch.zeros(batch_size, device=z.device)
        random_correct_total = torch.zeros(batch_size, device=z.device)

        y_seq_concat = getattr(batch, "y_sequence", None)
        y_lens = getattr(batch, "y_len", None)

        if y_seq_concat is None or y_seq_concat.size(0) == 0:
            ret = [pi_log_prob_total, entropy_total, v_mean, v_variance, v_preds, correct_total, steps_total]
            if return_random_baseline:
                ret.append(random_correct_total)
            return tuple(ret)

        e_actions_batched = self.actor_pointer.compute_action_embeddings(x_dict, batch)
        if e_actions_batched.size(0) == 0:
            ret = [pi_log_prob_total, entropy_total, v_mean, v_variance, v_preds, correct_total, steps_total]
            if return_random_baseline:
                ret.append(random_correct_total)
            return tuple(ret)

        action_batch_idx = (
            batch["action"].batch
            if hasattr(batch["action"], "batch") and batch["action"].batch is not None
            else torch.zeros(e_actions_batched.size(0), dtype=torch.long, device=z.device)
        )
        parent_idx_concat = batch["action"].parent_idx
        next_parent_idx_concat = batch["action"].next_parent_idx
        exclusivity_group_concat = batch["action"].exclusivity_group

        y_offset = 0
        for b in range(batch_size):
            y_len = (
                int(y_lens[b].item())
                if y_lens is not None
                else (y_seq_concat.size(0) if y_seq_concat.dim() > 0 else 0)
            )
            if y_len == 0:
                continue

            y_seq = y_seq_concat[y_offset : y_offset + y_len]
            y_offset += y_len

            mask_b = action_batch_idx == b
            e_actions = e_actions_batched[mask_b]
            parent_idx = parent_idx_concat[mask_b]
            next_parent_idx = next_parent_idx_concat[mask_b]
            exclusivity_group = exclusivity_group_concat[mask_b]

            chosen_seq_indices = []
            chosen_embeddings = []

            for k in range(y_len):
                target_idx = int(y_seq[k].item())
                valid_mask = self._get_valid_step_mask(
                    parent_idx, next_parent_idx, exclusivity_group, chosen_seq_indices
                )

                if not valid_mask.any() or target_idx < 0 or target_idx >= valid_mask.size(0):
                    break
                if not valid_mask[target_idx]:
                    break

                history_tensor = (
                    None
                    if len(chosen_embeddings) == 0
                    else torch.stack(chosen_embeddings).unsqueeze(0)
                )
                s_k = self.actor_pointer.decode_sequence(z[b : b + 1], history_tensor)

                e_tier = e_actions[valid_mask]
                num_candidates = e_tier.size(0)
                if num_candidates > 0:
                    random_correct_total[b] += 1.0 / num_candidates

                act_batch_idx_b = torch.full((num_candidates,), b, dtype=torch.long, device=z.device)
                logits = self.actor_pointer.score_actions(s_k, e_tier, act_batch_idx_b, x_dict, batch)
                log_probs = torch.log_softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)

                global_indices = torch.where(valid_mask)[0]
                relative_target = (global_indices == target_idx).nonzero(as_tuple=True)[0]
                if relative_target.numel() == 0:
                    break

                rel_idx = int(relative_target.item())
                pi_log_prob_total[b] += log_probs[rel_idx]
                entropy_total[b] += -(probs * log_probs).sum()

                if int(logits.argmax().item()) == rel_idx:
                    correct_total[b] += 1
                steps_total[b] += 1

                chosen_seq_indices.append(target_idx)
                chosen_embeddings.append(e_actions[target_idx])

        ret = [pi_log_prob_total, entropy_total, v_mean, v_variance, v_preds, correct_total, steps_total]
        if return_random_baseline:
            ret.append(random_correct_total)
        return tuple(ret)

    def _decode_and_sample_step(
        self, z, history_embeddings, e_actions, valid_mask, batch_idx, hetero_data, x_dict, deterministic=False
    ):
        """
        Decodes the sequence using the history of chosen action embeddings,
        scores the valid candidate mask, and samples the selected action node index.
        """
        if not valid_mask.any():
            return None, 0.0, torch.empty((0,), device=z.device)

        history_tensor = (
            None
            if not history_embeddings
            else torch.stack(history_embeddings, dim=1)  # [1, len, hidden_dim]
        )
        s_k = self.actor_pointer.decode_sequence(z, history_tensor)

        indices = torch.where(valid_mask)[0]
        e_candidates = e_actions[indices]
        cand_batch_idx = batch_idx[indices] if batch_idx is not None else None

        logits = self.actor_pointer.score_actions(s_k, e_candidates, cand_batch_idx, x_dict, hetero_data)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        if deterministic:
            idx = int(torch.argmax(logits).item())
            log_prob = float(log_probs[idx].item())
        else:
            dist = torch.distributions.Categorical(logits=logits)
            sampled = dist.sample()
            idx = int(sampled.item())
            log_prob = float(log_probs[idx].item())

        selected_global_idx = int(indices[idx].item())
        return selected_global_idx, log_prob, probs

    def _sample_agnostic_trajectory(
        self, z, e_actions, batch_idx, hetero_data, x_dict, deterministic=False
    ):
        """
        Phase-agnostically traverses the Unified Hierarchical Action Tree using _get_valid_step_mask.
        """
        action_store = hetero_data["action"]
        parent_idx = action_store.parent_idx
        next_parent_idx = action_store.next_parent_idx
        exclusivity_group = action_store.exclusivity_group

        chosen_seq_indices = []
        history_embeddings = []
        total_log_prob = 0.0

        max_steps = 32
        for _ in range(max_steps):
            valid_mask = self._get_valid_step_mask(
                parent_idx, next_parent_idx, exclusivity_group, chosen_seq_indices
            )
            if not valid_mask.any():
                break

            selected_idx, log_prob, _ = self._decode_and_sample_step(
                z,
                history_embeddings,
                e_actions,
                valid_mask,
                batch_idx,
                hetero_data,
                x_dict,
                deterministic=deterministic,
            )
            if selected_idx is None:
                break

            chosen_seq_indices.append(selected_idx)
            total_log_prob += log_prob
            history_embeddings.append(e_actions[selected_idx].unsqueeze(0))

        return chosen_seq_indices, total_log_prob

    def _beam_search_agnostic(
        self, z, e_actions, batch_idx, hetero_data, x_dict, beam_width=3
    ):
        """
        Phase-agnostic beam search over the Unified Hierarchical Action Tree using _get_valid_step_mask.
        """
        action_store = hetero_data["action"]
        parent_idx = action_store.parent_idx
        next_parent_idx = action_store.next_parent_idx
        exclusivity_group = action_store.exclusivity_group

        root_mask = self._get_valid_step_mask(parent_idx, next_parent_idx, exclusivity_group, [])
        if not root_mask.any():
            return [], 0.0

        s_0 = self.actor_pointer.decode_sequence(z, None)
        root_indices = torch.where(root_mask)[0]
        e_cands = e_actions[root_indices]
        cand_batch_idx = batch_idx[root_indices] if batch_idx is not None else None
        logits_0 = self.actor_pointer.score_actions(s_0, e_cands, cand_batch_idx, x_dict, hetero_data)
        log_probs_0 = torch.log_softmax(logits_0, dim=-1)

        beams = []
        for i, node_idx_t in enumerate(root_indices):
            node_idx = int(node_idx_t.item())
            lp = float(log_probs_0[i].item())
            beams.append({
                "trajectory": [node_idx],
                "history": [e_actions[node_idx].unsqueeze(0)],
                "log_prob_sum": lp,
                "done": False,
            })

        beams = sorted(beams, key=lambda x: x["log_prob_sum"], reverse=True)[:beam_width]

        max_steps = 32
        for _ in range(max_steps):
            if all(b["done"] for b in beams):
                break
            new_beams = []
            for b in beams:
                if b["done"]:
                    new_beams.append(b)
                    continue

                valid_mask = self._get_valid_step_mask(
                    parent_idx, next_parent_idx, exclusivity_group, b["trajectory"]
                )
                if not valid_mask.any():
                    b["done"] = True
                    new_beams.append(b)
                    continue

                s_k = self.actor_pointer.decode_sequence(z, torch.stack(b["history"], dim=1))
                child_indices = torch.where(valid_mask)[0]
                e_cands = e_actions[child_indices]
                cand_batch_idx = batch_idx[child_indices] if batch_idx is not None else None
                logits_k = self.actor_pointer.score_actions(s_k, e_cands, cand_batch_idx, x_dict, hetero_data)
                log_probs_k = torch.log_softmax(logits_k, dim=-1)

                for i, child_idx_t in enumerate(child_indices):
                    child_idx = int(child_idx_t.item())
                    lp = float(log_probs_k[i].item())
                    is_terminal = int(next_parent_idx[child_idx].item()) == -2
                    new_beams.append({
                        "trajectory": b["trajectory"] + [child_idx],
                        "history": b["history"] + [e_actions[child_idx].unsqueeze(0)],
                        "log_prob_sum": b["log_prob_sum"] + lp,
                        "done": is_terminal,
                    })

            beams = sorted(new_beams, key=lambda x: x["log_prob_sum"], reverse=True)[:beam_width]

        best_beam = beams[0] if beams else None
        if not best_beam:
            return [], 0.0
        return best_beam["trajectory"], best_beam["log_prob_sum"]

    @staticmethod
    def _decode_trajectory(hetero_data, chosen_seq_indices):
        """
        Decodes a sequence of chosen action node indices into the MegaMek RLActionResponse dict.
        """
        phase_type = int(round(float(hetero_data.global_context[0, 2].item())))
        if not chosen_seq_indices or hetero_data["action"].x.size(0) == 0:
            if phase_type == 1:
                return {"selected_entity_id": -1, "twist": 0, "attacks": []}
            if phase_type == 2:
                return {"selected_entity_id": -1, "attack": None}
            return {"selected_entity_id": -1, "selected_path_index": -1}

        action_store = hetero_data["action"]
        action_types = action_store.action_type_id
        source_units = action_store.source_unit_idx
        raw_meta = action_store.raw_meta

        selected_entity_id = int(source_units[chosen_seq_indices[0]].item())

        if phase_type == 0:  # MOVEMENT or DEPLOYMENT
            selected_path_index = -1
            for idx in chosen_seq_indices:
                if int(action_types[idx].item()) == StateParser.ACTION_MOVE_PATH:
                    selected_path_index = int(raw_meta[idx, 0].item())
                    break
            return {
                "selected_entity_id": selected_entity_id,
                "selected_path_index": selected_path_index,
            }

        if phase_type == 1:  # WEAPON
            twist = 0
            attacks = []
            for idx in chosen_seq_indices:
                act_type = int(action_types[idx].item())
                if act_type == StateParser.ACTION_TORSO_TWIST:
                    twist = int(raw_meta[idx, 1].item())
                elif act_type == StateParser.ACTION_FIRE_WEAPON:
                    attacks.append({
                        "target_id": int(raw_meta[idx, 2].item()),
                        "weapon_id": int(raw_meta[idx, 3].item()),
                    })
            return {
                "selected_entity_id": selected_entity_id,
                "twist": twist,
                "attacks": attacks,
            }

        if phase_type == 2:  # PHYSICAL
            attack = None
            for idx in chosen_seq_indices:
                act_type = int(action_types[idx].item())
                if act_type == StateParser.ACTION_PHYSICAL_ATTACK:
                    attack = {
                        "target_id": int(raw_meta[idx, 2].item()),
                        "action_type": int(raw_meta[idx, 3].item()),
                    }
                    break
            return {
                "selected_entity_id": selected_entity_id,
                "attack": attack,
            }

        return {"selected_entity_id": selected_entity_id, "selected_path_index": -1}

    def compute_ui_action_probabilities(self, hetero_data, mask_tree):
        """
        Computes exact hierarchical probabilities over the Unified Action Tree
        and annotates `mask_tree` in-place for `interactive_debugger.py`.
        Returns a list of path probabilities (for Movement) or a dict of node probabilities.
        """
        if (
            hetero_data is None
            or "action" not in hetero_data.node_types
            or hetero_data["action"].x is None
            or hetero_data["action"].x.size(0) == 0
        ):
            return []

        with torch.no_grad():
            z, x_dict = self.encoder(hetero_data)
            e_actions = self.actor_pointer.compute_action_embeddings(x_dict, hetero_data)
            action_store = hetero_data["action"]
            parent_idx = action_store.parent_idx
            next_parent_idx = action_store.next_parent_idx
            exclusivity_group = action_store.exclusivity_group
            action_types = action_store.action_type_id
            source_units = action_store.source_unit_idx
            raw_meta = action_store.raw_meta
            phase_type = int(round(float(hetero_data.global_context[0, 2].item())))

            def score_step(seq_indices):
                vmask = self._get_valid_step_mask(parent_idx, next_parent_idx, exclusivity_group, seq_indices)
                if not vmask.any():
                    return {}, []
                hist = None if not seq_indices else torch.stack([e_actions[i] for i in seq_indices]).unsqueeze(0)
                s_k = self.actor_pointer.decode_sequence(z, hist)
                cands = torch.where(vmask)[0]
                logits = self.actor_pointer.score_actions(s_k, e_actions[cands], None, x_dict, hetero_data)
                probs = torch.softmax(logits, dim=-1)
                return {int(idx.item()): float(p.item()) for idx, p in zip(cands, probs)}, [int(i.item()) for i in cands]

            # Step 0: SELECT_UNIT
            unit_probs, unit_nodes = score_step([])
            unit_node_by_entity = {int(source_units[u].item()): u for u in unit_nodes}

            if phase_type == 0:  # MOVEMENT
                valid_paths = mask_tree.get("valid_paths", []) if isinstance(mask_tree, dict) else []
                path_probs = [0.0] * len(valid_paths)
                for u_node in unit_nodes:
                    u_prob = unit_probs.get(u_node, 0.0)
                    hex_probs, hex_nodes = score_step([u_node])
                    for h_node in hex_nodes:
                        h_prob = hex_probs.get(h_node, 0.0)
                        p_probs, p_nodes = score_step([u_node, h_node])
                        for p_node in p_nodes:
                            p_prob = p_probs.get(p_node, 0.0)
                            p_idx = int(raw_meta[p_node, 0].item())
                            joint_prob = u_prob * h_prob * p_prob
                            if 0 <= p_idx < len(path_probs):
                                path_probs[p_idx] = joint_prob
                                if isinstance(valid_paths[p_idx], dict):
                                    valid_paths[p_idx]["unit_prob"] = u_prob
                                    valid_paths[p_idx]["hex_prob"] = h_prob
                                    valid_paths[p_idx]["facing_prob"] = p_prob
                                    valid_paths[p_idx]["prob"] = joint_prob
                return path_probs

            elif phase_type == 1:  # WEAPON
                node_prob_map = {str(k): v for k, v in unit_probs.items()}
                valid_twists = mask_tree.get("valid_twists", []) if isinstance(mask_tree, dict) else []
                for u_node in unit_nodes:
                    u_prob = unit_probs.get(u_node, 0.0)
                    u_ent = int(source_units[u_node].item())
                    t_probs, t_nodes = score_step([u_node])
                    for t_node in t_nodes:
                        t_prob = t_probs.get(t_node, 0.0)
                        t_val = int(raw_meta[t_node, 1].item())
                        node_prob_map[str(t_node)] = u_prob * t_prob

                        # Match twist dict in mask_tree
                        twist_dict = next(
                            (
                                tw
                                for tw in valid_twists
                                if tw.get("source_entity_index", u_ent) == u_ent and int(tw.get("twist", 0)) == t_val
                            ),
                            None,
                        )
                        if twist_dict is not None:
                            twist_dict["node_idx"] = t_node
                            twist_dict["unit_prob"] = u_prob
                            twist_dict["prob"] = u_prob * t_prob
                            twist_dict["children"] = []

                        step2_probs, step2_nodes = score_step([u_node, t_node])
                        for s2_node in step2_nodes:
                            s2_prob = step2_probs.get(s2_node, 0.0)
                            node_prob_map[str(s2_node)] = s2_prob
                            act_t = int(action_types[s2_node].item())
                            if act_t == StateParser.ACTION_END_DECLARATION:
                                if twist_dict is not None:
                                    twist_dict["end_prob"] = s2_prob
                                    twist_dict["children"].append({"type": "END", "node_idx": s2_node, "prob": s2_prob})
                            elif act_t == StateParser.ACTION_SELECT_TARGET:
                                tgt_ent = int(raw_meta[s2_node, 2].item())
                                tgt_dict = None
                                if twist_dict is not None:
                                    tgt_dict = next(
                                        (
                                            tg
                                            for tg in twist_dict.get("valid_targets", [])
                                            if int(tg.get("target_entity_index", -1)) == tgt_ent
                                        ),
                                        None,
                                    )
                                if tgt_dict is not None:
                                    tgt_dict["node_idx"] = s2_node
                                    tgt_dict["prob"] = s2_prob

                                w_probs, w_nodes = score_step([u_node, t_node, s2_node])
                                for w_node in w_nodes:
                                    w_prob = w_probs.get(w_node, 0.0)
                                    node_prob_map[str(w_node)] = w_prob
                                    w_id = int(raw_meta[w_node, 3].item())
                                    if tgt_dict is not None:
                                        w_dict = next(
                                            (
                                                wp
                                                for wp in tgt_dict.get("valid_weapons", [])
                                                if int(wp.get("weapon_id", -1)) == w_id
                                            ),
                                            None,
                                        )
                                        if w_dict is not None:
                                            w_dict["node_idx"] = w_node
                                            w_dict["prob"] = w_prob
                return node_prob_map

            elif phase_type == 2:  # PHYSICAL
                node_prob_map = {str(k): v for k, v in unit_probs.items()}
                valid_targets = mask_tree.get("valid_targets", []) if isinstance(mask_tree, dict) else []
                for u_node in unit_nodes:
                    u_prob = unit_probs.get(u_node, 0.0)
                    u_ent = int(source_units[u_node].item())
                    s1_probs, s1_nodes = score_step([u_node])
                    for s1_node in s1_nodes:
                        s1_prob = s1_probs.get(s1_node, 0.0)
                        node_prob_map[str(s1_node)] = u_prob * s1_prob
                        act_t = int(action_types[s1_node].item())
                        if act_t == StateParser.ACTION_SELECT_TARGET:
                            tgt_ent = int(raw_meta[s1_node, 2].item())
                            tgt_dict = next(
                                (
                                    tg
                                    for tg in valid_targets
                                    if tg.get("source_entity_index", u_ent) == u_ent
                                    and int(tg.get("target_entity_index", -1)) == tgt_ent
                                ),
                                None,
                            )
                            if tgt_dict is not None:
                                tgt_dict["node_idx"] = s1_node
                                tgt_dict["prob"] = u_prob * s1_prob
                            p_probs, p_nodes = score_step([u_node, s1_node])
                            for p_node in p_nodes:
                                p_prob = p_probs.get(p_node, 0.0)
                                node_prob_map[str(p_node)] = p_prob
                                p_type = int(raw_meta[p_node, 3].item())
                                if tgt_dict is not None:
                                    att_dict = next(
                                        (
                                            at
                                            for at in tgt_dict.get("valid_attacks", [])
                                            if int(at.get("action_type", -1)) == p_type
                                        ),
                                        None,
                                    )
                                    if att_dict is not None:
                                        att_dict["node_idx"] = p_node
                                        att_dict["prob"] = p_prob
                return node_prob_map

            return []

    def score_partial_weapon_trajectory(self, state_graph, mask_tree, hetero_data, partial_action):
        """
        Scores the next valid choices in the Unified Action Tree given a partial weapon action dict.
        Also computes conditional probabilities for the FIRE_WEAPON children of each valid SELECT_TARGET.
        """
        target_Graph = state_graph if hasattr(state_graph, "node_types") else hetero_data
        with torch.no_grad():
            z, x_dict = self.encoder(target_Graph)
            e_actions = self.actor_pointer.compute_action_embeddings(x_dict, target_Graph)
            action_store = target_Graph["action"]
            parent_idx = action_store.parent_idx
            next_parent_idx = action_store.next_parent_idx
            exclusivity_group = action_store.exclusivity_group
            action_types = action_store.action_type_id
            source_units = action_store.source_unit_idx
            raw_meta = action_store.raw_meta

            chosen_seq = []
            valid_mask = self._get_valid_step_mask(parent_idx, next_parent_idx, exclusivity_group, chosen_seq)
            if not valid_mask.any():
                return {}

            # 1. Match SELECT_UNIT
            req_unit = partial_action.get("selected_entity_id", partial_action.get("selected_entity_index"))
            root_cands = torch.where(valid_mask)[0].tolist()
            unit_node = None
            if req_unit is not None:
                unit_node = next((i for i in root_cands if int(source_units[i].item()) == int(req_unit)), None)
            if unit_node is None and root_cands:
                unit_node = root_cands[0]
            if unit_node is None:
                return {}
            chosen_seq.append(unit_node)

            # 2. Match TORSO_TWIST if provided
            if "twist" in partial_action and partial_action["twist"] is not None:
                valid_mask = self._get_valid_step_mask(parent_idx, next_parent_idx, exclusivity_group, chosen_seq)
                twist_cands = torch.where(valid_mask)[0].tolist()
                twist_val = int(partial_action["twist"])
                twist_node = next(
                    (
                        i
                        for i in twist_cands
                        if int(action_types[i].item()) == StateParser.ACTION_TORSO_TWIST
                        and int(raw_meta[i, 1].item()) == twist_val
                    ),
                    None,
                )
                if twist_node is None:
                    return {}
                chosen_seq.append(twist_node)

                for att in partial_action.get("attacks", []):
                    t_id = int(att.get("target_id", -1))
                    w_id = int(att.get("weapon_id", -1))
                    valid_mask = self._get_valid_step_mask(parent_idx, next_parent_idx, exclusivity_group, chosen_seq)
                    t_cands = torch.where(valid_mask)[0].tolist()
                    t_node = next(
                        (
                            i
                            for i in t_cands
                            if int(action_types[i].item()) == StateParser.ACTION_SELECT_TARGET
                            and int(raw_meta[i, 2].item()) == t_id
                        ),
                        None,
                    )
                    if t_node is None:
                        return {}
                    chosen_seq.append(t_node)

                    valid_mask = self._get_valid_step_mask(parent_idx, next_parent_idx, exclusivity_group, chosen_seq)
                    w_cands = torch.where(valid_mask)[0].tolist()
                    w_node = next(
                        (
                            i
                            for i in w_cands
                            if int(action_types[i].item()) == StateParser.ACTION_FIRE_WEAPON
                            and int(raw_meta[i, 3].item()) == w_id
                        ),
                        None,
                    )
                    if w_node is None:
                        return {}
                    chosen_seq.append(w_node)

            valid_mask = self._get_valid_step_mask(parent_idx, next_parent_idx, exclusivity_group, chosen_seq)
            if not valid_mask.any():
                return {}

            history_embeddings = torch.stack([e_actions[idx] for idx in chosen_seq]).unsqueeze(0)
            s_next = self.actor_pointer.decode_sequence(z, history_embeddings)
            child_indices = torch.where(valid_mask)[0]
            logits = self.actor_pointer.score_actions(s_next, e_actions[child_indices], None, x_dict, target_Graph)
            probs = torch.softmax(logits, dim=-1)
            result = {str(int(idx.item())): float(p.item()) for idx, p in zip(child_indices, probs)}

            # Also score FIRE_WEAPON children of each valid SELECT_TARGET candidate
            for idx_t in child_indices:
                c_idx = int(idx_t.item())
                if int(action_types[c_idx].item()) == StateParser.ACTION_SELECT_TARGET:
                    sub_seq = chosen_seq + [c_idx]
                    sub_mask = self._get_valid_step_mask(parent_idx, next_parent_idx, exclusivity_group, sub_seq)
                    if sub_mask.any():
                        sub_hist = torch.stack([e_actions[i] for i in sub_seq]).unsqueeze(0)
                        s_sub = self.actor_pointer.decode_sequence(z, sub_hist)
                        w_cands = torch.where(sub_mask)[0]
                        w_logits = self.actor_pointer.score_actions(s_sub, e_actions[w_cands], None, x_dict, target_Graph)
                        w_probs = torch.softmax(w_logits, dim=-1)
                        for w_idx, w_p in zip(w_cands, w_probs):
                            result[str(int(w_idx.item()))] = float(w_p.item())

            return result

    def get_action(self, hetero_data, mask_tree, deterministic=False):
        """
        Forward pass of the agent for inference/sampling.
        Supports deterministic (beam search) and stochastic (categorical) action selection.
        Returns: response_dict, v_mean, probs, mu_log_prob
        """
        z, x_dict = self.encoder(hetero_data)
        v_mean, _ = self.compute_values(z)

        if (
            "action" not in hetero_data.node_types
            or hetero_data["action"].x is None
            or hetero_data["action"].x.size(0) == 0
        ):
            return {"selected_entity_id": -1, "selected_path_index": -1}, v_mean, torch.empty((0,)), 0.0

        e_actions = self.actor_pointer.compute_action_embeddings(x_dict, hetero_data)
        batch_idx = (
            hetero_data["action"].batch
            if hasattr(hetero_data["action"], "batch") and getattr(hetero_data["action"], "batch") is not None
            else None
        )

        if deterministic:
            chosen_seq_indices, mu_log_prob = self._beam_search_agnostic(
                z, e_actions, batch_idx, hetero_data, x_dict
            )
        else:
            chosen_seq_indices, mu_log_prob = self._sample_agnostic_trajectory(
                z, e_actions, batch_idx, hetero_data, x_dict, deterministic=False
            )

        response_dict = self._decode_trajectory(hetero_data, chosen_seq_indices)
        probs = torch.empty((0,), device=z.device)
        return response_dict, v_mean, probs, mu_log_prob
