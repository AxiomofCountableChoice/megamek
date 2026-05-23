import torch
import torch.nn as nn
from models.encoder import MegaMekHGTEncoder
from models.actor import ActionConditionedPointer

class MegaMekAgent(nn.Module):
    """
    Main IMPALA Actor-Critic Module defined in ARCHITECTURE.md (Sections 3c & 3d).
    Contains:
      - The frozen cache HGT Encoder.
      - The Value Ensemble (E=8 by default for Epistemic Exploration).
      - The Autoregressive Pointer Decoder (Action head).
    """
    def __init__(self, hidden_dim=128, ensemble_size=8, action_feature_dim=8):
        super().__init__()
        self.encoder = MegaMekHGTEncoder(hidden_dim=hidden_dim)
        
        # 1. Epistemic Value Ensemble (Critic)
        # Branching from z (dim = hidden_dim * 2 since z_graph + z_context)
        latent_dim = hidden_dim * 2
        
        self.ensembles = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(ensemble_size)
        ])
        
        # 2. Autoregressive Transformer Decoder (Teacher Forcing & Action Scoring)
        self.actor_pointer = ActionConditionedPointer(hidden_dim=hidden_dim, action_feature_dim=action_feature_dim)
        
    def compute_values(self, z):
        """
        Computes the V-Trace Value Ensembles for Epistemic Bonus Target (Section 2c)
        """
        # Forward pass all critics independently
        v_preds = torch.stack([critic(z) for critic in self.ensembles], dim=-1)
        # v_preds shape: [batch, 1, E]
        
        v_mean = v_preds.mean(dim=-1)
        v_var = v_preds.var(dim=-1, unbiased=False)
        return v_mean, v_var

    def evaluate_actions(self, batch):
        """
        Evaluates an existing HeteroData batch (must contain y_sequence target actions).
        Returns pi_log_prob_total, entropy_total, v_mean, v_variance
        """
        z, x_dict = self.encoder(batch)
        
        # Critic values
        v_preds = torch.stack([critic(z) for critic in self.ensembles], dim=-1) # [batch_size, 1, E]
        v_mean = v_preds.mean(dim=-1).squeeze(1) # [batch_size]
        v_variance = v_preds.var(dim=-1, unbiased=False).squeeze(1) # [batch_size]
        
        pi_log_prob_total = torch.zeros(z.size(0), device=z.device)
        entropy_total = torch.zeros(z.size(0), device=z.device)
        
        y_seq = getattr(batch, 'y_sequence', None)
        if y_seq is None or y_seq.size(0) == 0:
            return pi_log_prob_total, entropy_total, v_mean, v_variance
            
        e_actions = self.actor_pointer.compute_action_embeddings(x_dict, batch)
        if e_actions.size(0) == 0:
            return pi_log_prob_total, entropy_total, v_mean, v_variance
            
        step_indices = batch['action'].step_idx
        seq_len = y_seq.size(0)
        chosen_embeddings = []
        
        for k in range(seq_len):
            target_idx = y_seq[k].item()
            
            if getattr(batch, 'context', [""])[0] == "MOVEMENT_BC":
                valid_mask = (step_indices == k)
            else:
                target_type = batch['action'].x[target_idx, 6].item()
                valid_mask = (batch['action'].x[:, 6] == target_type)
                
            if not valid_mask.any():
                break
                
            if not valid_mask[target_idx]:
                break
                
            if len(chosen_embeddings) == 0:
                history_tensor = None
            else:
                history_tensor = torch.stack(chosen_embeddings).unsqueeze(0)
                
            s_k = self.actor_pointer.decode_sequence(z, history_tensor)
            
            e_tier = e_actions[valid_mask]
            tier_batch_idx = batch['action'].batch[valid_mask] if hasattr(batch['action'], 'batch') and getattr(batch['action'], 'batch') is not None else None
            
            logits = self.actor_pointer.score_actions(s_k, e_tier, tier_batch_idx) # (num_tier_actions,)
            probs = torch.softmax(logits, dim=-1)
            
            global_indices = torch.where(valid_mask)[0]
            relative_target = (global_indices == target_idx).nonzero(as_tuple=True)[0]
            
            if relative_target.numel() == 0:
                break
                
            # Log prob
            pi_prob = probs[relative_target]
            pi_log_prob_total += torch.log(pi_prob + 1e-10).sum()
            
            # Entropy
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            entropy_total += entropy
            
            chosen_embeddings.append(e_actions[target_idx])
            
        return pi_log_prob_total, entropy_total, v_mean, v_variance

    def _decode_and_sample_step(self, z, history_embeddings, e_actions, candidates, batch_idx, deterministic=False):
        """
        Decodes the sequence using the history of chosen action embeddings,
        scores candidates (either a boolean mask or a list of option dicts),
        and samples the selected action.
        """
        # 1. Decode sequence based on history
        if not history_embeddings:
            history_tensor = None
        else:
            history_tensor = torch.stack(history_embeddings, dim=1) # [1, len, hidden_dim]
            
        s_k = self.actor_pointer.decode_sequence(z, history_tensor)
        
        # 2. Score candidates
        if isinstance(candidates, torch.Tensor):
            # Mask based scoring
            if not candidates.any():
                return None, 0.0, torch.empty((0,))
            e_candidates = e_actions[candidates]
            cand_batch_idx = batch_idx[candidates] if batch_idx is not None else None
            logits = self.actor_pointer.score_actions(s_k, e_candidates, cand_batch_idx)
            probs = torch.softmax(logits, dim=-1)
            
            if deterministic:
                rel_idx = torch.argmax(probs).item()
                log_prob = torch.log(probs[rel_idx] + 1e-8).item()
            else:
                dist = torch.distributions.Categorical(probs)
                rel_idx = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(rel_idx, device=probs.device)).item()
                
            global_indices = torch.where(candidates)[0]
            selected_global_idx = global_indices[rel_idx].item()
            return selected_global_idx, log_prob, probs
        else:
            # Option list based scoring
            if not candidates:
                return None, 0.0, torch.empty((0,))
                
            indices = [opt.get("node_idx", -1) for opt in candidates]
            e_candidates = e_actions[indices]
            cand_batch_idx = batch_idx[indices] if batch_idx is not None else None
            logits = self.actor_pointer.score_actions(s_k, e_candidates, cand_batch_idx)
            probs = torch.softmax(logits, dim=-1)
            
            if deterministic:
                idx = torch.argmax(probs).item()
                log_prob = torch.log(probs[idx] + 1e-8).item()
            else:
                dist = torch.distributions.Categorical(probs)
                idx = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(idx, device=probs.device)).item()
                
            return candidates[idx], log_prob, probs

    def _get_root_choices(self, phase_type, mask_tree, hetero_data):
        """
        Parses the phase's mask tree into initial DecisionNode choices.
        """
        if phase_type == 0 or phase_type == 3: # MOVEMENT or DEPLOYMENT
            active_entity_idx = mask_tree.get("active_entity_index", -1) if mask_tree else -1
            step_indices = hetero_data['action'].step_idx
            source_unit_indices = hetero_data['action'].source_unit_idx
            target_hex_indices = hetero_data['action'].target_hex_idx
            path_indices = hetero_data['action'].path_idx
            
            # Find Step 0 hex nodes
            step0_mask = (step_indices == 0) & (source_unit_indices == active_entity_idx)
            if not step0_mask.any():
                return []
                
            global_indices = torch.where(step0_mask)[0].tolist()
            choices = []
            for node_idx in global_indices:
                selected_hex = target_hex_indices[node_idx].item()
                # Children are Step 1 path nodes for this hex
                step1_mask = (step_indices == 1) & (source_unit_indices == active_entity_idx) & (target_hex_indices == selected_hex)
                child_indices = torch.where(step1_mask)[0].tolist()
                
                children = []
                for c_node_idx in child_indices:
                    raw_path_idx = path_indices[c_node_idx].item()
                    children.append({
                        "node_idx": c_node_idx,
                        "type": "PATH",
                        "data": {"path_index": raw_path_idx},
                        "children": []
                    })
                
                choices.append({
                    "node_idx": node_idx,
                    "type": "HEX",
                    "data": {"dest_index": selected_hex},
                    "children": children
                })
            return choices
            
        elif phase_type == 1: # WEAPON
            valid_twists = mask_tree.get("valid_twists", [])
            choices = []
            for twist in valid_twists:
                choices.append({
                    "node_idx": twist.get("node_idx", -1),
                    "type": "TWIST",
                    "data": twist,
                    "children": []
                })
            return choices
            
        elif phase_type == 2: # PHYSICAL
            valid_targets = mask_tree.get("valid_targets", [])
            
            active_entity_idx = -1
            if valid_targets:
                active_entity_idx = valid_targets[0].get("source_entity_index", -1)
                
            # Find END node index dynamically
            end_node_idx = -1
            if active_entity_idx != -1:
                action_x = hetero_data['action'].x
                source_unit_indices = hetero_data['action'].source_unit_idx
                end_mask = (action_x[:, 6] == 3.0) & (source_unit_indices == active_entity_idx)
                if end_mask.any():
                    end_node_idx = torch.where(end_mask)[0][0].item()
                    
            choices = []
            if end_node_idx != -1:
                choices.append({
                    "node_idx": end_node_idx,
                    "type": "END",
                    "data": None,
                    "children": []
                })
            for target in valid_targets:
                children = []
                for att in target.get("valid_attacks", []):
                    children.append({
                        "node_idx": att.get("node_idx", -1),
                        "type": "ATTACK",
                        "data": att,
                        "children": []
                    })
                choices.append({
                    "node_idx": target.get("node_idx", -1),
                    "type": "TARGET",
                    "data": target,
                    "children": children
                })
            return choices
            
        return []

    def _get_dynamic_children(self, node, fired_weapons, mask_tree, hetero_data):
        """
        Computes dynamic, state-dependent children (specifically for WEAPON phase).
        """
        node_type = node["type"]
        
        twist_data = node["data"] if node_type == "TWIST" else mask_tree.get("_active_twist_data")
        active_entity_idx = twist_data.get("source_entity_index", -1) if twist_data else -1
        
        # Find END node index dynamically
        end_node_idx = -1
        if active_entity_idx != -1:
            action_x = hetero_data['action'].x
            source_unit_indices = hetero_data['action'].source_unit_idx
            end_mask = (action_x[:, 6] == 3.0) & (source_unit_indices == active_entity_idx)
            if end_mask.any():
                end_node_idx = torch.where(end_mask)[0][0].item()
        
        if node_type == "TWIST" or node_type == "WEAPON":
            choices = []
            if end_node_idx != -1:
                choices.append({
                    "node_idx": end_node_idx,
                    "type": "END",
                    "data": None,
                    "children": []
                })
                
            if twist_data:
                for tm in twist_data.get("valid_targets", []):
                    unfired = [wm for wm in tm.get("valid_weapons", []) if wm.get("weapon_id", -1) not in fired_weapons]
                    if unfired:
                        choices.append({
                            "node_idx": tm.get("node_idx", -1),
                            "type": "TARGET",
                            "data": tm,
                            "unfired_weapons": unfired,
                            "children": []
                        })
            return choices
            
        elif node_type == "TARGET":
            choices = []
            unfired = node.get("unfired_weapons", [])
            for wm in unfired:
                choices.append({
                    "node_idx": wm.get("node_idx", -1),
                    "type": "WEAPON",
                    "data": wm,
                    "children": []
                })
            return choices
            
        return []

    def _sample_agnostic_trajectory(self, z, e_actions, batch_idx, phase_type, mask_tree, hetero_data, deterministic=False):
        """
        Agnostically traverses the decision tree, scores and samples decision nodes,
        and aggregates the selected nodes into a trajectory.
        """
        trajectory = []
        history_embeddings = []
        total_log_prob = 0.0
        fired_weapons = set()
        
        # 1. Retrieve initial choices
        current_choices = self._get_root_choices(phase_type, mask_tree, hetero_data)
        if not current_choices:
            return [], 0.0
            
        while current_choices:
            # Score and sample step agnostically
            selected_node, log_prob, _ = self._decode_and_sample_step(
                z, history_embeddings, e_actions, current_choices, batch_idx, deterministic
            )
            if not selected_node:
                break
                
            trajectory.append(selected_node)
            total_log_prob += log_prob
            history_embeddings.append(e_actions[selected_node["node_idx"]].unsqueeze(0))
            
            # Dynamic state tracking
            if selected_node["type"] == "TWIST":
                mask_tree["_active_twist_data"] = selected_node["data"]
            elif selected_node["type"] == "WEAPON":
                w_id = selected_node["data"].get("weapon_id", -1)
                fired_weapons.add(w_id)
            elif selected_node["type"] == "END":
                break
                
            # 2. Retrieve children for next step
            if phase_type == 1:
                current_choices = self._get_dynamic_children(selected_node, fired_weapons, mask_tree, hetero_data)
            else:
                current_choices = selected_node.get("children", [])
                
        mask_tree.pop("_active_twist_data", None)
        return trajectory, total_log_prob

    def _beam_search_agnostic(self, z, e_actions, batch_idx, phase_type, mask_tree, hetero_data, beam_width=3):
        """
        Agnostic step-by-step beam search tree search over the generic DecisionNode tree.
        """
        root_choices = self._get_root_choices(phase_type, mask_tree, hetero_data)
        if not root_choices:
            return [], 0.0
            
        s_0 = self.actor_pointer.decode_sequence(z, None)
        root_indices = [c["node_idx"] for c in root_choices]
        e_cands = e_actions[root_indices]
        cand_batch_idx = batch_idx[root_indices] if batch_idx is not None else None
        logits_0 = self.actor_pointer.score_actions(s_0, e_cands, cand_batch_idx)
        log_probs_0 = torch.log_softmax(logits_0, dim=-1)
        
        beams = []
        for i, choice in enumerate(root_choices):
            lp = log_probs_0[i].item()
            beams.append({
                "trajectory": [choice],
                "history": [e_actions[choice["node_idx"]].unsqueeze(0)],
                "log_prob_sum": lp,
                "fired_weapons": set(),
                "done": False
            })
            
        beams = sorted(beams, key=lambda x: x["log_prob_sum"], reverse=True)[:beam_width]
        
        while True:
            if all(b["done"] for b in beams):
                break
            new_beams = []
            for b in beams:
                if b["done"]:
                    new_beams.append(b)
                    continue
                    
                last_node = b["trajectory"][-1]
                
                # Check for Twist active state
                if last_node["type"] == "TWIST":
                    mask_tree["_active_twist_data"] = last_node["data"]
                elif last_node["type"] == "END":
                    b["done"] = True
                    new_beams.append(b)
                    continue
                    
                # Get children
                if phase_type == 1:
                    children = self._get_dynamic_children(last_node, b["fired_weapons"], mask_tree, hetero_data)
                else:
                    children = last_node.get("children", [])
                    
                if not children:
                    b["done"] = True
                    new_beams.append(b)
                    continue
                    
                # Score children
                s_k = self.actor_pointer.decode_sequence(z, torch.stack(b["history"], dim=1))
                child_indices = [c["node_idx"] for c in children]
                e_cands = e_actions[child_indices]
                cand_batch_idx = batch_idx[child_indices] if batch_idx is not None else None
                logits_k = self.actor_pointer.score_actions(s_k, e_cands, cand_batch_idx)
                log_probs_k = torch.log_softmax(logits_k, dim=-1)
                
                for i, child in enumerate(children):
                    lp = log_probs_k[i].item()
                    new_fired = set(b["fired_weapons"])
                    if child["type"] == "WEAPON":
                        w_id = child["data"].get("weapon_id", -1)
                        new_fired.add(w_id)
                        
                    new_b = {
                        "trajectory": b["trajectory"] + [child],
                        "history": b["history"] + [e_actions[child["node_idx"]].unsqueeze(0)],
                        "log_prob_sum": b["log_prob_sum"] + lp,
                        "fired_weapons": new_fired,
                        "done": False
                    }
                    if child["type"] == "END":
                        new_b["done"] = True
                    new_beams.append(new_b)
                    
            beams = sorted(new_beams, key=lambda x: x["log_prob_sum"], reverse=True)[:beam_width]
            
        mask_tree.pop("_active_twist_data", None)
        best_beam = beams[0] if beams else None
        trajectory = best_beam["trajectory"] if best_beam else []
        log_prob = best_beam["log_prob_sum"] if best_beam else 0.0
        return trajectory, log_prob

    def _decode_movement_action(self, trajectory):
        """
        Decodes selected MOVEMENT trajectory [HEX, PATH] into megamek action response.
        """
        if len(trajectory) < 2:
            return {"selected_path_index": -1}
        path_node = trajectory[1]
        path_idx = path_node["data"].get("path_index", -1)
        return {"selected_path_index": path_idx}

    def _decode_weapon_action(self, trajectory):
        """
        Decodes selected WEAPON trajectory [TWIST, TARGET, WEAPON, TARGET, WEAPON, ..., END] into megamek action response.
        """
        response = {"twist": 0, "attacks": []}
        if not trajectory:
            return response
            
        twist_node = trajectory[0]
        response["twist"] = twist_node["data"].get("twist", 0)
        response["selected_entity_id"] = twist_node["data"].get("source_entity_index", -1)
        
        # Parse attacks
        current_target_id = -1
        for node in trajectory[1:]:
            if node["type"] == "TARGET":
                current_target_id = node["data"].get("target_entity_index", -1)
            elif node["type"] == "WEAPON":
                w_id = node["data"].get("weapon_id", -1)
                response["attacks"].append({
                    "target_id": current_target_id,
                    "weapon_id": w_id
                })
        return response

    def _decode_physical_action(self, trajectory):
        """
        Decodes selected PHYSICAL trajectory [TARGET, ATTACK] into megamek action response.
        """
        response = {"attack": None}
        if len(trajectory) < 2:
            return response
            
        target_node = trajectory[0]
        attack_node = trajectory[1]
        
        target_entity = target_node["data"].get("target_entity_index", -1)
        action_type = attack_node["data"].get("action_type", -1)
        
        response["selected_entity_id"] = target_node["data"].get("source_entity_index", -1)
        response["attack"] = {
            "target_id": target_entity,
            "action_type": action_type
        }
        return response

    def get_action(self, hetero_data, mask_tree, deterministic=False):
        """
        Forward pass of the agent for inference/sampling.
        Supports deterministic (greedy) and stochastic (categorical) action selection.
        Returns: response_dict, v_mean, probs, mu_log_prob
        """
        # 1. Forward Encoder
        z, x_dict = self.encoder(hetero_data)
        
        # 2. Extract Value Ensembles
        v_mean, v_var = self.compute_values(z)
        
        # 3. Compute embeddings for all action nodes
        e_actions = self.actor_pointer.compute_action_embeddings(x_dict, hetero_data)
        batch_idx = hetero_data['action'].batch if hasattr(hetero_data['action'], 'batch') and getattr(hetero_data['action'], 'batch') is not None else None
        
        # Determine Phase Type from first action feature
        if hetero_data['action'].x is None or hetero_data['action'].x.size(0) == 0:
            return {"selected_path_index": -1}, v_mean, torch.empty((0,)), 0.0
            
        phase_type = int(hetero_data['action'].x[0, 7].item())
        
        # 4. Agnostically sample the trajectory (stochastically or via beam search)
        if deterministic:
            trajectory, mu_log_prob = self._beam_search_agnostic(
                z, e_actions, batch_idx, phase_type, mask_tree, hetero_data
            )
        else:
            trajectory, mu_log_prob = self._sample_agnostic_trajectory(
                z, e_actions, batch_idx, phase_type, mask_tree, hetero_data, deterministic=False
            )
            
        # 5. Decode trajectory back to megamek format using opinionated decoders
        if phase_type == 0 or phase_type == 3: # MOVEMENT or DEPLOYMENT
            response_dict = self._decode_movement_action(trajectory)
            probs = torch.empty((0,))
        elif phase_type == 1:
            response_dict = self._decode_weapon_action(trajectory)
            probs = torch.empty((0,))
        elif phase_type == 2:
            response_dict = self._decode_physical_action(trajectory)
            probs = torch.empty((0,))
        else:
            response_dict = {"selected_path_index": -1}
            probs = torch.empty((0,))
            
        return response_dict, v_mean, probs, mu_log_prob
