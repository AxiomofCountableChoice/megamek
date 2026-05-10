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
    def __init__(self, hidden_dim=128, ensemble_size=8, action_feature_dim=6):
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
                target_type = batch['action'].x[target_idx, 4].item()
                valid_mask = (batch['action'].x[:, 4] == target_type)
                
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

    def get_action(self, hetero_data, mask_tree, deterministic=False):
        """
        Forward pass of the agent for inference/sampling.
        Supports deterministic (greedy) and stochastic (categorical) action selection.
        Returns: response_dict, v_mean, probs, mu_log_prob
        """
        def _sample_step(logits, valid_options, det):
            if not valid_options:
                return None, 0.0
                
            indices = [opt.get("node_idx", -1) for opt in valid_options]
            valid_logits = logits[indices]
            probs = torch.softmax(valid_logits, dim=-1)
            
            if det:
                idx = torch.argmax(probs).item()
                log_prob = torch.log(probs[idx] + 1e-8).item()
            else:
                dist = torch.distributions.Categorical(probs)
                idx = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(idx)).item()
                
            return valid_options[idx], log_prob

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
            
        phase_type = int(hetero_data['action'].x[0, 5].item())
        
        if phase_type == 0 or phase_type == 3: # MOVEMENT or DEPLOYMENT (assuming similar flat action space)
            s_0 = self.actor_pointer.decode_sequence(z, None) 
            logits = self.actor_pointer.score_actions(s_0, e_actions, batch_idx)
            
            if logits.size(0) > 0:
                probs = torch.softmax(logits, dim=-1)
                if deterministic:
                    action_idx = torch.argmax(probs).item()
                    mu_log_prob = torch.log(probs[action_idx] + 1e-8).item()
                else:
                    dist = torch.distributions.Categorical(probs)
                    action_idx = dist.sample().item()
                    mu_log_prob = dist.log_prob(torch.tensor(action_idx)).item()
            else:
                action_idx = -1
                probs = torch.empty((0,))
                mu_log_prob = 0.0
                
            return {"selected_path_index": action_idx}, v_mean, probs, mu_log_prob
            
        elif phase_type == 1:
            # WEAPON_INFERENCE: Autoregressive Decoding
            response = {"twist": 0, "attacks": []}
            mu_log_prob_total = 0.0
            
            if not mask_tree or "valid_twists" not in mask_tree:
                return response, v_mean, torch.empty((0,)), 0.0
                
            history_embeddings = []
            
            # Step 0: Torso Twist
            s_0 = self.actor_pointer.decode_sequence(z, None)
            logits_0 = self.actor_pointer.score_actions(s_0, e_actions, batch_idx)
            
            valid_twists = mask_tree.get("valid_twists", [])
            
            if deterministic:
                # Beam Search for Evaluation
                beam_width = 3
                beams = []
                twist_indices = [opt.get("node_idx", -1) for opt in valid_twists]
                twist_logits = logits_0[twist_indices]
                twist_log_probs = torch.log_softmax(twist_logits, dim=-1)
                
                for i, opt_tm in enumerate(valid_twists):
                    lp = twist_log_probs[i].item()
                    beams.append({
                        "twist": opt_tm,
                        "attacks": [],
                        "history": [e_actions[opt_tm.get("node_idx")].unsqueeze(0)],
                        "fired_weapons": set(),
                        "log_prob_sum": lp,
                        "done": False
                    })
                    
                beams = sorted(beams, key=lambda x: x["log_prob_sum"], reverse=True)[:beam_width]
                end_node_idx = mask_tree.get("end_node_idx", -1)
                
                while True:
                    if all(b["done"] for b in beams): break
                    new_beams = []
                    for b in beams:
                        if b["done"]:
                            new_beams.append(b)
                            continue
                            
                        s_k = self.actor_pointer.decode_sequence(z, torch.stack(b["history"], dim=1))
                        logits_k = self.actor_pointer.score_actions(s_k, e_actions, batch_idx)
                        
                        valid_options = []
                        if end_node_idx != -1:
                            valid_options.append({"type": "END", "node_idx": end_node_idx, "data": None})
                            
                        for tm in b["twist"].get("valid_targets", []):
                            unfired = [wm for wm in tm.get("valid_weapons", []) if wm.get("weapon_id", -1) not in b["fired_weapons"]]
                            if unfired:
                                valid_options.append({"type": "TARGET", "node_idx": tm.get("node_idx", -1), "data": tm, "unfired": unfired})
                                
                        if not valid_options:
                            b["done"] = True
                            new_beams.append(b)
                            continue
                            
                        opt_indices = [opt["node_idx"] for opt in valid_options]
                        opt_log_probs = torch.log_softmax(logits_k[opt_indices], dim=-1)
                        
                        for i, opt in enumerate(valid_options):
                            lp = opt_log_probs[i].item()
                            if opt["type"] == "END":
                                new_b = dict(b)
                                new_b["history"] = list(b["history"])
                                new_b["attacks"] = list(b["attacks"])
                                new_b["fired_weapons"] = set(b["fired_weapons"])
                                new_b["log_prob_sum"] = b["log_prob_sum"] + lp
                                new_b["done"] = True
                                new_beams.append(new_b)
                            else:
                                target_entity = opt["data"].get("target_entity_index", -1)
                                temp_history = list(b["history"]) + [e_actions[opt["node_idx"]].unsqueeze(0)]
                                s_k1 = self.actor_pointer.decode_sequence(z, torch.stack(temp_history, dim=1))
                                logits_k1 = self.actor_pointer.score_actions(s_k1, e_actions, batch_idx)
                                
                                w_opts = opt["unfired"]
                                w_indices = [w.get("node_idx", -1) for w in w_opts]
                                w_log_probs = torch.log_softmax(logits_k1[w_indices], dim=-1)
                                
                                for j, w_opt in enumerate(w_opts):
                                    w_lp = w_log_probs[j].item()
                                    w_id = w_opt.get("weapon_id", -1)
                                    new_b = dict(b)
                                    new_b["history"] = temp_history + [e_actions[w_opt["node_idx"]].unsqueeze(0)]
                                    new_b["attacks"] = list(b["attacks"]) + [{"target_id": target_entity, "weapon_id": w_id}]
                                    new_b["fired_weapons"] = set(b["fired_weapons"]) | {w_id}
                                    new_b["log_prob_sum"] = b["log_prob_sum"] + lp + w_lp
                                    new_b["done"] = False
                                    new_beams.append(new_b)
                    beams = sorted(new_beams, key=lambda x: x["log_prob_sum"], reverse=True)[:beam_width]
                    
                best_beam = beams[0] if beams else None
                if best_beam:
                    response["twist"] = best_beam["twist"].get("twist", 0)
                    response["attacks"] = best_beam["attacks"]
                    mu_log_prob_total = best_beam["log_prob_sum"]
            else:
                # Stochastic Sampling
                best_twist, twist_log_prob = _sample_step(logits_0, valid_twists, deterministic)
                
                if not best_twist:
                    return response, v_mean, torch.empty((0,)), 0.0
                    
                mu_log_prob_total += twist_log_prob
                response["twist"] = best_twist.get("twist", 0)
                chosen_twist_node = best_twist.get("node_idx")
                history_embeddings.append(e_actions[chosen_twist_node].unsqueeze(0))
                
                end_node_idx = mask_tree.get("end_node_idx", -1)
                fired_weapons = set()
                
                while True:
                    history_tensor = torch.stack(history_embeddings, dim=1)
                    s_k = self.actor_pointer.decode_sequence(z, history_tensor)
                    logits_k = self.actor_pointer.score_actions(s_k, e_actions, batch_idx)
                    
                    valid_options = []
                    if end_node_idx != -1:
                        valid_options.append({"type": "END", "node_idx": end_node_idx, "data": None})
                        
                    valid_targets = best_twist.get("valid_targets", [])
                    for tm in valid_targets:
                        unfired_weapons = [wm for wm in tm.get("valid_weapons", []) if wm.get("weapon_id", -1) not in fired_weapons]
                        if unfired_weapons:
                            n_idx = tm.get("node_idx", -1)
                            if n_idx != -1:
                                valid_options.append({"type": "TARGET", "node_idx": n_idx, "data": tm, "unfired": unfired_weapons})
                                
                    best_opt, opt_log_prob = _sample_step(logits_k, valid_options, deterministic)
                            
                    if not best_opt or best_opt["type"] == "END":
                        if best_opt:
                            mu_log_prob_total += opt_log_prob
                        break
                        
                    mu_log_prob_total += opt_log_prob
                    chosen_target = best_opt["data"]
                    target_entity = chosen_target.get("target_entity_index", -1)
                    history_embeddings.append(e_actions[best_opt["node_idx"]].unsqueeze(0))
                    
                    s_k1 = self.actor_pointer.decode_sequence(z, torch.stack(history_embeddings, dim=1))
                    logits_k1 = self.actor_pointer.score_actions(s_k1, e_actions, batch_idx)
                    
                    best_w_opt, w_log_prob = _sample_step(logits_k1, best_opt["unfired"], deterministic)
                            
                    if not best_w_opt:
                        break
                        
                    mu_log_prob_total += w_log_prob
                    w_id = best_w_opt.get("weapon_id", -1)
                    fired_weapons.add(w_id)
                    history_embeddings.append(e_actions[best_w_opt["node_idx"]].unsqueeze(0))
                    
                    response["attacks"].append({"target_id": target_entity, "weapon_id": w_id})
                    
            return response, v_mean, torch.empty((0,)), mu_log_prob_total
            
        elif phase_type == 2:
            # PHYSICAL_INFERENCE
            response = {"attack": None}
            mu_log_prob_total = 0.0
            
            if not mask_tree or "valid_targets" not in mask_tree:
                return response, v_mean, torch.empty((0,)), 0.0
                
            history_embeddings = []
            s_0 = self.actor_pointer.decode_sequence(z, None)
            logits_0 = self.actor_pointer.score_actions(s_0, e_actions, batch_idx)
            
            end_node_idx = mask_tree.get("end_node_idx", -1)
            valid_targets = mask_tree.get("valid_targets", [])
            
            valid_options = []
            if end_node_idx != -1:
                valid_options.append({"type": "END", "node_idx": end_node_idx, "data": None})
            for tm in valid_targets:
                n_idx = tm.get("node_idx", -1)
                if n_idx != -1:
                    valid_options.append({"type": "TARGET", "node_idx": n_idx, "data": tm})
                    
            best_opt, opt_log_prob = _sample_step(logits_0, valid_options, deterministic)
                    
            if not best_opt or best_opt["type"] == "END":
                if best_opt:
                    mu_log_prob_total += opt_log_prob
                return response, v_mean, torch.empty((0,)), mu_log_prob_total
                
            mu_log_prob_total += opt_log_prob
            chosen_target = best_opt["data"]
            target_entity = chosen_target.get("target_entity_index", -1)
            history_embeddings.append(e_actions[best_opt["node_idx"]].unsqueeze(0))
            
            s_1 = self.actor_pointer.decode_sequence(z, torch.stack(history_embeddings, dim=1))
            logits_1 = self.actor_pointer.score_actions(s_1, e_actions, batch_idx)
            
            best_att, att_log_prob = _sample_step(logits_1, chosen_target.get("valid_attacks", []), deterministic)
                    
            if best_att:
                mu_log_prob_total += att_log_prob
                response["attack"] = {"target_id": target_entity, "action_type": best_att.get("action_type", -1)}
                
            return response, v_mean, torch.empty((0,)), mu_log_prob_total
            
        else:
            return {"selected_path_index": -1}, v_mean, torch.empty((0,)), 0.0
