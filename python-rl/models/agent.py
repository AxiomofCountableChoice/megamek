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

    def get_action(self, hetero_data, mask_tree):
        """
        Forward pass of the agent for inference/sampling using Greedy Decoding.
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
            return {"selected_path_index": -1}, v_mean, torch.empty((0,))
            
        phase_type = int(hetero_data['action'].x[0, 5].item())
        
        if phase_type == 0 or phase_type == 3: # MOVEMENT or DEPLOYMENT (assuming similar flat action space)
            s_0 = self.actor_pointer.decode_sequence(z, None) 
            logits = self.actor_pointer.score_actions(s_0, e_actions, batch_idx)
            
            if logits.size(0) > 0:
                probs = torch.softmax(logits, dim=-1)
                action_idx = torch.argmax(probs).item()
            else:
                action_idx = -1
                probs = torch.empty((0,))
                
            return {"selected_path_index": action_idx}, v_mean, probs
            
        elif phase_type == 1:
            # WEAPON_INFERENCE: Autoregressive Greedy Decoding
            response = {"twist": 0, "attacks": []}
            if not mask_tree or "valid_twists" not in mask_tree:
                return response, v_mean, torch.empty((0,))
                
            history_embeddings = []
            
            # Step 0: Torso Twist
            s_0 = self.actor_pointer.decode_sequence(z, None)
            logits_0 = self.actor_pointer.score_actions(s_0, e_actions, batch_idx)
            
            valid_twists = mask_tree.get("valid_twists", [])
            best_twist, best_twist_logit = None, -float('inf')
            
            for tm in valid_twists:
                n_idx = tm.get("node_idx", -1)
                if n_idx != -1 and logits_0[n_idx].item() > best_twist_logit:
                    best_twist_logit = logits_0[n_idx].item()
                    best_twist = tm
                    
            if not best_twist:
                return response, v_mean, torch.empty((0,))
                
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
                            
                best_opt, best_opt_logit = None, -float('inf')
                for opt in valid_options:
                    n_idx = opt["node_idx"]
                    if logits_k[n_idx].item() > best_opt_logit:
                        best_opt_logit = logits_k[n_idx].item()
                        best_opt = opt
                        
                if not best_opt or best_opt["type"] == "END":
                    break
                    
                chosen_target = best_opt["data"]
                target_entity = chosen_target.get("target_entity_index", -1)
                history_embeddings.append(e_actions[best_opt["node_idx"]].unsqueeze(0))
                
                s_k1 = self.actor_pointer.decode_sequence(z, torch.stack(history_embeddings, dim=1))
                logits_k1 = self.actor_pointer.score_actions(s_k1, e_actions, batch_idx)
                
                best_w_opt, best_w_logit = None, -float('inf')
                for wm in best_opt["unfired"]:
                    n_idx = wm.get("node_idx", -1)
                    if n_idx != -1 and logits_k1[n_idx].item() > best_w_logit:
                        best_w_logit = logits_k1[n_idx].item()
                        best_w_opt = wm
                        
                if not best_w_opt:
                    break
                    
                w_id = best_w_opt.get("weapon_id", -1)
                fired_weapons.add(w_id)
                history_embeddings.append(e_actions[best_w_opt["node_idx"]].unsqueeze(0))
                
                response["attacks"].append({"target_id": target_entity, "weapon_id": w_id})
                
            return response, v_mean, torch.empty((0,))
            
        elif phase_type == 2:
            # PHYSICAL_INFERENCE
            response = {"attack": None}
            if not mask_tree or "valid_targets" not in mask_tree:
                return response, v_mean, torch.empty((0,))
                
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
                    
            best_opt, best_opt_logit = None, -float('inf')
            for opt in valid_options:
                n_idx = opt["node_idx"]
                if logits_0[n_idx].item() > best_opt_logit:
                    best_opt_logit = logits_0[n_idx].item()
                    best_opt = opt
                    
            if not best_opt or best_opt["type"] == "END":
                return response, v_mean, torch.empty((0,))
                
            chosen_target = best_opt["data"]
            target_entity = chosen_target.get("target_entity_index", -1)
            history_embeddings.append(e_actions[best_opt["node_idx"]].unsqueeze(0))
            
            s_1 = self.actor_pointer.decode_sequence(z, torch.stack(history_embeddings, dim=1))
            logits_1 = self.actor_pointer.score_actions(s_1, e_actions, batch_idx)
            
            best_att, best_att_logit = None, -float('inf')
            for am in chosen_target.get("valid_attacks", []):
                n_idx = am.get("node_idx", -1)
                if n_idx != -1 and logits_1[n_idx].item() > best_att_logit:
                    best_att_logit = logits_1[n_idx].item()
                    best_att = am
                    
            if best_att:
                response["attack"] = {"target_id": target_entity, "action_type": best_att.get("action_type", -1)}
                
            return response, v_mean, torch.empty((0,))
            
        else:
            return {"selected_path_index": -1}, v_mean, torch.empty((0,))
