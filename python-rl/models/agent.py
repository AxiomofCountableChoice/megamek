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
    def __init__(self, hidden_dim=128, ensemble_size=8):
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
        self.actor_pointer = ActionConditionedPointer(hidden_dim=hidden_dim, action_feature_dim=3)
        
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
        Forward pass of the agent for inference/sampling.
        """
        # 1. Forward Encoder
        z, x_dict = self.encoder(hetero_data)
        
        # 2. Extract Value Ensembles
        v_mean, v_var = self.compute_values(z)
        
        # 3. Autoregressive Sampling (Pointer logic)
        # Assuming we are querying k=0 (the root move hex selection) for sampling here
        # (For structured multi-step sequence decoding, we'd iteratively sample logic in a loop)
        
        # Compute embeddings for valid action nodes
        e_actions = self.actor_pointer.compute_action_embeddings(x_dict, hetero_data)
        
        # Decode the initial pointer context
        s_k = self.actor_pointer.decode_sequence(z, None) 
        
        # Batch indexing
        batch_idx = hetero_data['action'].batch if hasattr(hetero_data['action'], 'batch') and getattr(hetero_data['action'], 'batch') is not None else None
        
        # Score the action logits
        logits = self.actor_pointer.score_actions(s_k, e_actions, batch_idx)
        
        # Action distribution
        if logits.size(0) > 0:
            probs = torch.softmax(logits, dim=-1)
            action_idx = torch.multinomial(probs, num_samples=1).item()
        else:
            action_idx = -1
            probs = torch.empty((0,))
            
        # Returns action format matching Java expectations
        return {"selected_path_index": action_idx}, v_mean, probs
