import torch
import torch.nn as nn
from models.encoder import MegaMekHANEncoder

class MegaMekAgent(nn.Module):
    """
    Main IMPALA Actor-Critic Module defined in ARCHITECTURE.md (Sections 3c & 3d).
    Contains:
      - The frozen cache HAN Encoder.
      - The Value Ensemble (E=3 by default for Epistemic Exploration).
      - The Autoregressive Pointer Decoder (Action head).
    """
    def __init__(self, hidden_dim=128, ensemble_size=3):
        super().__init__()
        self.encoder = MegaMekHANEncoder(hidden_dim=hidden_dim)
        
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
        
        # 2. Pointer Embedder (Action MLP)
        # Represents non-spatial sub-actions (e.g. Weapon Selections) dynamically
        self.action_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim), # e.g. Base Dmg, Heat, Range, Is_Cluster, Size
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Autoregressive Transformer Decoder
        # A lightweight cross-attention layer pointing back to the valid H_updated graphs
        # We manually implement a scaled dot-product attention projection here.
        self.query_proj = nn.Linear(latent_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        
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
        
        # 2. Extract Value Ensembles for logging (not strictly needed for sampling)
        v_mean, v_var = self.compute_values(z)
        
        # 3. Autoregressive Sampling (Pointer logic)
        # The mask_tree determines what node indices are structurally valid destinations right now.
        # e.g., action dictates picking an adjacent hex:
        # We grab the hex node embeddings: H_valid = x_dict['hex'][mask_tree['valid_hex_indices']]
        
        # For scaffolding, we mock a dot product against the first valid selection
        q = self.query_proj(z) # [batch, hidden_dim]
        
        # Example: Mocking checking hexes
        k = self.key_proj(x_dict['hex']) # [num_hexes, hidden_dim]
        
        # Dot product scores scaling
        scores = torch.matmul(q, k.T) / (q.size(-1) ** 0.5) # [batch, num_hexes]
        
        # In production:
        # apply masks (scores[~mask] = -1e9)
        # then sample:
        probs = torch.softmax(scores, dim=-1)
        action_idx = torch.multinomial(probs, num_samples=1).item()
        
        # Returns action format matching Java expectations
        return {"selected_path_index": action_idx}, v_mean.item(), probs
