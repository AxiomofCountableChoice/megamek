import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return x

class ActionConditionedPointer(nn.Module):
    def __init__(self, hidden_dim=128, action_feature_dim=8):
        super().__init__()
        
        # e_action = ActionMLP( H_unit (+) H_target_node (+) X_action_node )
        # H_unit is hidden_dim
        # H_target_node is hidden_dim
        # X_action_node is action_feature_dim
        # Total input is hidden_dim*2 + action_feature_dim
        
        # We also pass z? The Architecture says `s_k` (the internal decoder cell) is initialized with `z`.
        # For our single-step Movement action pass, we can just fuse `z` as well.
        # Actually in ARCH.md: "Logits_j = s_k^T * e_action_j".
        # Let's project e_action_j to hidden_dim, and s_k is hidden_dim.
        
        input_dim = hidden_dim * 2 + action_feature_dim
        
        self.action_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initializing the starting decoder state s_0 from global z
        self.s_0_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Autoregressive Decoder (Transformer / Decoder-Only Style)
        # Using TransformerEncoder with causal masking to emulate GPT-style autoregressive step tracking.
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)

    def decode_sequence(self, z, chosen_action_embeddings_seq):
        """
        Processes a sequence of chosen actions up to step `k` and evaluates the hidden contextual state `s_{k}`.
        `z`: Global context tensor (B, hidden_dim*2)
        `chosen_action_embeddings_seq`: Sequence of past action embeddings (B, seq_len, hidden_dim).
                                        Pass None or an empty sequence for step 0.
        Returns:
            s_k: The output state mapped to the last token natively ready for dot-product scoring. (B, hidden_dim)
        """
        # Step 0 token acts as the sequence <START>
        s_0 = self.s_0_proj(z).unsqueeze(1) # (B, 1, hidden_dim)
        
        if chosen_action_embeddings_seq is None or chosen_action_embeddings_seq.size(1) == 0:
            seq_inputs = s_0
        else:
            # Concatenate history: [s_0, a_1, a_2, ..., a_{k-1}]
            seq_inputs = torch.cat([s_0, chosen_action_embeddings_seq], dim=1) # (B, seq_len+1, hidden_dim)
            
        seq_len = seq_inputs.size(1)
        
        # Apply positional encodings
        seq_inputs = self.pos_encoder(seq_inputs)
        
        # Causal inference generation
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=z.device)
        decoded_seq = self.transformer_decoder(seq_inputs, mask=causal_mask, is_causal=True)
            
        # The hidden representation corresponding to the most recent step context
        # Extract the last token in the sequence dimension (-1) for all batches
        s_k = decoded_seq[:, -1, :] # (B, hidden_dim)
        return s_k

    def compute_action_embeddings(self, node_embeddings_dict, hetero_data):
        """ Evaluates e_action_j for all available interim action roots. 
        node_embeddings_dict: Dict from encoder containing 'hex', 'unit', 'weapon' tensors (num_nodes, hidden_dim)
        hetero_data: The PyG HeteroData batch
        
        Returns:
            e_action: (num_actions, hidden_dim) action embeddings.
        """
        device = node_embeddings_dict['hex'].device
        if 'action' not in hetero_data.node_types or hetero_data['action'].x is None or hetero_data['action'].x.size(0) == 0:
            return torch.empty((0, self.action_mlp[-1].out_features), device=device)
            
        action_features = hetero_data['action'].x  # (num_actions, action_feature_dim)
        num_actions = action_features.size(0)
        
        # We need to gather the embeddings of the corresponding source units and target nodes for each action.
        H_targets = torch.zeros((num_actions, node_embeddings_dict['hex'].size(-1)), device=device)
        
        # Check all valid spatial targets for actions (Hexes, Units, Weapons)
        if hasattr(hetero_data['action'], 'target_hex_idx'):
            tgt_idx = hetero_data['action'].target_hex_idx
            valid_mask = tgt_idx >= 0
            if valid_mask.any():
                max_hex = node_embeddings_dict['hex'].size(0) - 1
                valid_tgt_idx = tgt_idx[valid_mask]
                if (valid_tgt_idx > max_hex).any() or (valid_tgt_idx < 0).any():
                    print(f"WARNING: Out of bounds hex index! Max allowed: {max_hex}. Got min: {valid_tgt_idx.min()}, max: {valid_tgt_idx.max()}")
                    valid_tgt_idx = torch.clamp(valid_tgt_idx, 0, max_hex)
                H_targets[valid_mask] = node_embeddings_dict['hex'][valid_tgt_idx]
                
        if hasattr(hetero_data['action'], 'target_unit_idx'):
            tgt_idx = hetero_data['action'].target_unit_idx
            valid_mask = tgt_idx >= 0
            if valid_mask.any():
                max_unit = node_embeddings_dict['unit'].size(0) - 1
                valid_tgt_idx = tgt_idx[valid_mask]
                if max_unit < 0:
                    H_targets[valid_mask] = 0.0 # No units available
                else:
                    valid_tgt_idx = torch.clamp(valid_tgt_idx, 0, max_unit)
                    H_targets[valid_mask] = node_embeddings_dict['unit'][valid_tgt_idx]
                
        if hasattr(hetero_data['action'], 'target_weapon_idx'):
            tgt_idx = hetero_data['action'].target_weapon_idx
            valid_mask = tgt_idx >= 0
            if valid_mask.any():
                max_weapon = node_embeddings_dict['weapon'].size(0) - 1
                valid_tgt_idx = tgt_idx[valid_mask]
                if max_weapon < 0:
                    H_targets[valid_mask] = 0.0 # No weapons available
                else:
                    valid_tgt_idx = torch.clamp(valid_tgt_idx, 0, max_weapon)
                    H_targets[valid_mask] = node_embeddings_dict['weapon'][valid_tgt_idx]
        
        # Source Units
        H_units = torch.zeros((num_actions, node_embeddings_dict['unit'].size(-1)), device=device)
        if hasattr(hetero_data['action'], 'source_unit_idx'):
            src_idx = hetero_data['action'].source_unit_idx
            valid_mask = src_idx >= 0
            if valid_mask.any():
                max_unit = node_embeddings_dict['unit'].size(0) - 1
                valid_src_idx = src_idx[valid_mask]
                if max_unit < 0:
                    H_units[valid_mask] = 0.0
                else:
                    valid_src_idx = torch.clamp(valid_src_idx, 0, max_unit)
                    H_units[valid_mask] = node_embeddings_dict['unit'][valid_src_idx]
        
        # Concat: H_unit (+) H_target_node (+) X_action_node
        fused_inputs = torch.cat([H_units, H_targets, action_features], dim=-1)
        
        # Compute dynamic action embeddings (e_action)
        e_action = self.action_mlp(fused_inputs) # (num_actions, hidden_dim)
        
        return e_action

    def score_actions(self, s_k, e_action, action_batch_idx=None):
        """
        Computes dot-product logits for the current state s_k against action embeddings.
        Returns unnormalized logits (num_actions,).
        """
        num_actions = e_action.size(0)
        if num_actions == 0:
            return torch.empty((0,), device=s_k.device)
            
        device = s_k.device
        
        if action_batch_idx is not None:
            batch_idx = action_batch_idx
        else: # Single graph
            batch_idx = torch.zeros(num_actions, dtype=torch.long, device=device)
            
        s_k_expanded = s_k[batch_idx] # (num_actions, hidden_dim)
        
        # Dot product scoring: s_k^T * e_action
        logits = (s_k_expanded * e_action).sum(dim=-1)
        
        return logits
