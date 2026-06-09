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
    def __init__(self, hidden_dim=128, action_feature_dim=8, latent_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # e_action = ActionMLP( H_unit (+) H_target_node (+) X_action_node )
        # H_unit is hidden_dim
        # H_target_node is hidden_dim
        # X_action_node is action_feature_dim
        # Total input is hidden_dim*2 + action_feature_dim
        
        input_dim = hidden_dim * 2 + action_feature_dim
        
        self.action_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initializing the starting decoder state s_0 from global z
        if latent_dim is None:
            latent_dim = hidden_dim * 2
        self.s_0_proj = nn.Linear(latent_dim, hidden_dim)
        
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

        # Spatial Cross-Attention over un-pooled graph nodes
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.W_pointer = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W_pointer)

    def decode_sequence(self, z, chosen_action_embeddings_seq):
        """
        Processes a sequence of chosen actions up to step `k` and evaluates the hidden contextual state `s_{k}`.
        `z`: Global context tensor (B, latent_dim)
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

    def score_actions(self, s_k, e_action, action_batch_idx, x_dict, hetero_data):
        """
        Computes attention-based logits for the current state s_k against action embeddings.
        Utilizes a Perceiver/pointer cross-attention mechanism over the un-pooled graph nodes.
        """
        num_actions = e_action.size(0)
        if num_actions == 0:
            return torch.empty((0,), device=s_k.device)
            
        device = s_k.device
        
        # 1. Determine batch index for each action
        if action_batch_idx is not None:
            act_batch_idx = action_batch_idx
        else:
            act_batch_idx = torch.zeros(num_actions, dtype=torch.long, device=device)
            
        if s_k.size(0) == 1:
            s_k_expanded = s_k.expand(num_actions, -1)
        else:
            s_k_expanded = s_k[act_batch_idx]
            
        # 2. Compute candidate action query q_{a_k} = QueryMLP(s_k \oplus e_{a_k})
        fused_q = torch.cat([s_k_expanded, e_action], dim=-1)
        q_action = self.query_mlp(fused_q)  # (num_actions, hidden_dim)
        
        # 3. Gather and pad the un-pooled graph nodes for each action
        if hasattr(hetero_data['hex'], 'batch') and hetero_data['hex'].batch is not None:
            batch_size = int(hetero_data['hex'].batch.max().item() + 1)
        else:
            batch_size = 1
            
        batch_nodes_list = []
        for b in range(batch_size):
            if hasattr(hetero_data['hex'], 'batch') and hetero_data['hex'].batch is not None:
                hex_mask = (hetero_data['hex'].batch == b)
            else:
                hex_mask = torch.ones(x_dict['hex'].size(0), dtype=torch.bool, device=device)
            hex_nodes = x_dict['hex'][hex_mask]
            
            if hasattr(hetero_data['unit'], 'batch') and hetero_data['unit'].batch is not None:
                unit_mask = (hetero_data['unit'].batch == b)
            else:
                unit_mask = torch.ones(x_dict['unit'].size(0), dtype=torch.bool, device=device)
            unit_nodes = x_dict['unit'][unit_mask]
            
            if hasattr(hetero_data['weapon'], 'batch') and hetero_data['weapon'].batch is not None:
                weapon_mask = (hetero_data['weapon'].batch == b)
            else:
                weapon_mask = torch.ones(x_dict['weapon'].size(0), dtype=torch.bool, device=device)
            weapon_nodes = x_dict['weapon'][weapon_mask]
            
            nodes_b = torch.cat([hex_nodes, unit_nodes, weapon_nodes], dim=0)
            batch_nodes_list.append(nodes_b)
            
        referenced_batches = act_batch_idx.unique().tolist()
        max_nodes = max(batch_nodes_list[b].size(0) for b in referenced_batches)
        if max_nodes == 0:
            max_nodes = 1
            
        K_padded = torch.zeros((num_actions, max_nodes, self.hidden_dim), device=device)
        key_padding_mask = torch.ones((num_actions, max_nodes), dtype=torch.bool, device=device)
        
        if num_actions > 0:
            first_b = int(act_batch_idx[0].item())
            if (act_batch_idx == first_b).all():
                # Fast vectorized path when all actions share the same batch item
                nodes_b = batch_nodes_list[first_b]
                num_nodes = nodes_b.size(0)
                if num_nodes > 0:
                    K_padded[:, :num_nodes, :] = nodes_b.unsqueeze(0)
                    key_padding_mask[:, :num_nodes] = False
                else:
                    key_padding_mask[:, 0] = False
            else:
                # Fallback to loop if action batch indices are heterogeneous
                for j in range(num_actions):
                    b = int(act_batch_idx[j].item())
                    if b >= len(batch_nodes_list):
                        b = 0
                    nodes_b = batch_nodes_list[b]
                    num_nodes = nodes_b.size(0)
                    if num_nodes > 0:
                        K_padded[j, :num_nodes] = nodes_b
                        key_padding_mask[j, :num_nodes] = False
                    else:
                        key_padding_mask[j, 0] = False
                
        # 4. Multi-Head Cross-Attention
        Q = q_action.unsqueeze(1)
        
        context, _ = self.cross_attention(
            query=Q,
            key=K_padded,
            value=K_padded,
            key_padding_mask=key_padding_mask
        )  # (num_actions, 1, hidden_dim)
        context = context.squeeze(1)  # (num_actions, hidden_dim)
        
        # 5. Logits(a_k) = (1/sqrt(d)) * q_{a_k}^T * W_pointer * Context_{a_k}
        q_W = torch.matmul(q_action, self.W_pointer)
        logits = torch.sum(q_W * context, dim=-1) / math.sqrt(self.hidden_dim)
        
        return logits
