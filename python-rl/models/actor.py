import torch
import torch.nn as nn

class ActionConditionedPointer(nn.Module):
    def __init__(self, hidden_dim=128, action_feature_dim=3):
        super().__init__()
        
        # e_action = ActionMLP( H_unit (+) H_target_hex (+) X_action_node )
        # H_unit is hidden_dim
        # H_target_hex is hidden_dim
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
        
        # Causal inference generation (if sequence is larger than 1)
        if seq_len > 1:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=z.device)
            decoded_seq = self.transformer_decoder(seq_inputs, mask=causal_mask, is_causal=True)
        else:
            decoded_seq = self.transformer_decoder(seq_inputs)
            
        # The hidden representation corresponding to the most recent step context
        s_k = decoded_seq[:, -1, :] # (B, hidden_dim)
        return s_k

    def compute_action_embeddings(self, node_embeddings_dict, hetero_data):
        """ Evaluates e_action_j for all available interim action roots. """
        z: Global context tensor (B, hidden_dim*2)
        node_embeddings_dict: Dict from encoder containing 'hex', 'mech' tensors (num_nodes, hidden_dim)
        hetero_data: The PyG HeteroData batch
        
        Returns:
            logits: (num_actions,) vector of unnormalized scores per action.
        """
        device = node_embeddings_dict['hex'].device
        if 'action' not in hetero_data.node_types or hetero_data['action'].x.size(0) == 0:
            return torch.empty((0, self.action_mlp[-1].out_features), device=device)
            
        action_features = hetero_data['action'].x  # (num_actions, action_feature_dim)
        num_actions = action_features.size(0)
        
        # We need to gather the embeddings of the corresponding source mechs and target hexes for each action.
        # 1. Target Hexes: using ['action', 'targets', 'hex'].edge_index
        target_edges = hetero_data['action', 'targets', 'hex'].edge_index
        # target_edges[0] is the action idx, target_edges[1] is the hex idx
        
        # Sort or map hexes to actions
        # Action edges should be 1-to-1 for MOVEMENT roots.
        H_hexes = torch.zeros((num_actions, node_embeddings_dict['hex'].size(-1)), device=device)
        H_hexes[target_edges[0]] = node_embeddings_dict['hex'][target_edges[1]]
        
        # 2. Source Mechs: using ['mech', 'considers', 'action'].edge_index
        source_edges = hetero_data['mech', 'considers', 'action'].edge_index
        # source_edges[0] is mech idx, source_edges[1] is action idx
        H_mechs = torch.zeros((num_actions, node_embeddings_dict['mech'].size(-1)), device=device)
        H_mechs[source_edges[1]] = node_embeddings_dict['mech'][source_edges[0]]
        
        # Concat: H_unit (+) H_target_hex (+) X_action_node
        fused_inputs = torch.cat([H_mechs, H_hexes, action_features], dim=-1)
        
        # Compute dynamic action embeddings (e_action)
        e_action = self.action_mlp(fused_inputs) # (num_actions, hidden_dim)
        
        return e_action

    def score_actions(self, s_k, e_action, hetero_data):
        """
        Computes dot-product logits for the current state s_k against action embeddings.
        Returns unnormalized logits (num_actions,).
        """
        num_actions = e_action.size(0)
        if num_actions == 0:
            return torch.empty((0,), device=s_k.device)
            
        # Broadcast the internal state s_k to each action evaluating 
        # using PyG batch indexes.
        s_0 = self.s_0_proj(z) # (B, hidden_dim)
        
        # Get the batch index for each action from PyG
        # if PyG DataLoader batched it, hetero_data['action'].batch has the batch indices.
        if hasattr(hetero_data['action'], 'batch') and getattr(hetero_data['action'], 'batch') is not None:
            batch_idx = hetero_data['action'].batch
        else: # Single graph
            batch_idx = torch.zeros(num_actions, dtype=torch.long, device=device)
            
        s_k_expanded = s_k[batch_idx] # (num_actions, hidden_dim)
        
        # Dot product scoring: s_k^T * e_action
        logits = (s_k_expanded * e_action).sum(dim=-1)
        
        return logits
