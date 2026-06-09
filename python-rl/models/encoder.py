import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.nn.aggr import AttentionalAggregation
import torch.nn.functional as F

# Default structural graph metadata
# Adjusted to match the formal specification in ARCHITECTURE.md
DEFAULT_METADATA = (
    ['hex', 'unit', 'weapon'], 
    [
        ('hex', 'hexAdj_0', 'hex'),
        ('hex', 'hexAdj_1', 'hex'),
        ('hex', 'hexAdj_2', 'hex'),
        ('hex', 'hexAdj_3', 'hex'),
        ('hex', 'hexAdj_4', 'hex'),
        ('hex', 'hexAdj_5', 'hex'),
        ('unit', 'occupies', 'hex'),
        ('weapon', 'equips', 'unit'),
        ('weapon', 'targeted', 'unit'),
        ('unit', 'moveTypeTMM_0', 'hex'),
        ('unit', 'moveTypeTMM_1', 'hex'),
        ('unit', 'moveTypeTMM_2', 'hex'),
        ('unit', 'moveTypeTMM_3', 'hex'),
        ('unit', 'moveTypeTMM_4', 'hex'),
        ('unit', 'movementThreat', 'hex'),
        ('unit', 'LOSThreat', 'hex'),
        ('unit', 'LOSTarget', 'unit'),
        ('unit', 'partialCover', 'hex'),
    ]
)

class MegaMekHGTEncoder(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) Encoder defined in ARCHITECTURE.md (Section 3a & 3b).
    Takes a PyG HeteroData object (topology + deltas) and yields a global latent state z.
    """
    def __init__(self, hidden_dim=128, metadata=DEFAULT_METADATA, feature_dims=None, num_latent_queries=32, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.metadata = metadata
        self.num_latent_queries = num_latent_queries
        self.num_layers = num_layers
        
        if feature_dims is None:
            feature_dims = {'hex': 14, 'unit': 45, 'weapon': 10}

        # 1. Node Linear Embeddings
        # Assumes input features for hexes and units are projected into a shared latent space.
        # Action space logic evaluates valid candidates directly against the cached graph states, 
        # so 'action' nodes and their projection are decoupled from this network.
        # Using LazyLinear (-1) or predefined sizes. HGTConv natively expects projection first.
        self.node_proj = nn.ModuleDict({
            'hex': nn.Linear(feature_dims.get('hex', 14), hidden_dim),
            'unit': nn.Linear(feature_dims.get('unit', 45), hidden_dim), 
            'weapon': nn.Linear(feature_dims.get('weapon', 10), hidden_dim)
        })
        
        # 2. HGT Layers (Edge-Type Specific Message Formulation)
        in_channels_dict = {node: hidden_dim for node in metadata[0]}
        self.hgt_layers = nn.ModuleList([
            HGTConv(in_channels=in_channels_dict, out_channels=hidden_dim, 
                    metadata=metadata, heads=4)
            for _ in range(num_layers)
        ])
                            
        # 3. Perceiver-style Latent Query Pooling
        # K learnable latent queries of size hidden_dim
        self.latent_queries = nn.Parameter(torch.empty(self.num_latent_queries, hidden_dim))
        nn.init.normal_(self.latent_queries, std=0.02)
        
        # Standard MultiheadAttention for Perceiver Cross-Attention
        self.perceiver_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # 4. Context MLP (Phase Metadata -> z_context)
        self.context_mlp = nn.Sequential(
            nn.Linear(12, 64), # 12 global/metadata features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, hidden_dim)
        )

    def forward(self, hetero_data):
        """
        Forward pass producing the cached latent graph and context state.
        Returns:
            z: The combined latent state (z_graph (+) z_context)
            x_dict: The per-node updated embeddings (useful for Autoregressive pointer selection)
        """
            
        # 1. Node Projections
        x_dict = {}
        for node_type in self.metadata[0]:
            if node_type in hetero_data.node_types and hetero_data[node_type].x is not None and hetero_data[node_type].x.size(0) > 0:
                x_dict[node_type] = self.node_proj[node_type](hetero_data[node_type].x)
            else:
                x_dict[node_type] = torch.empty((0, self.hidden_dim), device=hetero_data['hex'].x.device)
                
        # Filter edge_index_dict to strictly match metadata to prevent PyG internal offset corruption
        edge_index_dict = {
            edge_type: hetero_data.edge_index_dict[edge_type] 
            for edge_type in self.metadata[1] 
            if edge_type in hetero_data.edge_index_dict
        }
        
        # 2. HGT Spatial Pass
        for conv in self.hgt_layers:
            x_dict_out = conv(x_dict, edge_index_dict)
            # If HGTConv drops isolated nodes from its output dict, fallback to previous layer tensor to preserve graph shape
            x_dict = {k: F.gelu(x_dict_out.get(k, x_dict[k])) for k in self.metadata[0]}
        
        # 3. Perceiver-style Latent Query Pooling
        # Group units and weapons per batch item.
        ctx = hetero_data.global_context
        if ctx.dim() == 1:
            ctx = ctx.view(-1, 12)
        batch_size = ctx.size(0)
        z_context = self.context_mlp(ctx)
        device = hetero_data['hex'].x.device
        
        batch_entities_list = []
        for b in range(batch_size):
            # Gather units for batch b
            if hasattr(hetero_data['unit'], 'batch') and hetero_data['unit'].batch is not None:
                unit_mask = (hetero_data['unit'].batch == b)
            else:
                unit_mask = torch.ones(x_dict['unit'].size(0), dtype=torch.bool, device=device)
            units_b = x_dict['unit'][unit_mask]
            
            # Gather weapons for batch b
            if hasattr(hetero_data['weapon'], 'batch') and hetero_data['weapon'].batch is not None:
                weapon_mask = (hetero_data['weapon'].batch == b)
            else:
                weapon_mask = torch.ones(x_dict['weapon'].size(0), dtype=torch.bool, device=device)
            weapons_b = x_dict['weapon'][weapon_mask]
            
            entities_b = torch.cat([units_b, weapons_b], dim=0)
            batch_entities_list.append(entities_b)
            
        max_NE = max(entities.size(0) for entities in batch_entities_list)
        if max_NE == 0:
            max_NE = 1
            
        H_entities_padded = torch.zeros((batch_size, max_NE, self.hidden_dim), device=device)
        key_padding_mask = torch.ones((batch_size, max_NE), dtype=torch.bool, device=device)
        
        for b in range(batch_size):
            entities_b = batch_entities_list[b]
            NE = entities_b.size(0)
            if NE > 0:
                H_entities_padded[b, :NE] = entities_b
                key_padding_mask[b, :NE] = False
            else:
                key_padding_mask[b, 0] = False  # Avoid completely empty mask warning/error
                
        # Z_latent = MultiHeadAttention(Q=L, K=H_entities, V=H_entities)
        # Expand latent queries for the batch
        Q_latent = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, K, d)
        
        # Cross attention
        z_latent, _ = self.perceiver_attn(
            query=Q_latent,
            key=H_entities_padded,
            value=H_entities_padded,
            key_padding_mask=key_padding_mask
        )  # (batch_size, K, d)
        
        # Flatten
        z_graph = z_latent.reshape(batch_size, -1)  # (batch_size, K * d)
        
        # 5. Latent Fusion
        z = torch.cat([z_graph, z_context], dim=-1)
        
        return z, x_dict
