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
    def __init__(self, hidden_dim=128, metadata=DEFAULT_METADATA, feature_dims=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.metadata = metadata
        
        if feature_dims is None:
            feature_dims = {'hex': 14, 'unit': 45, 'weapon': 10}

        # 1. Node Linear Embeddings
        # Assumes input features for hexes and units are projected into a shared latent space.
        # Action space logic evaluates valid candidates directly against the cached graph states, 
        # so 'action' nodes and their projection are decoupled from this network.
        # Using LazyLinear (-1) or predefined sizes. HGTConv natively expects projection first.
        self.node_proj = nn.ModuleDict({
            'hex': nn.Linear(feature_dims.get('hex', 14), hidden_dim),
            'unit': nn.Linear(feature_dims.get('unit', 37), hidden_dim), 
            'weapon': nn.Linear(feature_dims.get('weapon', 10), hidden_dim)
        })
        
        # 2. HGT Layers (Edge-Type Specific Message Formulation)
        in_channels_dict = {node: hidden_dim for node in metadata[0]}
        self.hgt1 = HGTConv(in_channels=in_channels_dict, out_channels=hidden_dim, 
                            metadata=metadata, heads=4)
        self.hgt2 = HGTConv(in_channels=in_channels_dict, out_channels=hidden_dim, 
                            metadata=metadata, heads=4)
        self.hgt3 = HGTConv(in_channels=in_channels_dict, out_channels=hidden_dim, 
                            metadata=metadata, heads=4)
                            
        # 3. Global Attention Pooling (Graph Readout)
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.global_pool = AttentionalAggregation(gate_nn=self.gate_nn)
        
        # 4. Context MLP (Phase Metadata -> z_context)
        self.context_mlp = nn.Sequential(
            nn.Linear(2, 64), # phase length & turn num
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

    def forward(self, hetero_data):
        """
        Forward pass producing the cached latent graph and context state.
        Returns:
            z: The combined latent state (z_graph (+) z_context)
            x_dict: The per-node updated embeddings (useful for Autoregressive pointer selection)
        """
            
        # 1. Linear Node Embeddings
        x_dict = {}
        for node_type in self.metadata[0]:
            if node_type in hetero_data.node_types and hetero_data[node_type].x is not None and hetero_data[node_type].x.size(0) > 0:
                x_dict[node_type] = self.node_proj[node_type](hetero_data[node_type].x)
            else:
                x_dict[node_type] = torch.empty((0, self.hidden_dim), device=hetero_data['hex'].x.device)
                
        # PyG HGTConv requires all node types defined in metadata to be present in x_dict
        for node in self.metadata[0]:
            if node not in x_dict:
                x_dict[node] = torch.empty((0, self.hidden_dim), device=hetero_data['hex'].x.device)
                
        # Filter edge_index_dict to strictly match metadata to prevent PyG internal offset corruption
        edge_index_dict = {
            edge_type: hetero_data.edge_index_dict[edge_type] 
            for edge_type in self.metadata[1] 
            if edge_type in hetero_data.edge_index_dict
        }
        
        # 2. HGT Spatial Pass
        # Layer 1
        x_dict_out = self.hgt1(x_dict, edge_index_dict)
        # If HGTConv drops isolated nodes from its output dict, fallback to previous layer tensor to preserve graph shape
        x_dict = {k: F.gelu(x_dict_out.get(k, x_dict[k])) for k in self.metadata[0]}
        
        # Layer 2
        x_dict_out = self.hgt2(x_dict, edge_index_dict)
        x_dict = {k: F.gelu(x_dict_out.get(k, x_dict[k])) for k in self.metadata[0]}

        # Layer 3
        x_dict_out = self.hgt3(x_dict, edge_index_dict)
        x_dict = {k: F.gelu(x_dict_out.get(k, x_dict[k])) for k in self.metadata[0]}
        
        # 3. Graph Readout (Global Attention Pooling)
        # We need to construct a single batched node matrix for pooling.
        # Here we concat across core object components (hexes, units, weapons) to form the environment state
        flat_nodes = torch.cat([x_dict['hex'], x_dict['unit'], x_dict['weapon']], dim=0)
        
        # batch tensor handles disconnected subgraphs in PyG, defaulting to 0 for a single graph
        if flat_nodes.size(0) > 0:
            batch = torch.zeros(flat_nodes.size(0), dtype=torch.long, device=flat_nodes.device)
            z_graph = self.global_pool(flat_nodes, batch)
        else:
            z_graph = torch.zeros((1, self.hidden_dim), device=flat_nodes.device)
        
        # 4. Context processing
        z_context = self.context_mlp(hetero_data.global_context.unsqueeze(0))
        
        # 5. Latent Fusion
        z = torch.cat([z_graph, z_context], dim=-1)
        
        return z, x_dict
