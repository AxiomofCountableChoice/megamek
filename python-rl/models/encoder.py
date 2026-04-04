import torch
import torch.nn as nn

try:
    from torch_geometric.nn import HANConv, global_mean_pool
    # PyG does not have a "GlobalAttention" as a direct class doing exactly what ARCH.md wants,
    # but it does have `GlobalAttention` in `torch_geometric.nn.glob`. 
    from torch_geometric.nn.glob import GlobalAttention
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

class MegaMekHANEncoder(nn.Module):
    """
    Heterogeneous Graph Attention Network Encoder defined in ARCHITECTURE.md (Section 3a & 3b).
    Takes a PyG HeteroData object (topology + deltas) and yields a global latent state z.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        if not HAS_PYG:
            return
            
        # 1. Node Linear Embeddings
        # Assumes input features for hexes and mechs are projected into a shared latent space
        self.hex_proj = nn.Linear(5, hidden_dim)
        self.mech_proj = nn.Linear(12, hidden_dim) 
        
        # 2. HAN Layers (Intra-Meta-Path Attention & Semantic Fusing)
        # We define the meta-paths natively
        metadata = (
            ['hex', 'mech'], 
            [
                ('hex', 'adjacent_to', 'hex'),
                ('mech', 'occupies', 'hex'),
                # ('mech', 'los', 'mech') # Additional edge to implement later
            ]
        )
        
        self.han1 = HANConv(in_channels=hidden_dim, out_channels=hidden_dim, 
                            metadata=metadata, heads=4)
        self.han2 = HANConv(in_channels=hidden_dim, out_channels=hidden_dim, 
                            metadata=metadata, heads=4)
                            
        # 3. Global Attention Pooling (Graph Readout)
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.global_pool = GlobalAttention(gate_nn=self.gate_nn)
        
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
        if not HAS_PYG:
            # Dummy fallback if PyG missing
            z = torch.zeros((1, self.hidden_dim * 2))
            return z, {"hex": torch.zeros((1, self.hidden_dim))}
            
        # 1. Linear Node Embeddings
        x_dict = {
            'hex': self.hex_proj(hetero_data['hex'].x),
            'mech': self.mech_proj(hetero_data['mech'].x)
        }
        edge_index_dict = hetero_data.edge_index_dict
        
        # 2. HAN Spatial Pass
        x_dict = self.han1(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        x_dict = self.han2(x_dict, edge_index_dict)
        
        # 3. Graph Readout (Global Attention Pooling)
        # We need to construct a single batched node matrix for pooling.
        # Since node types are different, we pool them independently and mix, 
        # or concatenate all into a flat sequence if they share the hidden_dim.
        # Here we concat across node types to allow the gate to value specific mechs/hexes.
        flat_nodes = torch.cat([x_dict['hex'], x_dict['mech']], dim=0)
        
        # batch tensor handles disconnected subgraphs in PyG, defaulting to 0 for a single graph
        batch = torch.zeros(flat_nodes.size(0), dtype=torch.long, device=flat_nodes.device)
        
        z_graph = self.global_pool(flat_nodes, batch)
        
        # 4. Context processing
        z_context = self.context_mlp(hetero_data.global_context.unsqueeze(0))
        
        # 5. Latent Fusion
        z = torch.cat([z_graph, z_context], dim=-1)
        
        return z, x_dict
