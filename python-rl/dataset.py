import torch
from torch.utils.data import IterableDataset

try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

class MegaMekTrajectoryDataset(IterableDataset):
    """
    Offline Behavioural Cloning Dataset (Section 5c).
    Streams binary chunks dumped from the headless Java `Princess` exporter 
    and delta reconstructs PyG HeteroData exactly like env.py.
    """
    def __init__(self, binary_export_path):
        super().__init__()
        self.binary_path = binary_export_path
        self.topology_cache = None

    def __iter__(self):
        # In production this will yield (hetero_data, valid_mask_tree, expert_action_idx) tuples.
        # By iterating offline, we warm start the Transformer.
        while True:
            # yield self._parse_next_delta()
            break 
