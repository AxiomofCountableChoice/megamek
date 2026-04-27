import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from models.agent import MegaMekAgent

dataset = torch.load('../bc_dataset_master_padded.pt', weights_only=False, map_location='cpu')
loader = DataLoader(dataset, batch_size=1, shuffle=False)
agent = MegaMekAgent(hidden_dim=128)

for i, batch in enumerate(loader):
    try:
        agent.encoder(batch)
    except Exception as e:
        print(f"CRASH AT BATCH {i}")
        tot = 0
        for k in ['hex', 'mech', 'action']:
            sz = batch[k].x.size(0) if getattr(batch, k, None) is not None and getattr(batch[k], 'x', None) is not None else 0
            print(f"Node {k}: {sz}")
            tot += sz
        print("Total nodes expected:", tot)
        import traceback
        traceback.print_exc()
        break
