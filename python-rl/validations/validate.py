import os
import torch
try:
    print('Starting')
    if not os.path.exists('bc_dataset'):
        print('No bc_dataset')
        exit(1)
    filename = os.path.join('bc_dataset', os.listdir('bc_dataset')[0])
    print('Loading:', filename)
    data = torch.load(filename, map_location='cpu', weights_only=False)
    trajectories = data['trajectories']
    print('Loaded trajectories:', len(trajectories))
    if trajectories:
        graph = trajectories[0]
        edges = sum([v.edge_index.size(1) for k, v in graph.edge_items() if 'hexAdj' in k[1]])
        print('Total hex nodes:', graph['hex'].x.size(0))
        print('Total hexAdj edges:', edges)
except Exception as e:
    print('Error:', e)
