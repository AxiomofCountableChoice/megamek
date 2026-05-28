import torch
import sys

traj_path = "data/rl_princess_trajectories/96281695/traj_6_1779802339.pt"
data = torch.load(traj_path, weights_only=False, map_location='cpu')

print(f"File: {traj_path}")
print(f"Steps: {len(data['steps'])}")

# Print first 5 and last 5 steps to see the phase progression
steps = data['steps']
for i, step in enumerate(steps):
    if i < 5 or i >= len(steps) - 5:
        payload = step.get('raw_payload', {})
        turn = payload.get('turn_number', -1)
        phase = payload.get('phase', 'UNKNOWN')
        entities = payload.get('entities', [])
        print(f"Step {i}: Turn {turn}, Phase {phase}, Entities count: {len(entities)}")
    elif i == 5:
        print("...")
