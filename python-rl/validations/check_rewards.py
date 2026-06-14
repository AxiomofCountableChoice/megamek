import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
metrics_file = os.path.abspath(os.path.join(script_dir, "..", "data", "rl_princess_trajectories", "metrics_log.csv"))

df = pd.read_csv(metrics_file, names=['timestamp', 'hash', 'win', 'reward'])
if len(df) > 0:
    # Print average reward for blocks of 500 matches
    chunk_size = 500
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        avg_reward = chunk['reward'].mean()
        win_rate = chunk['win'].mean() * 100
        print(f"Matches {i} to {i+len(chunk)}: Avg Reward = {avg_reward:.2f}, Win Rate = {win_rate:.2f}%")
else:
    print("No data.")
