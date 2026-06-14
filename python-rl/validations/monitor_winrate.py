import time
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
metrics_file = os.path.abspath(os.path.join(script_dir, "..", "data", "rl_princess_trajectories", "metrics_log.csv"))

def get_win_rate():
    if not os.path.exists(metrics_file):
        return 0.0
    try:
        df = pd.read_csv(metrics_file, names=['timestamp', 'hash', 'win', 'reward'])
        if len(df) == 0: return 0.0
        # Calculate win rate over the last 100 matches
        recent = df.tail(100)
        # win column is string 'True'/'False' or bool
        wins = recent['win'].apply(lambda x: str(x).lower() == 'true').sum()
        return wins / len(recent)
    except Exception as e:
        print(f"Error reading metrics: {e}")
        return 0.0

print("Monitoring win rate...")
start_time = time.time()
while True:
    wr = get_win_rate()
    elapsed = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] Current Win Rate (last 100 matches): {wr*100:.2f}%", flush=True)
    if wr >= 0.15:
        print(f"Goal Reached! Win Rate is {wr*100:.2f}%")
        break
    time.sleep(60)
