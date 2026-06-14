import csv
import datetime
import os
import math

def check_trend():
    # Resolve the path to metrics_log.csv relative to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "data", "rl_princess_trajectories", "metrics_log.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: metrics_log.csv does not exist yet at: {csv_path}")
        return
        
    records = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append({
                    "timestamp": int(row["timestamp"]),
                    "worker_id": row["worker_id"],
                    "win": row["win"].strip().lower() in ("true", "1", "yes"),
                    "total_reward": float(row["total_reward"])
                })
            except (ValueError, KeyError):
                continue
                
    if not records:
        print("No valid matches logged in metrics_log.csv yet.")
        return
        
    # Sort chronologically
    records.sort(key=lambda x: x["timestamp"])
    
    total_matches = len(records)
    print("=" * 60)
    print(f" IMPALA RL TRAINING PROGRESS REPORT ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 60)
    print(f"Total Matches Played: {total_matches}")
    
    # Calculate global stats
    wins = sum(1 for r in records if r["win"])
    win_rate = (wins / total_matches) * 100
    rewards = [r["total_reward"] for r in records]
    mean_rew = sum(rewards) / total_matches
    min_rew = min(rewards)
    max_rew = max(rewards)
    
    print(f"Overall Win Rate: {win_rate:.2f}% ({wins}/{total_matches})")
    print(f"Overall Reward - Mean: {mean_rew:.4f} | Min: {min_rew:.4f} | Max: {max_rew:.4f}\n")
    
    # Bucket stats (bucket size of 10)
    bucket_size = 10
    num_buckets = math.ceil(total_matches / bucket_size)
    
    print("| Bucket | Matches | Win Rate | Mean Reward | Min Reward | Max Reward | Std Dev |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for i in range(num_buckets):
        start_idx = i * bucket_size
        end_idx = min(start_idx + bucket_size, total_matches)
        bucket_records = records[start_idx:end_idx]
        b_len = len(bucket_records)
        
        b_wins = sum(1 for r in bucket_records if r["win"])
        b_win_rate = (b_wins / b_len) * 100
        b_rewards = [r["total_reward"] for r in bucket_records]
        b_mean = sum(b_rewards) / b_len
        b_min = min(b_rewards)
        b_max = max(b_rewards)
        
        b_variance = sum((r - b_mean) ** 2 for r in b_rewards) / b_len
        b_std = math.sqrt(b_variance)
        
        print(f"| {i+1} ({start_idx+1}-{end_idx}) | {b_len} | {b_win_rate:.1f}% | {b_mean:.4f} | {b_min:.4f} | {b_max:.4f} | {b_std:.4f} |")
        
    # Print trend analysis
    if num_buckets >= 2:
        first_bucket = records[:bucket_size]
        last_bucket = records[-bucket_size:]
        first_mean = sum(r["total_reward"] for r in first_bucket) / len(first_bucket)
        last_mean = sum(r["total_reward"] for r in last_bucket) / len(last_bucket)
        diff = last_mean - first_mean
        print(f"\nTrend Analysis (Last {len(last_bucket)} vs First {len(first_bucket)} matches):")
        print(f"  First Bucket Mean Reward: {first_mean:.4f}")
        print(f"  Last Bucket Mean Reward:  {last_mean:.4f}")
        if diff > 0:
            print(f"  Improvement: +{diff:.4f} (Upward Trend)")
        elif diff < 0:
            print(f"  Improvement: {diff:.4f} (Downward Trend)")
        else:
            print(f"  No change in mean reward.")
    print("=" * 60)

if __name__ == "__main__":
    check_trend()
