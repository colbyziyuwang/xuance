import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import uniform_filter1d
import matplotlib.cm as cm

# Set up root paths
atari_root = os.path.expanduser("~/Desktop/Atari")
output_dir = os.path.expanduser("~/Desktop/images_sum")
os.makedirs(output_dir, exist_ok=True)

# Load scalar data and compute sum/std by step
def get_sum_std_values(file_pattern):
    matched_files = glob.glob(file_pattern)
    print(f"🔍 Found {len(matched_files)} files for pattern: {file_pattern}")

    dataframes = []
    for filename in matched_files:
        df = pd.read_csv(filename)

        # Normalize column names
        df.columns = [col.lower() for col in df.columns]

        if 'step' in df.columns and 'value' in df.columns:
            df = df[['step', 'value']].rename(columns={'step': 'Step', 'value': 'Value'})
            if not df.empty:
                dataframes.append(df)
        else:
            print(f"⚠️ Skipping {filename} — missing 'step'/'value' columns")

    if not dataframes:
        return np.array([]), np.array([]), np.array([])

    combined_df = pd.concat(dataframes)
    grouped = combined_df.groupby('Step')['Value']
    return grouped.sum().index.values, grouped.sum().values, grouped.std().values

# Smoothing
def smooth_data(data, window_size=5):
    return uniform_filter1d(data, size=window_size, mode='nearest')

# Traverse environments
env_list = [env for env in os.listdir(atari_root) if env.endswith("-v5") and os.path.isdir(os.path.join(atari_root, env))]

for env_name in env_list:
    env_path = os.path.join(atari_root, env_name)
    all_data = {}
    min_training_duration = float('inf')

    # Traverse algorithms inside the env
    algo_list = [algo for algo in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, algo))]

    for algo in algo_list:
        file_pattern = os.path.join(env_path, algo, "env-*.csv")
        steps, sum_values, std_values = get_sum_std_values(file_pattern)
        if len(steps) == 0:
            continue
        all_data[algo.upper()] = (steps, sum_values, std_values)
        min_training_duration = min(min_training_duration, max(steps))

    if not all_data:
        print(f"⚠️ Skipping {env_name}: no valid data.")
        continue

    # Plotting
    plt.figure(figsize=(12, 8))
    colors = cm.tab10(np.linspace(0, 1, len(all_data)))

    for (method, (steps, sum_values, std_values)), color in zip(all_data.items(), colors):
        mask = steps <= min_training_duration
        truncated_steps = steps[mask]
        truncated_sum = sum_values[mask]
        truncated_std = std_values[mask]
        smoothed_sum = smooth_data(truncated_sum, window_size=50)

        plt.plot(truncated_steps, smoothed_sum, label=method, color=color)

    plt.xlabel("Step")
    plt.ylabel("Summed Episode Reward")
    plt.title(f"Training Rewards (Summed): {env_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_name = env_name.replace("-", "_") + "_sum.png"
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")

print("\n🎉 All plots saved to ~/Desktop/images_sum/")
