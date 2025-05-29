import os
import re
import pandas as pd
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_root = os.path.expanduser("~/Desktop/logs")
output_root = os.path.expanduser("~/Desktop/Atari")

# Traverse algorithms
algos = [a for a in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, a))]

for algo in tqdm(algos, desc="Algorithms"):
    algo_path = os.path.join(log_root, algo)
    ale_dir = os.path.join(algo_path, "torch", "ALE")
    if not os.path.isdir(ale_dir):
        continue

    envs = [e for e in os.listdir(ale_dir) if os.path.isdir(os.path.join(ale_dir, e))]

    for env in tqdm(envs, desc=f"{algo}: Envs", leave=False):
        env_path = os.path.join(ale_dir, env)
        seed_dirs = [s for s in os.listdir(env_path) if s.startswith("seed_")]
        
        for seed in tqdm(seed_dirs, desc=f"{algo}/{env}: Seeds", leave=False):
            seed_path = os.path.join(env_path, seed)

            reward_dirs = [
                d for d in os.listdir(seed_path)
                if d.startswith("Train-Episode-Rewards_rank_0_env-")
                   and os.path.isdir(os.path.join(seed_path, d))
            ]

            for reward_dir in tqdm(reward_dirs, desc=f"{algo}/{env}/{seed}: Logs", leave=False):
                full_path = os.path.join(seed_path, reward_dir)
                match = re.search(r'env-(\d+)', reward_dir)
                if not match:
                    continue
                env_index = match.group(1)

                try:
                    ea = EventAccumulator(full_path)
                    ea.Reload()

                    tag = "Train-Episode-Rewards/rank_0"
                    if tag not in ea.Tags()["scalars"]:
                        continue

                    events = ea.Scalars(tag)
                    df = pd.DataFrame(events)

                    output_dir = os.path.join(output_root, env, algo)
                    os.makedirs(output_dir, exist_ok=True)

                    output_file = os.path.join(output_dir, f"env-{env_index}.csv")
                    df.to_csv(output_file, index=False)
                except Exception as e:
                    tqdm.write(f"❌ Failed at {full_path}: {e}")
