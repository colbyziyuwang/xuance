import argparse
from xuance import get_runner
import os

import torch

# Limit CPU threads to avoid excessive usage
torch.set_num_threads(4)

def parse_args():
    parser = argparse.ArgumentParser("Run a demo.")
    parser.add_argument("--method", type=str, default="dqn")
    parser.add_argument("--env", type=str, default="classic_control")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="mps")

    return parser.parse_args()


if __name__ == '__main__':
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    parser = parse_args()
    parser.env = "Atari"
    parser.env_id = "ALE/Breakout-v5"
    parser.method = "perdqn"
    ########################################################################
    # You can also modify the arguments not listed above here. For example:
    # parser.parallels = 1
    # parser.running_steps = 1000000
    # ...
    ########################################################################
    runner = get_runner(method=parser.method,
                        env=parser.env,
                        env_id=parser.env_id,
                        parser_args=parser,
                        is_test=False)
    runner.run()
