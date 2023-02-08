import os
import sys
import torch
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine.mono_engine import MonoEngine
from utils.engine_utils import tprint, load_cfg, set_random_seed, generate_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    args = parser.parse_args()

    return args

def main():
    # Some Torch Settings
    torch_version = int(torch.__version__.split('.')[1])
    if torch_version >= 7:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    args = parse_args()
    # Get Config from 'config/defaults.py'
    cfg = load_cfg(args.config)

    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.OUTPUT_DIR = args.work_dir


    # Set Benchmark
    # If this is set to True, it may consume more memory. (Default: True)
    if cfg.get('USE_BENCHMARK', True):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        tprint(f"CuDNN Benchmark is enabled.")


    # Set Random Seed
    seed = cfg.get('SEED', -1)
    seed = generate_random_seed(seed)
    set_random_seed(seed)
    
    cfg.SEED = seed
    tprint(f"Using Random Seed {seed}")

    cfg.freeze()
    # Initialize Engine
    engine = MonoEngine(cfg)


    # Start Training from Scratch
    # Output files will be saved to 'cfg.OUTPUT_DIR'.
    engine.train()



if __name__ == '__main__':
    main()