import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import sys
import argparse


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine.ddp_engine import DDPEngine
from utils.engine_utils import tprint, load_cfg, set_random_seed, generate_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--eval', type=bool, default=False, help='Whether to evaluate ddp model.')
    args = parser.parse_args()

    return args

def init_dist():
    dist.init_process_group('nccl', init_method='env://')

    rank = dist.get_rank()
    local_rank = os.environ['LOCAL_RANK']
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    print(f"rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank}")
    torch.cuda.set_device(rank)
    return rank

def main():
    args = parse_args()
    # Get Config from 'config/defaults.py'
    cfg = load_cfg(args.config)

    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.OUTPUT_DIR = args.work_dir

    local_rank = init_dist()

    torch_version = int(torch.__version__.split('.')[1])
    if torch_version >= 7:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


    # Set Benchmark
    # If this is set to True, it may consume more memory. (Default: True)
    if cfg.get('USE_BENCHMARK', True):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if local_rank == 0:
            tprint(f"CuDNN Benchmark is enabled.")


    # Set Random Seed
    seed = cfg.get('SEED', -1)
    seed = generate_random_seed(seed)
    set_random_seed(seed)
    
    cfg.SEED = seed
    cfg.GPU_ID = local_rank
    cfg.freeze()
    engine = DDPEngine(cfg)

    if args.eval:
        if local_rank == 0: eval_dict = engine.evaluate()
    else: engine.train()
    


if __name__ =='__main__':
    main()
