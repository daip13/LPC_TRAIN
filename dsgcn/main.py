from __future__ import division
import sys
import os
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv import Config
from utils import (create_logger, set_random_seed,
                    rm_suffix, mkdir_if_no_exists)

from dsgcn.models import build_model
from dsgcn import build_handler


def parse_args():
    parser = argparse.ArgumentParser(description='Cluster Detection and Segmentation')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--stage', choices=['det', 'seg', 'mall'], default='det')
    parser.add_argument('--phase', choices=['test', 'train'], default='test')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--load_from1', default=None, help='the checkpoint file to load from')
    parser.add_argument('--load_from2', default=None, help='the checkpoint file to load from')
    parser.add_argument('--load_from3', default=None, help='the checkpoint file to load from')
    parser.add_argument('--resume_from', default=None, help='the checkpoint file to resume from')
    parser.add_argument('--gpus', type=int, default=8,
            help='number of gpus(only applicable to non-distributed training)')
    parser.add_argument('--distributed', action='store_true', default=False)
    parser.add_argument('--save_output', action='store_true', default=False)
    parser.add_argument('--launcher', action='store_true', default='pytorch')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    args = parser.parse_args()

    return args

def _init_dist_pytorch(backend, **kwargs):
        # TODO: use local_rank instead of rank % num_gpus
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend=backend, **kwargs)

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # set cuda
    cfg.cuda = not args.no_cuda and torch.cuda.is_available()
    # set cudnn_benchmark & cudnn_deterministic
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('cudnn_deterministic', False):
        torch.backends.cudnn.deterministic = True
    # update configs according to args
    if not hasattr(cfg, 'work_dir'):
        if args.work_dir is not None:
            cfg.work_dir = args.work_dir
        else:
            cfg_name = rm_suffix(os.path.basename(args.config))
            cfg.work_dir = os.path.join('./data/work_dir', cfg_name)
    mkdir_if_no_exists(cfg.work_dir, is_folder=True)
    if not hasattr(cfg, 'stage'):
        cfg.stage = args.stage

    cfg.load_from1 = args.load_from1
    cfg.load_from2 = args.load_from2
    cfg.load_from3 = args.load_from3
    cfg.resume_from = args.resume_from

    #cfg.gpus = args.gpus
    cfg.distributed = args.distributed
    cfg.save_output = args.save_output
    cfg.phase = args.phase
    logger = create_logger()

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = [build_model(cfg.model1['type'], **cfg.model1['kwargs']), \
            build_model(cfg.model2['type'], **cfg.model2['kwargs']), \
            build_model(cfg.model3['type'], **cfg.model3['kwargs'])]
    if cfg.phase == 'train':
        if cfg.load_from1:
            model1, model2, model3 = model[0], model[1], model[2]
            model1.load_state_dict(torch.load(cfg.load_from1))
            model[0] = model1
        if cfg.load_from2:
            model2.load_state_dict(torch.load(cfg.load_from2))
            model[1] = model2
        if cfg.load_from3:
            model3.load_state_dict(torch.load(cfg.load_from3))
            model[2] = model3
    handler = build_handler(args.phase, args.stage)

    handler(model, cfg, logger)


if __name__ == '__main__':
    main()
