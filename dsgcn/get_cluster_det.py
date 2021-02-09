from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

import json
import os
import torch
import numpy as np

from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from dsgcn.datasets import build_dataset, build_processor, build_dataloader

def get_data(model, output_path, cfg, is_test = False, num_runs = 1):
    if is_test:
        dataset = build_dataset(cfg.test_data)
    else:
        dataset = build_dataset(cfg.train_data)
    processor = build_processor(cfg.stage)

    data_loader = build_dataloader(
            dataset,
            processor,
            cfg.batch_size_per_gpu,
            cfg.workers_per_gpu,
            train=False)

    model = MMDataParallel(model, device_ids=range(cfg.gpus))
    if cfg.cuda:
        model.cuda()
    all_feas = []
    all_iops = []
    for k in range(num_runs):
        for i, data in enumerate(data_loader):
            print('\t running ' + str(k) + 'th run, ' + str(i) + 'th batch')
            with torch.no_grad():
                hist_std_fea, label, iop = model(data, return_loss = False, return_data = True)
                fea = hist_std_fea.cpu().numpy()
                bs, dim = fea.shape
                for j in range(bs):
                    this_fea = fea[j].tolist()
                    all_feas.append(this_fea)
                    all_iops.append(float(iop.cpu()[j]))

    lines = []
    print ('in total we have ' + str(len(all_iops)) + ' proposals')
    for fea, iop in zip(all_feas, all_iops):
        f_str = [str(f) for f in fea ]
        f_str.append(str(iop))
        line = ' '.join(f_str)
        line += '\n'
        lines.append(line)

    f = open(output_path, 'w')
    f.writelines(lines)
    f.close()


def get_cluster_det(model, cfg, logger):
    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.test_data, k, v)
        setattr(cfg.train_data, k, v)
    ### get eval data
    setattr(cfg.test_data, 'phase', 'test')
    setattr(cfg.train_data, 'phase', 'train')
    print ('get training data')
    output_path = '/ssd/sv_dsgcn/data/histgram_std_train.list'
    get_data(model, output_path, cfg, is_test = False, num_runs = 10)

    print ('get eval data')
    output_path = '/ssd/sv_dsgcn/data/histgram_std_eval.list'
    get_data(model, output_path, cfg, is_test = True, num_runs = 1)


