from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt 
from dsgcn.models.histgram_std import HistgramStd
import json
import os
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from dsgcn.datasets import build_dataset_mall, build_processor, build_dataloader


def test_cluster_mall(model1, cfg, logger):
    model = torch.load(cfg.load_from1)

    for k, v in cfg.model1['kwargs'].items():
        setattr(cfg.test_data, k, v)
    for k, v in cfg.model2['kwargs'].items():
        setattr(cfg.test_data, k, v)
    setattr(cfg.test_data, 'phase', 'test')
    dataset = build_dataset_mall(cfg.test_data)
    processor = build_processor(cfg.stage)

    losses = []
    output_probs = []
    IoP_GT = []
    IoP_binary_GT = []
    num_impure_pro = 0
    if cfg.gpus == 1:
        data_loader = build_dataloader(
                dataset,
                processor,
                cfg.batch_size_per_gpu,
                cfg.workers_per_gpu,
                train=False)

        model = MMDataParallel(model, device_ids=range(cfg.gpus))
        if cfg.cuda:
            model.cuda()
        output_IoP_loss = []
        model.eval()
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                output, loss = model(data, return_loss=True)
                losses += [loss.item()]
                num_impure_pro += (data[-1]==0).nonzero().shape[0]
                if i % cfg.log_config.interval == 0:
                    logger.info('[Test] Iter {}/{}: Loss {:.4f}'.format(i, len(data_loader), loss))
                if cfg.save_output:
                    output = output[:,1]
                    output = output.view(-1)
                    output_probs.append(output.tolist())
                    IoP_GT.append(data[-1].tolist())
    else:
        raise NotImplementedError
    output_probs1 = [iop for item in output_probs for iop in item]
    output_probs = np.array([iop for item in output_probs for iop in item])
    IoP_GT0 = [iop for item in IoP_GT for iop in item]
    IoP_GT = np.array([iop for item in IoP_GT for iop in item])
    output_probs = torch.from_numpy(output_probs)
    IoP_GT1 = torch.from_numpy(IoP_GT)
    #HistgramStd.eval_batch_new(output_probs, IoP_GT1, 'BCE')
    output_probs2 = np.array(output_probs1)
    # plot roc curve
    false_positive_rate,true_positive_rate,thresholds=roc_curve(IoP_GT, output_probs2)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.draw()
    plt.savefig(cfg.work_dir + '/ROC.jpg')
    plt.close()

    # plot IoP distribution curve
    pos01 = np.where( (IoP_GT1==0))
    iop_01 = output_probs2[pos01]
    pos02 = np.where( (IoP_GT1==1))
    iop_02 = output_probs2[pos02]
    if cfg.save_output:
        plt.figure(1)
        plt.subplot(1, 1, 1)
        plt.boxplot([iop_01.tolist(), iop_02.tolist()], notch=True)

        x_tricks = np.array([1, 2])
        plt.xticks(x_tricks)
        plt.grid(axis='y')
        plt.draw()
        plt.savefig(cfg.work_dir + '/Estimated_IoP.jpg')
        plt.close()
    
    estimated_iop_dict = {}
    for i, node in enumerate(dataset.lst):
        node_name = node.split('/')[-1]
        estimated_iop = output_probs1[i] 
        estimated_iop_dict[node_name] = estimated_iop
    with open(cfg.work_dir + '/Estimated_IoP_eval_dict.json', 'w') as f:
        json.dump(estimated_iop_dict, f)
    with open(cfg.work_dir + '/Estimated_IoP_eval.json', 'w') as f:
        json.dump(output_probs1, f)
    with open(cfg.work_dir + '/GT_IoP_eval.json', 'w') as f:
        json.dump(IoP_GT0, f)

