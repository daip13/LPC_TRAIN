from __future__ import division

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import DistSamplerSeedHook, obj_from_dict, load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from dsgcn.datasets import build_dataset_mall, build_processor, build_dataloader
from dsgcn.runner import Runner


def parse_losses(loss):
    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    return log_vars


def batch_processor(model, data, train_mode):
    _, loss = model(data, return_loss=True) 
    loss = torch.mean(loss)
    log_vars = parse_losses(loss)
    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data[-1]))

    return outputs

def train_cluster_mall(model, cfg, logger):
    # prepare data loaders
    for k, v in cfg.model1['kwargs'].items():
        setattr(cfg.train_data, k, v)
        setattr(cfg.test_data, k, v)
    for k, v in cfg.model2['kwargs'].items():
        setattr(cfg.train_data, k, v)
        setattr(cfg.test_data, k, v)
    setattr(cfg.train_data, 'phase', 'train')
    setattr(cfg.test_data, 'phase', 'test')
    dataset2 = build_dataset_mall(cfg.test_data)
    dataset1 = build_dataset_mall(cfg.train_data)
    processor = build_processor(cfg.stage)
    # train
    if cfg.distributed:
        _dist_train(model, data_loaders, cfg)
    else:
        _single_train(model, dataset1, dataset2, processor, cfg, logger)


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    assert paramwise_options is None
    return obj_from_dict(
        optimizer_cfg, torch.optim, dict(params=model.parameters()))


def _dist_train(model, data_loaders, cfg):
    # put model on gpus
    #model = MMDistributedDataParallel(model.cuda())
    model = MMDataParallel(model.cuda())
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)
    # register hooks
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _single_train(model, dataset1, dataset2, processor, cfg, logger):
    model = [MMDataParallel(item, device_ids=range(cfg.gpus)).cuda() for item in model]
    optimizer = [build_optimizer(item, cfg.optimizer) for item in model]

    epoch_num = 0
    model1, model2, model3 = model[0], model[1], model[2]
    optimizer1, optimizer2, optimizer3 = optimizer[0], optimizer[1], optimizer[2]
    data_loaders2 = build_dataloader(dataset2, processor, cfg.batch_size_per_gpu, cfg.workers_per_gpu, train=False, shuffle=False)
    data_loaders1 = build_dataloader(dataset1, processor, cfg.batch_size_per_gpu, cfg.workers_per_gpu, train=True, shuffle=True)
    lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=30)
    lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=30)
    lr_scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=30)
    while epoch_num < cfg.total_epochs:
        lr_scheduler1.step()
        lr_scheduler2.step()
        lr_scheduler3.step()
        for i, data_batch in enumerate(data_loaders1):
            model1.train()
            model2.train()
            model3.train()
            model1.zero_grad()
            model2.zero_grad()
            model3.zero_grad()
            x1, loss1 = model1(data_batch, return_loss=True)
            loss1 = torch.mean(loss1)
            x2, loss2 = model2(data_batch, return_loss=True)
            loss2 = torch.mean(loss2)
            x3, loss3 = model3(data_batch, return_loss=True)
            loss3 = torch.mean(loss3)
            if i % cfg.log_config.interval == 0:
                logger.info('[Epoch] {}, lr {}'.format(epoch_num, lr_scheduler1.get_lr()[0]))
                logger.info('[Train] [Conv3d] Iter {}/{}: Loss {:.4f}'.format(i, len(data_loaders1), loss1))
                logger.info('[Train] [dsgcn] Iter {}/{}: Loss {:.4f}'.format(i, len(data_loaders1), loss2))
                logger.info('[Train] [histgram_std] Iter {}/{}: Loss {:.4f}'.format(i, len(data_loaders1), loss3))
                with open(cfg.work_dir + "/cnn_model_iter_" + str(epoch_num) + "_" + str(i)  + ".pth", 'wb') as to_save1:
                    torch.save(model1.module, to_save1)
                with open(cfg.work_dir + "/dsgcn_model_iter_" + str(epoch_num) + "_" + str(i)  + ".pth", 'wb') as to_save2:
                    torch.save(model2.module, to_save2)
                with open(cfg.work_dir + "/histstd_model_iter_" + str(epoch_num) + "_" + str(i)  + ".pth", 'wb') as to_save3:
                    torch.save(model3.module, to_save3)
    
            loss12 = F.kl_div(F.log_softmax(x1, dim=1), x2.detach(), False)/x1.shape[0]
            loss13 = F.kl_div(F.log_softmax(x1, dim=1), x3.detach(), False)/x1.shape[0]
            loss1 = loss1 + loss12 + loss13
            loss1.backward()
            optimizer1.step()
            loss21 = F.kl_div(F.log_softmax(x2, dim=1), x1.detach(), False)/x2.shape[0]
            loss23 = F.kl_div(F.log_softmax(x2, dim=1), x3.detach(), False)/x2.shape[0]
            loss2 = loss2 + loss23
            loss2.backward()
            optimizer2.step()
            loss31 = F.kl_div(F.log_softmax(x3, dim=1), x1.detach(), False)/x3.shape[0]
            loss32 = F.kl_div(F.log_softmax(x3, dim=1), x2.detach(), False)/x3.shape[0]
            loss3 = loss3 + loss32
            loss3.backward()
            optimizer3.step()
         
        # save the model
        with open(cfg.work_dir + "/cnn_model_iter_" + str(epoch_num) + ".pth", 'wb') as to_save1:
            torch.save(model1.module, to_save1)
        with open(cfg.work_dir + "/dsgcn_model_iter_" + str(epoch_num) +".pth", 'wb') as to_save2:
            torch.save(model2.module, to_save2)
        with open(cfg.work_dir + "/histstd_model_iter_" + str(epoch_num) +".pth", 'wb') as to_save3:
            torch.save(model3.module, to_save3)

        if epoch_num % 2 == 0:
            logger.info('[Evaluation] [Epoch] {}'.format(epoch_num))
            model1.eval()
            model2.eval()
            model3.eval()
            for i, data_batch in enumerate(data_loaders2):
                with torch.no_grad():
                    _ = model1(data_batch, return_loss=False)
                    _ = model2(data_batch, return_loss=False)
                    _ = model2(data_batch, return_loss=False)
        epoch_num += 1
