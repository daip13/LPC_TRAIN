import torch
import torch.nn.functional as F
import numpy as np

from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from dsgcn.datasets.sampler import DistributedSampler, DistributedSequentialSampler, TwoStreamBatchSampler
      
def collate_mot(batch):
    bs = len(batch)
    if bs > 1:
        feat, adj, lb = zip(*batch)
        #print('feat shape is: {}'.format(feat[0].shape))
        #print('adj shape is: {}'.format(adj[0].shape))
        sizes = [f.shape[0] for f in feat]
        max_size = max(sizes)
        dim = feat[0].shape[1]
        lb = torch.from_numpy(np.array(lb))
        index = []
        for i, size in enumerate(sizes):
            index1 = [i*max_size+idx for idx in range(size)]
            index += index1
        index = torch.from_numpy(np.array(index))
        # pad to [x, 0]
        pad_feat = [
                F.pad(torch.from_numpy(f),
                (0, 0, 0, max_size - s),
                value=0)
                for f, s in zip(feat, sizes)]
        # pad to [[a, 0], [0, 0]]
        pad_adj = [
                F.pad(torch.from_numpy(a),
                (0, max_size - s, 0, max_size - s),
                value=0)
                for a, s in zip(adj, sizes)]
        # pad to [[a, 0], [0, i]]
        pad_adj = [a +
                F.pad(torch.eye(max_size - s),
                (s, 0, s, 0),
                value=0)
                if s < max_size else a
                for a, s in zip(pad_adj, sizes)]
        pad_feat = default_collate(pad_feat)
        pad_adj = default_collate(pad_adj)
        return pad_feat, pad_adj, lb
    else:
        raise NotImplementedError()


def build_dataloader(dataset,
                     processor,
                     batch_size_per_gpu,
                     workers_per_gpu,
                     shuffle=False,
                     train=False,
                     **kwargs):
    rank, world_size = get_dist_info()
    batch_size = batch_size_per_gpu
    num_workers = workers_per_gpu
    if train:
        sampler = TwoStreamBatchSampler(dataset, world_size, rank, shuffle, batch_size/num_workers)
    else:
        sampler = DistributedSequentialSampler(dataset, world_size, rank, batch_size)

    data_loader = DataLoader(
        processor(dataset),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_mot,
        pin_memory=False,
        **kwargs)

    return data_loader
