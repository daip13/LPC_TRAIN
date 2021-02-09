import math
import itertools
import random
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler as _DistributedSampler


__all__ = ["DistributedSampler", "DistributedSequentialSampler"]

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)

class TwoStreamBatchSampler(Sampler):
    """
    Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, batch_size=256):
        super().__init__(dataset)
        self.primary_indices = dataset.primary_indices
        self.secondary_indices = dataset.secondary_indices
        self.secondary_batch_size = int(batch_size/2)
        self.primary_batch_size = int(batch_size/2)
        self.lst = dataset.lst
    
    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        indices_output = []
        for (primary_batch, secondary_batch) in zip(grouper(primary_iter, self.primary_batch_size), grouper(secondary_iter, self.secondary_batch_size)):
            indices_output += primary_batch + secondary_batch
            for tt in secondary_batch:
                tt_name = self.lst[tt]
                assert 'impure' in tt_name.split('/')[-2] or '0' in tt_name.split('/')[-2]
        return iter(indices_output)
    def __len__(self):
        #print('length is {}'.format(len(self.primary_indices) // self.primary_batch_size))
        return (len(self.primary_indices) // self.primary_batch_size) * self.primary_batch_size * 2


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, batch_size=32):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.batch_size = batch_size
    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        total_size = (len(indices)//self.batch_size + 1)*self.batch_size
        assert total_size >= len(indices)
        indices += indices[:(total_size - len(indices))]

        # subsample
        indices = indices[self.rank:total_size:self.num_replicas]
        #assert len(indices) == self.num_samples

        return iter(indices)

class DistributedSequentialSampler(Sampler):
    def __init__(self, dataset, world_size, rank, batch_size):
        assert rank >= 0
        assert dataset.size >= world_size, '{} vs {}'.format(dataset.size, world_size)
        sub_num = int(math.ceil(1. * dataset.size / world_size))
        # add extra samples to make it evenly divisible
        tot_num = sub_num * world_size
        self.beg = sub_num * rank
        self.end = min(self.beg + sub_num, tot_num)
        self.batch_size = batch_size
    def __iter__(self):
        indices = list(range(self.beg, self.end))
        #total_size = (len(indices)//self.batch_size + 1)*self.batch_size
        #assert total_size >= len(indices)
        #indices += indices[:(total_size - len(indices))]
        #indices = indices[self.rank:total_size:self.num_replicas]
        #indices = list(range(self.end-1, self.beg-1, -1))
        return iter(indices)

    def __len__(self):
        return self.end - self.beg
