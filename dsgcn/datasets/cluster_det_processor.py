import numpy as np
import random
import time
import sys
import json
import numpy as np
from .cluster_dataset import l2norm
import torch


class ClusterDetProcessor_mall(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.dtype = np.float32

    def __len__(self):
        #print('self.dataset size is {}'.format(self.dataset.size))
        return self.dataset.size

    def build_data(self, fn_node):
        fn_node_info = json.load(open(fn_node, 'r'))
        reid_features = np.array(fn_node_info["reid_features"])
        assert reid_features.shape[1] == 2048
        reid_features_node = l2norm(reid_features)
        spatem_features = np.array(fn_node_info["spatem_features"])
        spatem_features_node = spatem_features
        # add the temporal spatial information into the feature vectors
        features_node = np.concatenate((reid_features_node, spatem_features_node), axis=1)
        label_output = fn_node_info["labels"]
        adj = np.array(fn_node_info["adjcent_matrix"])
        return features_node, adj, float(label_output)
    
    
    def __getitem__(self, idx):
        """ each vertices is a NxD matrix,
            each adj is a NxN matrix,
            each label is a Nx1 matrix,
        """
        if idx is None or idx > self.dataset.size:
            raise ValueError('idx({}) is not in the range of {}'.format(idx, self.dataset.size))
        fn_node = self.dataset.lst[idx]
        vertices1, adj, binary_label = self.build_data(fn_node)
        return vertices1.astype(self.dtype), \
               adj.astype(self.dtype), \
               np.array(binary_label, dtype=self.dtype)

