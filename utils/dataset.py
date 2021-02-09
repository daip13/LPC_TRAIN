#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
#import cv2, time
#import numpy as np
#import yaml, glob
#import argparse
#import random
#import os
#import sys
#import json
#import base64
#import networkx as nx
#import logging
#sys.path.append('/root/utils')
#from hdfs_check import *
import sys
sys.path.append('/root/')
from dp_learn_to_cluster_master.utils.misc import TextColors, l2norm, read_probs, read_meta


class BasicDataset():
    def __init__(self, name, prefix='data', dim=256, normalize=True, verbose=True):
        self.name = name
        self.dtype=np.float32
        self.dim = dim
        self.normalize = normalize
        if not os.path.exists(prefix):
            raise FileNotFoundError('folder({}) does not exist.'.format(prefix))
        self.prefix = prefix
        self.feat_path = os.path.join(prefix, name+'.json')
        gth_res = load_json_file(self.feat_path)
        feats = np.zeros([self.dim, len(gth_res)], dtype = float)
        labels = []
        num1 = 0
        for key, value in gth_res.items():
            labels.append(int(value[0]))
            feats[:,num1] = np.array(value[2][0])
            num1 += 1
        self.features = feats
        lb2idxs = {}
        idx2lb = {}
        for idx, x in enumerate(labels):
            if x not in lb2idxs:
                lb2idxs[x] = []
            lb2idxs[x] += [idx]
            idx2lb[idx] = x
        self.lb2idxs = lb2idxs
        self.idx2lb = idx2lb
        self.inst_num = len(self.idx2lb)
        self.cls_num = len(self.lb2idxs)
        if self.normalize:
            self.features = l2norm(self.features)

    def info(self):
        print("name:{}{}{}\ninst_num:{}\ncls_num:{}\ndim:{}\nnormalization:{}{}{}\ndtype:{}".\
                format(TextColors.OKGREEN, self.name, TextColors.ENDC, self.inst_num, self.cls_num, self.dim, \
                        TextColors.FATAL, self.normalize, TextColors.ENDC, self.dtype))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="example")
    parser.add_argument("--name", type=str, default='part1_test', help="image features")
    parser.add_argument("--prefix", type=str, default='./data', help="prefix of dataset")
    parser.add_argument("--dim", type=int, default=256, help="dimension of feature")
    parser.add_argument("--no_normalize", action='store_true', help="whether to normalize feature")
    args = parser.parse_args()

    ds = BasicDataset(name=args.name, prefix=args.prefix, dim=args.dim, normalize=not args.no_normalize)
    ds.info()
