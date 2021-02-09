import os
from struct import Struct
import glob
import json
import numpy as np
import sys
import random
import time

class Timer():
     def __init__(self, name='task', verbose=True):
        self.name = name
        self.verbose = verbose

     def __enter__(self):
        self.start = time.time()
        return self

     def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            print('[Time] {} consumes {:.4f} s'.format(self.name, time.time() - self.start))
        return exc_type is None


def l2norm(vec):
    vec /= np.linalg.norm(vec, axis=1).reshape(-1, 1)
    return vec


class ClusterDataset_mall(object):
    def __init__(self, cfg, is_test=False):
        dataset_paths = cfg['proposal_path']
        self.phase = cfg['phase']
        self._read(dataset_paths)
        print('#cluster: {}'.format(self.size))
   

    def convert_keys_to_int(self, json_res):
        new_json_res = {}
        for key in json_res:
            key_res = json_res[key]
            key = int(key)
            new_json_res[key] = key_res
        return new_json_res

    def unpack_records(self, format, data):
        record_struct = Struct(format)
        return (record_struct.unpack_from(data, offset)
                for offset in range(0, len(data), record_struct.size))


    def _read(self, dataset_paths):
        for root, dirs, files in os.walk(dataset_paths[0]):
            dataset_all1 = sorted(dirs)
            break
        if self.phase == 'train':
            dataset_all = dataset_all1
        elif self.phase == 'test':
            dataset_all = dataset_all1
        IoP_distribution, primary_indices, secondary_indices = [], [], []
        fn_node_pattern = '*.json'
        with Timer('read proposal list'):
            self.lst = []
            for dataset_name in dataset_all:
                GT_label = {}
                proposal_folder1 = os.path.join(root, dataset_name + "/pure/")
                proposal_folder2 = os.path.join(root, dataset_name + "/impure/")
                proposal_folders_all = [proposal_folder2, proposal_folder1]
                for proposal_folder in proposal_folders_all:
                    print('[{}] read proposals from folder: {}'.format(self.phase, proposal_folder))
                    fn_nodes = sorted(glob.glob(os.path.join(proposal_folder, fn_node_pattern)))
                    #assert len(fn_nodes) > 0, 'files under {} is 0'.format(proposal_folder)
                    num_nodes = 0
                    for fn_node in fn_nodes:
                        node_name = fn_node.split('/')[-1].split('_')[0]
                        if fn_node.split('/')[-2] == 'impure' or fn_node.split('/')[-2] == "0":
                            secondary_indices.append(len(self.lst))
                        else:
                            primary_indices.append(len(self.lst))

                        GT_label[node_name]=1
                        if True:
                            self.lst.append(fn_node)
                            num_nodes += 1
                    if num_nodes > 5:
                        IoP_distribution.append(num_nodes)
            self.size = len(self.lst)
            self.IoP_distribution = IoP_distribution
            self.primary_indices = primary_indices
            self.secondary_indices = secondary_indices
            print("Stats: pure proposals {}, impure proposals {}".format(len(primary_indices), len(secondary_indices)))
    
    def __len__(self):
        return self.size

