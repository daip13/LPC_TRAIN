#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    author: RenliangWeng
    date: 20190820
'''

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class HistgramStdBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, freeze_bn, dropout=0.0):
        super(HistgramStdBasicBlock, self).__init__()
        self.fc =  nn.Linear(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.freeze_bn = freeze_bn
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, x, D=None):
        x = self.fc(x)
        if self.freeze_bn:
            self.bn.eval()
        x = self.bn(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class HistgramStd(nn.Module):
    """ input is (bs, N, D), for featureless, D=1
        dev output is (bs, num_classes)
        seg output is (bs, N, num_classes)
    """
    def __init__(self, feature_dim, planes, loss_type='BCE',
            num_hist_bins = 0, num_classes=2, dropout=0.0, freeze_bn = False):
        super(HistgramStd, self).__init__()
        print ('num_hist_bins = ' + str(num_hist_bins))
        print ('loss_type = ' + loss_type)
        assert feature_dim > 0
        assert dropout >= 0 and dropout < 1
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.loss_type = loss_type
        if loss_type == 'mse':
            self.loss = torch.nn.MSELoss()
            self.classifier = nn.Linear(planes[-1], 1)
        else:
            self.loss = torch.nn.CrossEntropyLoss()
            self.classifier = nn.Linear(planes[-1], 2)

        self.num_hist_bins = num_hist_bins
        assert self.num_hist_bins >= 0
        if self.num_hist_bins > 0:
            self._get_bin_info()

        self.inplanes = feature_dim + num_hist_bins
        self.layers = self._make_layer(HistgramStdBasicBlock, planes, freeze_bn, dropout)
        #self.sigmoid = nn.Sigmoid()
        self.num_classes = num_classes
        #self.loss = torch.nn.BCELoss()
        self.softmax = torch.nn.Softmax(dim = 1)

    
    @staticmethod
    def eval_batch_new(y, label, loss_type):
        idx = 0
        acc_times = 0.0
        acc_pos_times = 0
        acc_neg_times = 0
        bs = len(y)
        bs_pos = len((label==1).nonzero())
        bs_neg = len((label==0).nonzero())
        if loss_type != 'mse':
            tp_mean_prob = 0
            tp_times = 0
            tn_mean_prob = 0
            tn_times = 0
            fp = 0.0
            fn = 0.0
            fp_iop = 0
            for prob in y:
                this_label = label[idx]
                if prob.cpu() > 0.5 and this_label.cpu() == 1:
                    acc_times += 1
                    acc_pos_times += 1
                    tp_times += 1
                    tp_mean_prob += prob.cpu()
                elif prob.cpu() < 0.5 and this_label.cpu() == 0:
                    acc_times += 1
                    acc_neg_times += 1
                    tn_times += 1
                    tn_mean_prob += prob.cpu()
                elif this_label.cpu() == 1:
                    fn += 1
                elif this_label.cpu() == 0:
                    fp += 1
                    fp_iop += prob.cpu()
                idx += 1
            acc = acc_times / bs
            if bs_pos > 0:
                acc_pos = acc_pos_times / bs_pos
            else:
                acc_pos = 0
            if bs_neg > 0:
                acc_neg = acc_neg_times / bs_neg
            else:
                acc_neg = 0
            if fp != 0:
                fp_iop /= fp
            else:
                fp_iop = 0
            fp = fp / bs
            fn = fn / bs
            if tp_times != 0:
                tp_mean_prob /= tp_times
            else:
                tp_mean_prob = 0
            if tn_times != 0:
                tn_mean_prob /= tn_times
            else:
                tn_mean_prob = 1
            print ('acc = ' + str(acc) + ', acc_pos = ' + str(acc_pos) + ', acc_neg = ' + str(acc_neg) +', fp = ' + str(fp) + ', fn = ' + str(fn) + ', fp_iop = ' + str(fp_iop) + ', tn_mean_prob = ' + str(tn_mean_prob) + ', tp_mean_prob = ' + str(tp_mean_prob))



    def forward(self, data, return_loss=False, return_data = False):
        epsilon = 0.0001
        feature = data[0].cpu().numpy() ### feature matrix after padding
        label = data[-1] ### IOP label, could be binary label or float label ranging from 0.0 to 1.0
        bs = len(label)

        hist_std_fea = self._get_std_feature(feature)
        hist_std_fea = torch.from_numpy(hist_std_fea).cuda()
        hist_std_fea.detach_()
        if return_data:
            return hist_std_fea, label

        x = hist_std_fea
        for layer in self.layers:
            x = layer(x)
        if self.dropout is not None:
            x = self.dropout(x)
        feature = x
        x = self.classifier(x)
        if self.loss_type != 'mse':
            y = self.softmax(x)
            y1 = y[:, 1]
        else:
            y = x
        self.eval_batch_new(y1, label, self.loss_type)

        if return_loss :
            purity_label_test = label.cpu().numpy()
            try:
                assert sum(purity_label_test == 1) and sum(purity_label_test == 0)
            except:
                a = 1
                #print('warning!!! label error {}, {} vs {}'.format(purity_label_test, sum(purity_label_test == 1), sum(purity_label_test == 0)))
           
            if self.loss_type != 'mse':
                loss = self.loss(x.view(bs, -1), data[-1].long())
            else:
                loss = self.loss(x.view(-1), data[-1].float())
            return y, loss
        else:
            return x, feature


    def _make_layer(self, block, planes, freeze_bn, dropout=0.0):
        layers = nn.ModuleList([])
        for i, plane in enumerate(planes):
            layers.append(block(self.inplanes, plane, freeze_bn, dropout))
            self.inplanes = plane
        return layers

    def _get_std_feature(self, batch_feature):
        bs, N, D = batch_feature.shape
        ### we need to get the non-padded batch_feature first
        batch_std_fea = np.zeros((bs, D), np.float32)
        for idx in range(bs):
            ### for each proposal
            proposal_fea = batch_feature[idx]
            sum_fea = np.sum(proposal_fea, 1)
            valid_idx = (sum_fea != 0).nonzero()
            proposal_fea = proposal_fea[valid_idx, :]
            proposal_fea = proposal_fea.reshape(int(valid_idx[0].size), D)
            mean_fea = np.mean(proposal_fea, 0)
            proposal_fea -= mean_fea
            # add the first order features
            proposal_fea1 = np.abs(proposal_fea)
            abs_fea = np.mean(proposal_fea1, 0)
            # add the second order features
            proposal_fea = np.square(proposal_fea)
            std_fea = np.mean(proposal_fea, 0)
            std_fea = np.sqrt(std_fea)
            batch_std_fea[idx] = std_fea
            #batch_std_fea[idx] = np.concatenate((abs_fea, std_fea))
        return batch_std_fea


    def _normalize_hist(self, hist):
        hist = np.asarray(hist)
        hist_sum = np.sum(hist)
        if hist_sum < 0.0001:
            ### to avoid numeric issue
            hist_sum = 0.0001
            #print ('sth is wrong with this data')
        hist /= hist_sum
        return hist

    def _get_hist_std_feature(self, adj, batch_feature):
        batch_std_fea = self._get_std_feature(batch_feature)
        if self.num_hist_bins > 0:
            hist_fea = self._get_hist(adj)
            output_fea = np.concatenate((hist_fea, batch_std_fea), axis=1)
            return output_fea
        else:
            return batch_std_fea


    def _get_hist(self, adj):
        bs, N, N = adj.shape
        ### adj is the padded adjecent matrix
        ### return the normalized histgram
        assert self.num_hist_bins > 0
        hist_fea = np.zeros((bs, self.num_hist_bins), np.float32)
        for idx in range(bs):
            hist = [0.0] * self.num_hist_bins
            epsilon = 0.0001
            ### get the mask
            upper_tri_idxes = np.triu_indices(N, 1) ## get the upper triangular matrix indexes excluding the diagnal ones 
            sims = adj[idx][upper_tri_idxes]
            sims = sims[np.where(np.logical_or(sims > epsilon, sims < -epsilon))]
            for sim in sims:
                if sim <= 0:
                    #print('found an negative sim item in adj matrix ' + str(sim))
                    bin_idx = 0
                else:
                    bin_idx = min(self.num_hist_bins - 1, int(sim / self.bin_size))
                hist[bin_idx] += 1
            hist = self._normalize_hist(hist)
            hist_fea[idx] = hist
        return hist_fea

    def _get_bin_info(self):
        self.bin_size = 1.0 / self.num_hist_bins
        self.bin_range = []
        for bin_idx in range(self.num_hist_bins):
            bin_start = bin_idx * self.bin_size 
            bin_end = bin_start + self.bin_size 
            self.bin_range.append([bin_start, bin_end])

def histgramstd(feature_dim, hidden_dims=[], \
        loss_type='mse', dropout=0.5, num_classes=2, freeze_bn=False, num_hist_bins = 0):
    return HistgramStd(num_hist_bins = num_hist_bins, 
                loss_type = loss_type,
                feature_dim=feature_dim,
                planes=hidden_dims,
                num_classes=num_classes, 
                dropout=dropout,
                freeze_bn=freeze_bn)
