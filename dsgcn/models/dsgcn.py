#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .histgram_std import HistgramStd

__all__ = ['dsgcn']

'''
Original implementation can be referred to:
    - GCN: https://github.com/tkipf/pygcn
    - SGC: https://github.com/Tiiiger/SGC
'''

def normalize(adj):
    ### TODO  ###
    bs, N, D = adj.shape
    for ii in range(bs):
        sum1 = torch.sum(adj[ii,:,:], dim=1)
        adj[ii,:,:] /= sum1
    return adj

class GraphConv(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, D=None):
        if x.dim() == 3:
            xw = torch.matmul(x, self.weight)
            output = torch.bmm(adj, xw)
        elif x.dim() == 2:
            xw = torch.mm(x, self.weight)
            output = torch.spmm(adj, xw)
        if D is not None:
            output = output * 1. / D
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, freeze_bn, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.gc = GraphConv(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.freeze_bn = freeze_bn
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, x, adj, index, D=None):
        x = self.gc(x, adj, D)
        bs, N, D = x.shape
        #print(str(bs))
        x = x.view(-1, D)
        x1 = x.sum(1)
        index1 = (x1!=0).nonzero()
        index1 = index1.view(-1)
        if self.freeze_bn:
            self.bn.eval()
        x_new = x[index1]
        #x_new = self.bn(x_new)
        x_new = self.relu(x_new)
        #if self.dropout is not None:
         #   x_new = self.dropout(x_new)
        x[index1] = x_new
        #x = x.view(bs, N, D)
        #print(self.bn.running_mean.mean())
        x = x.view(bs, N, D)
        return x

class GNN(nn.Module):
    """ input is (bs, N, D), for featureless, D=1
        dev output is (bs, num_classes)
        seg output is (bs, N, num_classes)
    """

    def __init__(self, planes, feature_dim, featureless,
            num_classes=1, dropout=0.0, reduce_method='max', stage='dev'):
        assert feature_dim > 0
        assert dropout >= 0 and dropout < 1
        if featureless:
            self.inplanes = 1
        else:
            self.inplanes = feature_dim
        self.num_classes = num_classes
        self.reduce_method = reduce_method
        super(GNN, self).__init__()
        if stage == 'dev':
            #self.loss = torch.nn.MSELoss()
            self.loss = torch.nn.CrossEntropyLoss()
            #self.loss = torch.nn.BCELoss()
        elif stage == 'seg':
            self.loss = torch.nn.NLLLoss()
        else:
            raise KeyError('Unknown stage: {}'.format(stage))

    def pool(self, x):
        # use global op to reduce N
        # make sure isomorphic graphs output the same representation
        if self.reduce_method == 'sum':
            return torch.sum(x, dim=1)
        elif self.reduce_method == 'mean':
            return torch.mean(x, dim=1)
        elif self.reduce_method == 'max':
            return torch.max(x, dim=1)[0]
        elif self.reduce_method == 'no_pool':
            return x # wo global pooling
        else:
            raise KeyError('Unkown reduce method', self.reduce_method)

    def forward(self, data, return_loss=False):
        purity_label = data[-1]
        x, feature = self.extract(data[0], data[1], data[2])
        y = self.softmax(x)
        y1 = y[:, 1]
        #HistgramStd.eval_batch_new(y1, data[-1], 'BCE')
        if return_loss:
            purity_label_test = purity_label.cpu().numpy()
            loss = self.loss(x.view(len(data[-1]),-1), data[-1].long())
            return y, loss
        else:
            return x, feature


class GCN(GNN):

    def __init__(self, planes, feature_dim, featureless,
            num_classes=1, freeze_bn=True, dropout=0.0, reduce_method='max', stage='dev'):
        super().__init__(
                planes, feature_dim, featureless,
                num_classes, dropout, reduce_method, stage)

        self.layers = self._make_layer(BasicBlock, planes, freeze_bn, dropout)
        self.classifier = nn.Linear(self.inplanes, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim = 1)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def _make_layer(self, block, planes, freeze_bn, dropout=0.0):
        layers = nn.ModuleList([])
        for i, plane in enumerate(planes):
            layers.append(block(self.inplanes, plane, freeze_bn, dropout))
            self.inplanes = plane
        return layers

    def extract(self, x, adj, index):
        ### here we assume adj is unnormalized ###
        ### normalize adj ###
        adj = normalize(adj)
        bs = x.size(0)
        adj.detach_()
        D = adj.sum(dim=2, keepdim=True)
        D.detach_()
        assert (D > 0).all(), print("D should larger than 0, otherwise gradient will be NaN. adj is {}".format(adj.tolist()))
        for layer in self.layers:
            x = layer(x, adj, index, D)
        x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(-1, self.inplanes)
        feature = x
        x = self.classifier(x)
        #x = self.sigmoid(x)
        if self.reduce_method == 'no_pool':
            if self.num_classes > 1:
                x = x.view(bs, -1, self.num_classes)
                x = torch.transpose(x, 1, 2).contiguous()
                x = F.log_softmax(x, dim=1)
            else:
                x = x.view(bs, -1)
        return x, feature


class SGC(GNN):

    def __init__(self, planes, feature_dim, featureless,
            num_classes=1, dropout=0.0, reduce_method='max', stage='dev'):
        super().__init__(
                planes, feature_dim, featureless,
                num_classes, dropout, reduce_method, stage)

        assert stage == 'dev'
        self.degree = len(planes)
        self.classifier = nn.Linear(self.inplanes, num_classes)

    def extract(self, x, adj):
        adj.detach_()
        D = adj.sum(dim=2, keepdim=True)
        D.detach_()
        assert (D > 0).all(), "D should larger than 0, otherwise gradient will be NaN."
        for _ in range(self.degree):
            if x.dim() == 3:
                x = torch.bmm(adj, x) / D
            elif x.dim() == 2:
                x = torch.spmm(adj, x) / D
        x = self.pool(x)
        x = self.classifier(x)
        return x


def _build_model(model_type):
    __model_type__ = {
        'gcn': GCN,
        'sgc': SGC,
    }
    if model_type not in __model_type__:
        raise KeyError("Unknown model_type:", model_type)
    return __model_type__[model_type]

def dsgcn(feature_dim, hidden_dims=[], featureless=True, \
        gcn_type='gcn', reduce_method='max', dropout=0.5, num_classes=2, freeze_bn=False):
    model = _build_model(gcn_type)
    return model(planes=hidden_dims,
                 feature_dim=feature_dim,
                 featureless=featureless,
                 reduce_method=reduce_method,
                 dropout=dropout,
                 num_classes=num_classes, freeze_bn=freeze_bn)
