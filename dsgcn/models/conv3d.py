#!/usr/bin/env python
# coding:utf8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .histgram_std import HistgramStd, HistgramStdBasicBlock, histgramstd
from .dsgcn import dsgcn 

torch.manual_seed(123456)

class Conv3d(nn.Module):
    """
        Implementation of Conv3d for track purity evaluation
    """

    def __init__(self, feature_dim_app, max_len=40, dropout=0.5):
        super(Conv3d, self).__init__()
        self.conv1=nn.Conv3d(feature_dim_app,128, kernel_size=(3,1,1),stride=1,padding=(1,0,0))
        self.relu=nn.ReLU()
        self.conv2=nn.Conv3d(128, 64,kernel_size=(3,1,1),stride=1,padding=(1,0,0))
        self.conv3=nn.Conv3d(64, 32,kernel_size=(3,1,1),stride=1,padding=(1,0,0))
        self.fc1=nn.Linear(32, 2)#TBA
        self.loss = nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim = 1)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
    def forward(self, data, return_loss=False):
        """

        :param sen_batch: (batch, seq_len, feat_dim), tensor for sequences
        :param sen_lengths:
        :return:
        """

        sen_batch1, purity_label = data[0], data[-1]
        batch_size1, seq_len1, feat_dim1 = sen_batch1.shape
        sen_batch1 = sen_batch1.permute(0, 2, 1)
        sen_batch1 = sen_batch1.view(batch_size1, feat_dim1, seq_len1, 1, 1)
        x = sen_batch1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        self.maxpool=nn.MaxPool3d(kernel_size=(seq_len1,1,1),stride=(1,1,1))
        x = self.maxpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(batch_size1, -1)
        feature = x
        x = self.fc1(x)
        y = self.softmax(x)
        y1 = y[:, 1]
        HistgramStd.eval_batch_new(y1, data[-1], 'BCE')
        if return_loss:
            purity_label_test = purity_label.cpu().numpy()
            try:
                assert sum(purity_label_test == 1) and sum(purity_label_test == 0)
            except:
                a = 1
                #print('warning!!! label error {}, {} vs {}'.format(purity_label_test, sum(purity_label_test == 1), sum(purity_label_test == 0)))
           
            loss = self.loss(x.view(len(purity_label), -1), purity_label.long())
            return y, loss
        else:
            return x, feature

