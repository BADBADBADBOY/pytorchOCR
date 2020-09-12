# -*- coding:utf-8 _*-
"""
@author:fxw
@file: det_model.py
@time: 2020/08/07
"""
import torch
import torch.nn as nn
from .. import create_module


class RecModel(nn.Module):
    def __init__(self, config):
        super(RecModel, self).__init__()
        self.algorithm = config['base']['algorithm']
        self.backbone = create_module(config['backbone']['function'])(config['base']['pretrained'],config['base']['is_gray'])
        self.head = create_module(config['head']['function'])(
                     use_conv=config['base']['use_conv'],
                     use_attention=config['base']['use_attention'],
                     use_lstm=config['base']['use_lstm'],
                     lstm_num=config['base']['lstm_num'],
                     inchannel=config['base']['inchannel'],
                     hiddenchannel=config['base']['hiddenchannel'],
                     classes=config['base']['classes'])

    def forward(self, img):
        x = self.backbone(img)
        x = self.head(x)
        return x


class RecLoss(nn.Module):
    def __init__(self, config):
        super(RecLoss, self).__init__()
        self.algorithm = config['base']['algorithm']
        if (config['base']['algorithm']) == 'CRNN':
            self.loss = create_module(config['loss']['function'])(config)
        else:
            assert True == False, ('not support this algorithm !!!')

    def forward(self, pre_batch, gt_batch):
        return self.loss(pre_batch, gt_batch)

