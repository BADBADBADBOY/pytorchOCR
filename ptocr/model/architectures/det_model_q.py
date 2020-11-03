# -*- coding:utf-8 _*-
"""
@author:fxw
@file: det_model.py
@time: 2020/08/07
"""
import torch
import torch.nn as nn
from .. import create_module
from torch.quantization import QuantStub, DeQuantStub

class DetModel(nn.Module):
    def __init__(self, config):
        super(DetModel, self).__init__()
        self.algorithm = config['base']['algorithm']
        self.backbone = create_module(config['backbone']['function'])(config['base']['pretrained'])
        
        self.head = create_module(config['head']['function']) \
            (config['base']['in_channels'],
             config['base']['inner_channels'])

        if (config['base']['algorithm']) == 'DB':
            self.seg_out = create_module(config['segout']['function'])(config['base']['inner_channels'])
       
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.k = config['base']['k']
        
    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))
    
    def forward(self, data):
        if self.training:
            if self.algorithm == "DB":
                img, gt, gt_mask, thresh_map, thresh_mask = data
                if torch.cuda.is_available():
                    img, gt, gt_mask, thresh_map, thresh_mask = \
                        img.cuda(), gt.cuda(), gt_mask.cuda(), thresh_map.cuda(), thresh_mask.cuda()
                gt_batch = dict(gt=gt)
                gt_batch['mask'] = gt_mask
                gt_batch['thresh_map'] = thresh_map
                gt_batch['thresh_mask'] = thresh_mask
        else:
            img = data
        img = self.quant(img)
        f_map= self.backbone(img)
        head_map = self.head(f_map[-1],f_map[-2],f_map[-3],f_map[-4])
        thresh,binary = self.seg_out(head_map)
        thresh = self.dequant(thresh)
        binary = self.dequant(binary)
        thresh_binary = self.step_function(binary,thresh)
        out = {}
        out['binary'] = binary
        out['thresh'] = thresh
        out['thresh_binary'] = thresh_binary
        if self.training:
            return out, gt_batch
        return out


class DetLoss(nn.Module):
    def __init__(self, config):
        super(DetLoss, self).__init__()
        self.algorithm = config['base']['algorithm']
        if (config['base']['algorithm']) == 'DB':
            self.loss = create_module(config['loss']['function'])(config['loss']['l1_scale'],
                                                                  config['loss']['bce_scale'])
        elif (config['base']['algorithm']) == 'PAN':
            self.loss = create_module(config['loss']['function'])(config['loss']['kernel_rate'],
                                                                  config['loss']['agg_dis_rate'])
        elif (config['base']['algorithm']) == 'PSE':
            self.loss = create_module(config['loss']['function'])(config['loss']['text_tatio'])

        elif (config['base']['algorithm']) == 'SAST':
            self.loss = create_module(config['loss']['function'])(config['loss']['tvo_lw'],
                                                                  config['loss']['tco_lw'],
                                                                  config['loss']['score_lw'],
                                                                  config['loss']['border_lw']
                                                                  )
        else:
            assert True == False, ('not support this algorithm !!!')

    def forward(self, pre_batch, gt_batch):
        return self.loss(pre_batch, gt_batch)

