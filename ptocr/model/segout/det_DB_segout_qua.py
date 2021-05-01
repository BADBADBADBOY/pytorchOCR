#-*- coding:utf-8 _*-
"""
@author:fxw
@file: det_DB_segout.py
@time: 2020/08/07
"""
from collections import OrderedDict
import torch
import torch.nn as nn
from ..CommonFunction_Q import ConvBnRelu,ConvBn
from torch.quantization import QuantStub, DeQuantStub, fuse_modules

class SegDetector(nn.Module):
    def __init__(self,inner_channels=256,bias=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(SegDetector, self).__init__()
        self.binarize = nn.Sequential(
            ConvBnRelu(inner_channels, inner_channels // 4, 3,1, padding=1,groups=1,bias=bias),
            ConvBn(inner_channels // 4, 1, 1,1,0,groups=1),
            nn.Upsample(scale_factor=4),
            nn.Sigmoid()
            )
 
        self.thresh = nn.Sequential(
            ConvBnRelu(inner_channels, inner_channels //4, 3, stride=1,padding=1, groups=1,bias=bias),
            ConvBn(inner_channels // 4, 1,1,1,0,groups=1,bias=bias),
            nn.Upsample(scale_factor=4),
            nn.Sigmoid()
            )
        self.binarize.apply(self.weights_init)   
        self.fuse_model()
        
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
            
    def forward(self, fuse):
        binary = self.binarize(fuse)
        thresh = self.thresh(fuse)
        return thresh,binary

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBnRelu:
                fuse_modules(m, ['conv', 'bn', 'relu'], inplace=True)
            if type(m) == ConvBn:
                fuse_modules(m, ['conv', 'bn'], inplace=True)

