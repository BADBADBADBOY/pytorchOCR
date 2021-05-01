#-*- coding:utf-8 _*-
"""
@author:fxw
@file: det_FPNHead.py
@time: 2020/08/07
"""
import torch
import torch.nn as nn
from ptocr.model.CommonFunction import ConvBnRelu,upsample_add,upsample

class FPN_Head(nn.Module):
    def __init__(self, in_channels, inner_channels,bias=False):
        """
        :param in_channels:
        :param inner_channels:
        :param bias:
        """
        super(FPN_Head, self).__init__()
        # Top layer
        self.toplayer = ConvBnRelu(in_channels[-1], inner_channels, kernel_size=1, stride=1,padding=0,bias=bias)  # Reduce channels
        # Smooth layers
        self.smooth1 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1,bias=bias)
        self.smooth2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1,bias=bias)
        self.smooth3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1,bias=bias)
        # Lateral layers
        self.latlayer1 = ConvBnRelu(in_channels[-2], inner_channels, kernel_size=1, stride=1, padding=0,bias=bias)
        self.latlayer2 = ConvBnRelu(in_channels[-3], inner_channels, kernel_size=1, stride=1, padding=0,bias=bias)
        self.latlayer3 = ConvBnRelu(in_channels[-4], inner_channels, kernel_size=1, stride=1, padding=0,bias=bias)
        # Out map
        self.conv_out = ConvBnRelu(inner_channels * 4, inner_channels, kernel_size=3, stride=1, padding=1,bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def forward(self, x):
        c2, c3, c4, c5 = x
        ##
        p5 = self.toplayer(c5)
        c4 = self.latlayer1(c4)
        p4 = upsample_add(p5, c4)
        p4 = self.smooth1(p4)
        c3 = self.latlayer2(c3)
        p3 = upsample_add(p4, c3)
        p3 = self.smooth2(p3)
        c2 = self.latlayer3(c2)
        p2 = upsample_add(p3, c2)
        p2 = self.smooth3(p2)
        ##
        p3 = upsample(p3, p2)
        p4 = upsample(p4, p2)
        p5 = upsample(p5, p2)

        fuse = torch.cat((p2, p3, p4, p5), 1)
        fuse = self.conv_out(fuse)
        return fuse