#-*- coding:utf-8 _*-
"""
@author:fxw
@file: DB_Head_Qua.py
@time: 2020/11/02
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..CommonFunction_Q import ConvBnRelu
from torch.quantization import QuantStub, DeQuantStub, fuse_modules

def upsample(x, y):
    _, _, H, W = y.size()
    # return F.upsample(x, size=(H // scale, W // scale), mode='nearest')
    return F.interpolate(x, size=(H , W), mode='nearest')

def upsample_add(x, y,skip):
    _, _, H, W = y.size()
    # return F.upsample(x, size=(H, W), mode='nearest') + y
    return skip.add(F.interpolate(x, size=(H, W), mode='nearest'),y)

class DB_Head(nn.Module):
    def __init__(self, in_channels, inner_channels, bias=False):
        """
        :param in_channels:
        :param inner_channels:
        :param bias:
        """
        super(DB_Head, self).__init__()
        self.skip = nn.quantized.FloatFunctional()

        self.in5 = ConvBnRelu(in_channels[-1], inner_channels, 1, 1, 0,groups=1, bias=bias)
        self.in4 = ConvBnRelu(in_channels[-2], inner_channels, 1, 1, 0,groups=1, bias=bias)
        self.in3 = ConvBnRelu(in_channels[-3], inner_channels, 1, 1, 0, groups=1,bias=bias)
        self.in2 = ConvBnRelu(in_channels[-4], inner_channels, 1, 1, 0,groups=1, bias=bias)

        self.out5 = ConvBnRelu(inner_channels, inner_channels // 4, 3, 1, 1, groups=1,bias=bias)
        self.out4 = ConvBnRelu(inner_channels, inner_channels // 4, 3, 1, 1,groups=1, bias=bias)
        self.out3 = ConvBnRelu(inner_channels, inner_channels // 4, 3, 1, 1, groups=1,bias=bias)
        self.out2 = ConvBnRelu(inner_channels, inner_channels // 4, 3, 1, 1, groups=1,bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
        self.fuse_model()

    def forward(self, x1,x2,x3,x4):
        in5 = self.in5(x1)
        in4 = self.in4(x2)
        in3 = self.in3(x3)
        in2 = self.in2(x4)

        out4 = upsample_add(in5,in4,self.skip) # 1/16
        out3 = upsample_add(out4,in3,self.skip) # 1/8
        out2 = upsample_add(out3,in2,self.skip)  # 1/4

        p5 = upsample(self.out5(in5),out2)
        p4 = upsample(self.out4(out4),out2)
        p3 = upsample(self.out3(out3),out2)
        p2 = self.out2(out2)
        fuse = self.skip.cat((p5, p4, p3, p2), 1)
        return fuse

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBnRelu:
                fuse_modules(m, ['conv', 'bn', 'relu'], inplace=True)

