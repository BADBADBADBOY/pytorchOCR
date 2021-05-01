#-*- coding:utf-8 _*-
"""
@author:fxw
@file: CommonFunction.py
@time: 2020/08/07
"""
import torch.nn.functional as F
import torch.nn as nn

class DeConvBnRelu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=4,stride=2,with_relu=False,padding=1,bias=False):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        """
        super(DeConvBnRelu,self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding,bias=bias)  # Reduce channels
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.with_relu = with_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x

class ConvBnRelu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,with_relu=True,bias=False):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        """
        super(ConvBnRelu,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,bias=bias) 
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.with_relu = with_relu
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x

class DWBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,bias=False):
        super(DWBlock,self).__init__()
        self.dw_conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding=kernel_size//2,groups=out_channels,bias=bias)
        self.point_conv = nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=bias)
        self.point_bn = nn.BatchNorm2d(out_channels)
        self.point_relu = nn.ReLU()

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.point_relu(self.point_bn(self.point_conv(x)))
        return x


def upsample(x, y, scale=1):
    _, _, H, W = y.size()
    # return F.upsample(x, size=(H // scale, W // scale), mode='nearest')
    return F.interpolate(x, size=(H // scale, W // scale), mode='nearest')

def upsample_add(x, y):
    _, _, H, W = y.size()
    # return F.upsample(x, size=(H, W), mode='nearest') + y
    return F.interpolate(x, size=(H, W), mode='nearest') + y