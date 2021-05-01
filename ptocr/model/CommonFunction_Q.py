#-*- coding:utf-8 _*-
"""
@author:fxw
@file: CommonFunction_Q.py
@time: 2020/11/02
"""
import torch.nn as  nn

class ConvBnRelu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,groups,bias=False):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        """
        super(ConvBnRelu,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,groups=groups,bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,groups,bias=False):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param bias:
        """
        super(ConvBn,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,groups=groups,bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x