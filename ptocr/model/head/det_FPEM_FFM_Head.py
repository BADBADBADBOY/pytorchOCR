#-*- coding:utf-8 _*-
"""
@author:fxw
@file: det_FPEM_FFM_Head.py
@time: 2020/08/07
"""
import torch
import torch.nn as nn
from ptocr.model.CommonFunction import DWBlock,ConvBnRelu,upsample_add,upsample

class FFM(nn.Module):
    def __init__(self,):
        super(FFM,self).__init__()

    def forward(self, x):
        map_add1,map_add2,map_add3,map_add4 = x[0]
        for i in range(1,len(x)):
            map_add1+=x[i][0]
            map_add2+=x[i][1]
            map_add3+=x[i][2]
            map_add4+=x[i][3]
        return map_add1,map_add2,map_add3,map_add4

class FPEM(nn.Module):
    def __init__(self,inner_channels):
        super(FPEM,self).__init__()
        self.up_block1 = DWBlock(inner_channels,inner_channels,3,1)
        self.up_block2 = DWBlock(inner_channels,inner_channels,3,1)
        self.up_block3 = DWBlock(inner_channels,inner_channels,3,1)

        self.down_block1 = DWBlock(inner_channels, inner_channels, 3, 2)
        self.down_block2 = DWBlock(inner_channels, inner_channels, 3, 2)
        self.down_block3 = DWBlock(inner_channels, inner_channels, 3, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def forward(self, x):
        x2, x3, x4, x5 = x
        up_add_map1 = upsample_add(x5,x4)
        up_map1 = self.up_block1(up_add_map1)

        up_add_map2 = upsample_add(up_map1,x3)
        up_map2 = self.up_block2(up_add_map2)

        up_add_map3 = upsample_add(up_map2,x2)
        up_map3 = self.up_block3(up_add_map3)

        down_add_map1 = upsample_add(up_map2,up_map3)
        down_map1 = self.down_block1(down_add_map1)

        down_add_map2= upsample_add(up_map1, up_map2)
        down_map2 = self.down_block2(down_add_map2)

        down_add_map3 = upsample_add(x5, up_map1)
        down_map3 = self.down_block3(down_add_map3)

        return up_map3,down_map1,down_map2,down_map3

class FPEM_FFM_Head(nn.Module):
    def __init__(self, in_channels, inner_channels,FPEM_NUM=2,bias=False):
        super(FPEM_FFM_Head, self).__init__()
        self.smooth1 = ConvBnRelu(in_channels=in_channels[3], out_channels=inner_channels, kernel_size=1, stride=1, padding=0,bias=bias)
        self.smooth2 = ConvBnRelu(in_channels=in_channels[2], out_channels=inner_channels, kernel_size=1, stride=1, padding=0,bias=bias)
        self.smooth3 = ConvBnRelu(in_channels=in_channels[1], out_channels=inner_channels, kernel_size=1, stride=1, padding=0,bias=bias)
        self.smooth4 = ConvBnRelu(in_channels=in_channels[0], out_channels=inner_channels, kernel_size=1, stride=1, padding=0,bias=bias)
        self.out = ConvBnRelu(in_channels=inner_channels*4, out_channels=inner_channels, kernel_size=1, stride=1, padding=0,bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

        for i in range(FPEM_NUM):
            setattr(self, 'fpem_{}'.format(i + 1), FPEM(inner_channels))
        self.FFM_BLOCK = FFM()
        self.FPEM_NUM = FPEM_NUM

    def forward(self, x):
        x_list = []
        x2, x3, x4, x5 = x
        base_map = self.smooth4(x2),self.smooth3(x3),self.smooth2(x4),self.smooth1(x5)
        for i in range(self.FPEM_NUM):
            base_map = getattr(self, 'fpem_{}'.format(i + 1))(base_map)
            x_list.append(base_map)
        outMap = self.FFM_BLOCK(x_list)
        fuse = torch.cat((outMap[0],
                            upsample(outMap[1], outMap[0]),
                            upsample(outMap[2], outMap[0]),
                            upsample(outMap[3], outMap[0])), 1)
        fuse = self.out(fuse)
        return fuse

