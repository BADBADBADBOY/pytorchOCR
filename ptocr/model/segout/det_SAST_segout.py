#-*- coding:utf-8 _*-
"""
@author:fxw
@file: det_SAST_segout.py
@time: 2020/08/18
"""
import torch
import torch.nn as nn
from ..CommonFunction import ConvBnRelu

class SASTHead1(nn.Module):
    def __init__(self):
        super(SASTHead1,self).__init__()
        self.f_score_conv1 = ConvBnRelu(128, 64, 1, 1, 0)
        self.f_score_conv2 = ConvBnRelu(64, 64, 3, 1, 1)
        self.f_score_conv3 = ConvBnRelu(64, 128, 1, 1, 0)
        self.f_score_conv4 = ConvBnRelu(128, 1, 3, 1, 1,with_relu=False)

        self.f_border_conv1 = ConvBnRelu(128, 64, 1, 1, 0)
        self.f_border_conv2 = ConvBnRelu(64, 64, 3, 1, 1)
        self.f_border_conv3 = ConvBnRelu(64, 128, 1, 1, 0)
        self.f_border_conv4 = ConvBnRelu(128, 4, 3, 1, 1,with_relu=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def forward(self, x):
        f_score = self.f_score_conv1(x)
        f_score = self.f_score_conv2(f_score)
        f_score = self.f_score_conv3(f_score)
        f_score = self.f_score_conv4(f_score)
        f_score = torch.sigmoid(f_score)

        f_border = self.f_border_conv1(x)
        f_border = self.f_border_conv2(f_border)
        f_border = self.f_border_conv3(f_border)
        f_border = self.f_border_conv4(f_border)

        return f_score,f_border

class SASTHead2(nn.Module):
    def __init__(self):
        super(SASTHead2, self).__init__()
        self.f_tvo_conv1 = ConvBnRelu(128, 64, 1, 1, 0)
        self.f_tvo_conv2 = ConvBnRelu(64, 64, 3, 1, 1)
        self.f_tvo_conv3 = ConvBnRelu(64, 128, 1, 1, 0)
        self.f_tvo_conv4 = ConvBnRelu(128, 8, 3, 1, 1, with_relu=False)

        self.f_tco_conv1 = ConvBnRelu(128, 64, 1, 1, 0)
        self.f_tco_conv2 = ConvBnRelu(64, 64, 3, 1, 1)
        self.f_tco_conv3 = ConvBnRelu(64, 128, 1, 1, 0)
        self.f_tco_conv4 = ConvBnRelu(128, 2, 3, 1, 1, with_relu=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def forward(self, x):
        f_tvo = self.f_tvo_conv1(x)
        f_tvo = self.f_tvo_conv2(f_tvo)
        f_tvo = self.f_tvo_conv3(f_tvo)
        f_tvo = self.f_tvo_conv4(f_tvo)

        f_tco = self.f_tco_conv1(x)
        f_tco = self.f_tco_conv2(f_tco)
        f_tco = self.f_tco_conv3(f_tco)
        f_tco = self.f_tco_conv4(f_tco)

        return f_tvo, f_tco

class SegDetector(nn.Module):
    def __init__(self):
        super(SegDetector,self).__init__()
        self.sast_head1 = SASTHead1()
        self.sast_head2 = SASTHead2()
    def forward(self, x,img):
        f_score,f_border = self.sast_head1(x)
        f_tvo, f_tco = self.sast_head2(x)
        predicts = {}
        predicts['f_score'] = f_score
        predicts['f_border'] = f_border
        predicts['f_tvo'] = f_tvo
        predicts['f_tco'] = f_tco
        return predicts