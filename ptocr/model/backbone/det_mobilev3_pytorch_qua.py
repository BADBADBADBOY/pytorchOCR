#-*- coding:utf-8 _*-
"""
@author:fxw
@file: mobilev3_new.py
@time: 2020/11/02
"""
import torch
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
import torch.nn as nn
from torch.nn import init
from ..CommonFunction_Q import ConvBnRelu,ConvBn


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            ConvBnRelu(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0,groups=1,bias=False),
            ConvBn(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, groups=1,bias=False),
            nn.Sigmoid()
        )
        self.skip = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.skip.mul(x,self.se(x))
        # return x*self.se(x)



class Block(nn.Module):

    def __init__(self, kernel_size, in_size, expand_size, out_size, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv_bn_relu1 = ConvBnRelu(in_size, expand_size, kernel_size=1, stride=1, padding=0,groups=1,bias=False)

        self.conv_bn_relu2 = ConvBnRelu(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.conv_bn1 = ConvBn(expand_size, out_size, kernel_size=1, stride=1, padding=0,groups=1,bias=False)

        self.skip = nn.quantized.FloatFunctional()

        self.shortcut = nn.Sequential()

        self.se = semodule

        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                ConvBn(in_size, out_size, kernel_size=1, stride=1, padding=0, groups=1,bias=False),
            )

    def forward(self, x):
        
        out =self.conv_bn_relu1(x)
        out =self.conv_bn_relu2(out)
        out = self.conv_bn1(out)

        out = self.se(out)
        out = self.skip.add(out , self.shortcut(x)) if self.stride == 1 else out
        return out



class MobileNetV3_Small(nn.Module):
    def __init__(self, ):
        super(MobileNetV3_Small, self).__init__()
        self.conv_bn_relu = ConvBnRelu(3, 16, kernel_size=3, stride=2, padding=1,groups=1, bias=False)
        self.layer1 = nn.Sequential(
            Block(3, 16, 16, 16,SeModule(16), 1),
            Block(3, 16, 72, 24,SeModule(24), 2),
            Block(3, 24, 88, 24, SeModule(24), 1))
        self.layer2 = nn.Sequential(
            Block(5, 24, 96, 40, SeModule(40), 2),
            Block(5, 40, 240, 40, SeModule(40), 1),
            Block(5, 40, 240, 40, SeModule(40), 1))
        self.layer3 = nn.Sequential(
            Block(5, 40, 120, 48, SeModule(48), 2),
            Block(5, 48, 144, 48, SeModule(48), 1))
        self.layer4 = nn.Sequential(
            Block(5, 48, 288, 96, SeModule(96), 2),
            Block(5, 96, 576, 96, SeModule(96), 1),
            Block(5, 96, 576, 96,  SeModule(96), 1))

        self.init_params()
        self.fuse_model()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)
        return [p1,p2,p3,p4]

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBnRelu:
                fuse_modules(m, ['conv', 'bn', 'relu'], inplace=True)
            if type(m) == ConvBn:
               fuse_modules(m, ['conv', 'bn'], inplace=False)
                
                
def mobilenet_v3_small(pretrained,**kwargs):
    model = MobileNetV3_Small()
    
    if pretrained:
        if torch.cuda.is_available():
            pretrained_dict = torch.load('./pre_model/mbv3_small.old.pth.tar')['state_dict']
        else:
            pretrained_dict = torch.load('./pre_model/mbv3_small.old.pth.tar',map_location='cpu')['state_dict']
        try:
            model.load_state_dict(pretrained_dict)
        except:
            state = model.state_dict()
            for key in state.keys():
                if 'module.' + key in pretrained_dict.keys():
                    state[key] = pretrained_dict['module.' + key]
            model.load_state_dict(state)
    return model               

                

