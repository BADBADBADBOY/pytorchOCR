#-*- coding:utf-8 _*-
"""
@author:fxw
@file: crnn_backbone.py
@time: 2020/10/12
"""
import torch
import torch.nn as nn

class conv_bn_relu(nn.Module):
    def __init__(self,in_c,out_c,k_s,s,p,with_bn=True):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_c,out_c,k_s,s,p)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.with_bn = with_bn
    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x

class crnn_backbone(nn.Module):
    def __init__(self,is_gray):
        super(crnn_backbone, self).__init__()
        if(is_gray):
            nc = 1
        else:
            nc = 3
        base_channel = 64
        self.cnn = nn.Sequential(
            conv_bn_relu(nc,base_channel,3,1,1),
            nn.MaxPool2d(2,2),
            conv_bn_relu(base_channel,base_channel*2,3,1,1),
            nn.MaxPool2d(2, 2),
            conv_bn_relu(base_channel*2,base_channel*4,3,1,1),
            conv_bn_relu(base_channel*4,base_channel*4,3,1,1),
            nn.MaxPool2d((2,1),(2,1)),
            conv_bn_relu(base_channel*4,base_channel*8,3,1,1,with_bn=True),
            conv_bn_relu(base_channel*8,base_channel*8,3,1,1,with_bn=True),
            nn.MaxPool2d((2,1), (2,1)),
            conv_bn_relu(base_channel*8,base_channel*8,(2,1),1,0)
            # conv_bn_relu(base_channel*8,base_channel*8,2,1,0)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
                
    def forward(self, x):
        x = self.cnn(x)
        return x

def rec_crnn_backbone(pretrained=False, is_gray=False,**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = crnn_backbone(is_gray)
    if pretrained:
        pretrained_model = torch.load('./pre_model/crnn_backbone.pth')
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                if (key=='features.0.weight' and is_gray):
                    state[key] = torch.mean(pretrained_model[key],1).unsqueeze(1)
                else:
                    state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model


