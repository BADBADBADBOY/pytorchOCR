#-*- coding:utf-8 _*-
"""
@author:fxw
@file: crnn_head.py
@time: 2020/07/24
"""
import torch.nn as nn
from torch.nn import init

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()

        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            nn.Sigmoid())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        return x * self.se(x)
    
class BLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN_Head(nn.Module):
    def __init__(self,use_conv=False,
                 use_attention=False,
                 use_lstm=True,
                     lstm_num=2,
                     inchannel=512,
                     hiddenchannel=128,
                     classes=1000):
        super(CRNN_Head,self).__init__()
        self.use_lstm = use_lstm
        self.lstm_num = lstm_num
        self.use_conv = use_conv
        if use_attention:
            self.attention = SeModule(inchannel)
        self.use_attention = use_attention
        if(use_lstm):
            assert lstm_num>0 ,Exception('lstm_num need to more than 0 if use_lstm = True')
            for i in range(lstm_num):
                if(i==0):
                    if(lstm_num==1):
                        setattr(self, 'lstm_{}'.format(i + 1), BLSTM(inchannel, hiddenchannel,classes))
                    else:
                        setattr(self, 'lstm_{}'.format(i + 1), BLSTM(inchannel,hiddenchannel,hiddenchannel))
                elif(i==lstm_num-1):
                    setattr(self, 'lstm_{}'.format(i + 1), BLSTM(hiddenchannel, hiddenchannel, classes))
                else:
                    setattr(self, 'lstm_{}'.format(i + 1), BLSTM(hiddenchannel, hiddenchannel, hiddenchannel))
        elif(use_conv):
            self.out = nn.Conv2d(inchannel, classes, kernel_size=1, padding=0)
        else:
            self.out = nn.Linear(inchannel,classes)

    def forward(self, x):
        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        
        if self.use_attention:
            x = self.attention(x)
            
        if(self.use_conv):
            x = self.out(x)
            x = x.squeeze(2)
            x = x.permute(2, 0, 1)
            return x
        
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)  # [w, b, c]
        if self.use_lstm:
            for i in range(self.lstm_num):
                x = getattr(self, 'lstm_{}'.format(i + 1))(x)
        else:
            x = self.out(x)
        return x