# -*- coding:utf-8 _*-
"""
@author:fxw
@file: crnn_head.py
@time: 2020/07/24
"""
import torch
import torch.nn as nn
from torch.nn import init


class channelattention(nn.Module):
    def __init__(self, time_step):
        super(channelattention, self).__init__()
        self.attention = nn.Sequential(nn.Linear(time_step, 8),
                                       nn.Linear(8, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        att = self.attention(x)
        att = torch.sigmoid(att)
        out = att * x
        return out.permute(2, 0, 1)


class timeattention(nn.Module):
    def __init__(self, inchannel):
        super(timeattention, self).__init__()
        self.attention = nn.Sequential(nn.Linear(inchannel, inchannel // 4),
                                       nn.Linear(inchannel // 4, inchannel // 8),
                                       nn.Linear(inchannel // 8, 1))

    def forward(self, x):
        att = self.attention(x)
        att = torch.sigmoid(att)
        out = att * x
        return out.permute(1, 0, 2)


class BLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()
            setattr(self, '_flattened', True)
        recurrent, _ = self.rnn(input)
        b, T, h = recurrent.size()
        t_rec = recurrent.contiguous().view(b * T, h)

        output = self.embedding(t_rec)  # [b * T, nOut]
        output = output.contiguous().view(b, T, -1)

        return output


class CRNN_Head(nn.Module):
    def __init__(self, use_attention=False,
                 use_lstm=True,
                 time_step=25,
                 lstm_num=2,
                 inchannel=512,
                 hiddenchannel=128,
                 classes=1000):
        super(CRNN_Head, self).__init__()
        self.use_lstm = use_lstm
        self.lstm_num = lstm_num
        self.use_attention = use_attention

        if use_attention:
            self.channel_attention = channelattention(time_step=time_step)
            self.time_attention = timeattention(inchannel)
        if (use_lstm):
            assert lstm_num > 0, Exception('lstm_num need to more than 0 if use_lstm = True')
            for i in range(lstm_num):
                if (i == 0):
                    if (lstm_num == 1):
                        setattr(self, 'lstm_{}'.format(i + 1), BLSTM(inchannel, hiddenchannel, classes))
                    else:
                        setattr(self, 'lstm_{}'.format(i + 1), BLSTM(inchannel, hiddenchannel, hiddenchannel))
                elif (i == lstm_num - 1):
                    setattr(self, 'lstm_{}'.format(i + 1), BLSTM(hiddenchannel, hiddenchannel, classes))
                else:
                    setattr(self, 'lstm_{}'.format(i + 1), BLSTM(hiddenchannel, hiddenchannel, hiddenchannel))
        else:
            self.out = nn.Linear(inchannel, classes)

    def forward(self, x):
        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"

        x = x.squeeze(2)
        x = x.permute(0, 2, 1)  # [b, w, c]

        ############
        if self.use_attention:
            x = self.channel_attention(x)
            x = self.time_attention(x)

        ############

        feau = []
        if self.use_lstm:
            for i in range(self.lstm_num):
                x = getattr(self, 'lstm_{}'.format(i + 1))(x)
                feau.append(x)
        else:
            feau.append(x)
            x = self.out(x)

        return x, feau


    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0