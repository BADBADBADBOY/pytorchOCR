#-*- coding:utf-8 _*-
"""
@author:fxw
@file: det_SASTHead.py
@time: 2020/08/17
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..CommonFunction import ConvBnRelu,DeConvBnRelu


class FPN_Up_Fusion(nn.Module):
    def __init__(self):
        super(FPN_Up_Fusion, self).__init__()

        self.fpn_up_conv1 = ConvBnRelu(2048, 256, 1, 1, 0, with_relu=False)
        self.fpn_up_conv2 = ConvBnRelu(2048, 256, 1, 1, 0, with_relu=False)
        self.fpn_up_conv3 = ConvBnRelu(1024, 192, 1, 1, 0, with_relu=False)
        self.fpn_up_conv4 = ConvBnRelu(512, 192, 1, 1, 0, with_relu=False)
        self.fpn_up_conv5 = ConvBnRelu(256, 128, 1, 1, 0, with_relu=False)

        self.fpn_up_conv6 = ConvBnRelu(256, 256, 3, 1, 1)
        self.fpn_up_conv7 = ConvBnRelu(192, 192, 3, 1, 1)
        self.fpn_up_conv8 = ConvBnRelu(192, 192, 3, 1, 1)

        self.fpn_up_conv9 = ConvBnRelu(128, 128, 3, 1, 1)
        self.fpn_up_conv10 = ConvBnRelu(128, 128, 1, 1, 0, with_relu=False)

        self.fpn_up_deconv1 = DeConvBnRelu(256, 256)
        self.fpn_up_deconv2 = DeConvBnRelu(256, 192)
        self.fpn_up_deconv3 = DeConvBnRelu(192, 192)
        self.fpn_up_deconv4 = DeConvBnRelu(192, 128)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def forward(self, x):
        x = [x[0], x[1], x[2], x[3], x[4]]
        h0 = self.fpn_up_conv1(x[0])
        h1 = self.fpn_up_conv2(x[1])
        h2 = self.fpn_up_conv3(x[2])
        h3 = self.fpn_up_conv4(x[3])
        h4 = self.fpn_up_conv5(x[4])

        g0 = self.fpn_up_deconv1(h0)

        g1 = g0 + h1
        g1 = F.relu(g1)
        g1 = self.fpn_up_conv6(g1)
        g1 = self.fpn_up_deconv2(g1)

        g2 = g1 + h2
        g2 = F.relu(g2)
        g2 = self.fpn_up_conv7(g2)
        g2 = self.fpn_up_deconv3(g2)

        g3 = g2 + h3
        g3 = F.relu(g3)
        g3 = self.fpn_up_conv8(g3)
        g3 = self.fpn_up_deconv4(g3)

        g4 = g3 + h4
        g4 = F.relu(g4)
        g4 = self.fpn_up_conv9(g4)
        g4 = self.fpn_up_conv10(g4)
        return g4


class FPN_Down_Fusion(nn.Module):
    def __init__(self):
        super(FPN_Down_Fusion, self).__init__()

        self.fpn_down_conv1 = ConvBnRelu(3, 32, 1, 1, 0, with_relu=False)
        self.fpn_down_conv2 = ConvBnRelu(128, 64, 1, 1, 0, with_relu=False) # for 3*3
#         self.fpn_down_conv2 = ConvBnRelu(64, 64, 1, 1, 0, with_relu=False) # for 7*7
        self.fpn_down_conv3 = ConvBnRelu(256, 128, 1, 1, 0, with_relu=False)

        self.fpn_down_conv4 = ConvBnRelu(32, 64, 3, 2, 1, with_relu=False)

        self.fpn_down_conv5 = ConvBnRelu(64, 128, 3, 1, 1)
        self.fpn_down_conv6 = ConvBnRelu(128, 128, 3, 2, 1, with_relu=False)

        self.fpn_down_conv7 = ConvBnRelu(128, 128, 3, 1, 1)
        self.fpn_down_conv8 = ConvBnRelu(128, 128, 1, 1, 0, with_relu=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def forward(self, x):
        x = [x[-1], x[-2], x[-3]]

        h0 = self.fpn_down_conv1(x[0])
        h1 = self.fpn_down_conv2(x[1])
        h2 = self.fpn_down_conv3(x[2])

        g0 = self.fpn_down_conv4(h0)
        g1 = g0 + h1
        g1 = F.relu(g1)
        g1 = self.fpn_down_conv5(g1)
        g1 = self.fpn_down_conv6(g1)

        g2 = g1 + h2
        g2 = F.relu(g2)
        g2 = self.fpn_down_conv7(g2)
        g2 = self.fpn_down_conv8(g2)
        return g2
  
class cross_attention(nn.Module):
    
    def __init__(self):
        super(cross_attention,self).__init__()
        self.conv_attention1 = ConvBnRelu(128,128,1,1,0)
        self.conv_attention2 = ConvBnRelu(128,128,1,1,0)
        self.conv_attention3 = ConvBnRelu(128,128,1,1,0)
        
        self.conv_attention4 = ConvBnRelu(128,128,1,1,0,with_relu=False)
        self.conv_attention5 = ConvBnRelu(128,128,1,1,0,with_relu=False)
        self.conv_attention6 = ConvBnRelu(128,128,1,1,0,with_relu=False)
        self.conv_attention7 = ConvBnRelu(128,128,1,1,0,with_relu=False)
        
        self.conv_attention8 = ConvBnRelu(256,128,1,1,0)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)

    def forward(self, x):
        f_shape = x.shape
        f_theta = self.conv_attention1(x)
        f_phi = self.conv_attention2(x)
        f_g = self.conv_attention3(x)
        fh_theta = f_theta
        fh_phi = f_phi
        fh_g = f_g
        # flatten
        fh_theta = fh_theta.permute((0, 2, 3, 1))
        fh_theta = torch.reshape(fh_theta, (f_shape[0] * f_shape[2], f_shape[3], 128))
        fh_phi = fh_phi.permute((0, 2, 3, 1))
        fh_phi = torch.reshape(fh_phi, (f_shape[0] * f_shape[2], f_shape[3], 128))
        fh_g = fh_g.permute((0, 2, 3, 1))
        fh_g = torch.reshape(fh_g, (f_shape[0] * f_shape[2], f_shape[3], 128))
        # correlation
        fh_attn = torch.matmul(fh_theta, fh_phi.permute((0, 2, 1)))
        # scale
        fh_attn = fh_attn / (128 ** 0.5)
        fh_attn = F.softmax(fh_attn,-1)
        # weighted sum
        fh_weight = torch.matmul(fh_attn, fh_g)
        fh_weight = torch.reshape(fh_weight, (f_shape[0], f_shape[2], f_shape[3], 128))
        # print("fh_weight: {}".format(fh_weight.shape))
        fh_weight = fh_weight.permute((0, 3, 1, 2))

        fh_weight = self.conv_attention4(fh_weight)
        fh_sc = self.conv_attention5(x)
        f_h = F.relu(fh_weight + fh_sc)

        # vertical
        fv_theta = f_theta.permute((0, 1, 3, 2))
        fv_phi = f_phi.permute((0, 1, 3, 2))
        fv_g = f_g.permute((0, 1, 3, 2))
        # flatten
        fv_theta = fv_theta.permute((0, 2, 3, 1))
        fv_theta = torch.reshape(fv_theta, (f_shape[0] * f_shape[3], f_shape[2], 128))
        fv_phi = fv_phi.permute((0, 2, 3, 1))
        fv_phi = torch.reshape(fv_phi, (f_shape[0] * f_shape[3], f_shape[2], 128))
        fv_g = fv_g.permute((0, 2, 3, 1))
        fv_g = torch.reshape(fv_g, (f_shape[0] * f_shape[3], f_shape[2], 128))
        # correlation
        fv_attn = torch.matmul(fv_theta, fv_phi.permute((0, 2, 1)))
        # scale
        fv_attn = fv_attn / (128 ** 0.5)
        fv_attn = F.softmax(fv_attn,-1)
        # weighted sum
        fv_weight = torch.matmul(fv_attn, fv_g)
        fv_weight = torch.reshape(fv_weight, (f_shape[0], f_shape[3], f_shape[2], 128))
        # print("fv_weight: {}".format(fv_weight.shape))
        fv_weight = fv_weight.permute((0, 3, 2, 1))
        fv_weight = self.conv_attention6(fv_weight)
        # short cut
        fv_sc = self.conv_attention7(x)
        f_v = F.relu(fv_weight + fv_sc)
        ######
        f_attn = torch.cat([f_h, f_v], 1)
        f_attn = self.conv_attention8(f_attn)
        return f_attn
    

class SASTHead(nn.Module):
    def __init__(self,with_attention=True):
        super(SASTHead,self).__init__()
        self.fpn_up = FPN_Up_Fusion()
        self.fpn_down = FPN_Down_Fusion()
        self.cross_attention = cross_attention()
        self.with_attention = with_attention
    def forward(self, x):
        f_down = self.fpn_down(x)
        f_up = self.fpn_up(x)
        f_common = f_down + f_up
        f_common = F.relu(f_common)

        if(self.with_attention):
            f_common = self.cross_attention(f_common)

        return f_common