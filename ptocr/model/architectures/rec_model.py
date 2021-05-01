# -*- coding:utf-8 _*-
"""
@author:fxw
@file: det_model.py
@time: 2020/08/07
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import create_module
import cv2


class RecModel(nn.Module):
    def __init__(self, config):
        super(RecModel, self).__init__()
        self.algorithm = config['base']['algorithm'] 
        
        self.backbone = create_module(config['backbone']['function'])(config['base']['pretrained'],config['base']['is_gray'])
        if self.algorithm=='CRNN':
            self.head = create_module(config['head']['function'])(
                         use_attention=config['base']['use_attention'],
                         use_lstm=config['base']['use_lstm'],
                         time_step=config['base']['img_shape'][1]//4,
                         lstm_num=config['base']['lstm_num'],
                         inchannel=config['base']['inchannel'],
                         hiddenchannel=config['base']['hiddenchannel'],
                         classes=config['base']['classes'])
        elif self.algorithm=='FC':
            self.head = create_module(config['head']['function'])(in_channels=config['base']['in_channels'],
                              out_channels=config['base']['out_channels'],
                              max_length = config['base']['max_length'],
                               num_class = config['base']['num_class'])

    def forward(self, x):  
        x1 = self.backbone(x)
        x1,feau = self.head(x1)
        return x1,feau

# class RecModel(nn.Module):
#     def __init__(self, config):
#         super(RecModel, self).__init__()
#         self.algorithm = config['base']['algorithm']
#         if config['base']['is_gray']:
#             in_planes = 1
#         else:
#             in_planes = 3
            
#         self.backbone = create_module(config['backbone']['function'])(config['base']['pretrained'],config['base']['is_gray'])
#         self.head = create_module(config['head']['function'])(
#                      use_conv=config['base']['use_conv'],
#                      use_attention=config['base']['use_attention'],
#                      use_lstm=config['base']['use_lstm'],
#                      lstm_num=config['base']['lstm_num'],
#                      inchannel=config['base']['inchannel'],
#                      hiddenchannel=config['base']['hiddenchannel'],
#                      classes=config['base']['classes'])
        
#         self.stn_head = create_module(config['stn']['function'])(in_planes,config)

#     def forward(self, x):
#         cv2.imwrite('stn_ori1.jpg',(x[0,0].cpu().detach().numpy()*0.5+0.5)*255)
       
#         x1= self.stn_head(x)
#         cv2.imwrite('stn1.jpg',(x1[0,0].cpu().detach().numpy()*0.5+0.5)*255)
        
#         x1 = self.backbone(x1)
#         x1 = self.head(x1)
#         return x1
    
# class RecModel(nn.Module):
#     def __init__(self, config):
#         super(RecModel, self).__init__()
#         self.algorithm = config['base']['algorithm']
#         self.tps_inputsize = config['stn']['tps_inputsize']
#         if config['base']['is_gray']:
#             in_planes = 1
#         else:
#             in_planes = 3
            
#         self.backbone = create_module(config['backbone']['function'])(config['base']['pretrained'],config['base']['is_gray'])
#         self.head = create_module(config['head']['function'])(
#                      use_conv=config['base']['use_conv'],
#                      use_attention=config['base']['use_attention'],
#                      use_lstm=config['base']['use_lstm'],
#                      lstm_num=config['base']['lstm_num'],
#                      inchannel=config['base']['inchannel'],
#                      hiddenchannel=config['base']['hiddenchannel'],
#                      classes=config['base']['classes'])
        
#         self.tps = create_module(config['stn']['t_function'])( output_image_size=tuple(config['stn']['tps_outputsize']),
#                                 num_control_points=config['stn']['num_control_points'],
#                                 margins=tuple(config['stn']['tps_margins']))
#         self.stn_head = create_module(config['stn']['function'])(in_planes=in_planes,
#                         num_ctrlpoints=config['stn']['num_control_points'],
#                         activation=config['stn']['stn_activation'])

#     def forward(self, x):
#         cv2.imwrite('stn_ori.jpg',(x[0,0].cpu().detach().numpy()*0.5+0.5)*255)
#         stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
#         stn_img_feat, ctrl_points = self.stn_head(stn_input)

#         x1, _ = self.tps(x, ctrl_points)
        
#         cv2.imwrite('stn.jpg',(x1[0,0].cpu().detach().numpy()*0.5+0.5)*255)
        
#         x1 = self.backbone(x1)
#         x1 = self.head(x1)
#         return x1


class RecLoss(nn.Module):
    def __init__(self, config):
        super(RecLoss, self).__init__()
        self.algorithm = config['base']['algorithm']
        if (config['base']['algorithm']) == 'CRNN':
            self.loss = create_module(config['loss']['function'])(config)
        elif self.algorithm=='FC':
            self.loss = create_module(config['loss']['function'])(ignore_index = config['base']['ignore_index'])
        else:
            assert True == False, ('not support this algorithm !!!')

    def forward(self, pre_batch, gt_batch):
        return self.loss(pre_batch, gt_batch)

