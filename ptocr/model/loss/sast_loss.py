#-*- coding:utf-8 _*-
"""
@author:fxw
@file: sast_loss.py
@time: 2020/08/18
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from .basical_loss import DiceLoss,BalanceCrossEntropyLoss,ohem_batch

class SASTLoss(nn.Module):
    def __init__(self,tvo_lw,tco_lw,score_lw,border_lw):
        super(SASTLoss,self).__init__()
        self.dict_loss = DiceLoss()
        self.bce_loss = BalanceCrossEntropyLoss()
        self.tvo_lw, self.tco_lw = tvo_lw,tco_lw #1.5, 1.5
        self.score_lw, self.border_lw = score_lw,border_lw #1.0, 1.0

    def forward(self, predicts,labels):

        f_score = predicts['f_score']
        f_border = predicts['f_border']
        f_tvo = predicts['f_tvo']
        f_tco = predicts['f_tco']

        l_score = labels['input_score']
        l_border = labels['input_border']
        l_mask = labels['input_mask']
        l_tvo = labels['input_tvo']
        l_tco = labels['input_tco']

        batch_size,_,w,h = f_score.shape


#         #score_loss add ohem
#         selected_masks = ohem_batch(f_score.squeeze(), l_score.squeeze(), l_mask.squeeze())
#         selected_masks = Variable(selected_masks).unsqueeze(1)
#         if torch.cuda.is_available():
#             selected_masks = selected_masks.cuda()
#         score_loss = self.dict_loss(f_score,l_score,selected_masks)
        
        #score_loss no ohem        
        intersection = torch.sum(f_score * l_score * l_mask)
        union = torch.sum(f_score * l_mask) + torch.sum(l_score * l_mask)
        score_loss = 1.0 - 2 * intersection / (union + 1e-5)

        # border loss
        l_border_split, l_border_norm = l_border[:,0:4,:,:], l_border[:,-1:,:,:]
        f_border_split = f_border
        l_border_norm_split = l_border_norm.expand((batch_size,4,w,h))
        l_border_score = l_score.expand((batch_size,4,w,h))
        l_border_mask = l_mask.expand((batch_size,4,w,h))
        border_diff = l_border_split - f_border_split
        abs_border_diff = torch.abs(border_diff)
        border_sign = abs_border_diff < 1.0
        border_sign = border_sign.float()
        border_sign.stop_gradient = True
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + \
                         (abs_border_diff - 0.5) * (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        border_loss = torch.sum(border_out_loss * l_border_score * l_border_mask) / \
                      (torch.sum(l_border_score * l_border_mask) + 1e-5)

        # tvo_loss
        l_tvo_split, l_tvo_norm = l_tvo[:,0:8,:,:],l_tvo[:,-1:,:,:] 
        f_tvo_split = f_tvo
        l_tvo_norm_split = l_tvo_norm.expand((batch_size,8,w,h))
        l_tvo_score = l_score.expand((batch_size,8,w,h))
        l_tvo_mask = l_mask.expand((batch_size,8,w,h))
        #
        tvo_geo_diff = l_tvo_split - f_tvo_split
        abs_tvo_geo_diff = torch.abs(tvo_geo_diff)
        tvo_sign = abs_tvo_geo_diff < 1.0
        tvo_sign = tvo_sign.float()
        tvo_sign.stop_gradient = True
        tvo_in_loss = 0.5 * abs_tvo_geo_diff * abs_tvo_geo_diff * tvo_sign + \
                      (abs_tvo_geo_diff - 0.5) * (1.0 - tvo_sign)
        tvo_out_loss = l_tvo_norm_split * tvo_in_loss
        tvo_loss = torch.sum(tvo_out_loss * l_tvo_score * l_tvo_mask) / \
                   (torch.sum(l_tvo_score * l_tvo_mask) + 1e-5)

        # tco_loss
        l_tco_split, l_tco_norm = l_tco[:,0:2,:,:],l_tco[:,-1:,:,:]
        f_tco_split = f_tco
        l_tco_norm_split = l_tco_norm.expand((batch_size,2,w,h))
        l_tco_score = l_score.expand((batch_size,2,w,h))
        l_tco_mask = l_mask.expand((batch_size,2,w,h))
        #
        tco_geo_diff = l_tco_split - f_tco_split
        abs_tco_geo_diff = torch.abs(tco_geo_diff)
        tco_sign = abs_tco_geo_diff < 1.0
        tco_sign = tco_sign.float()
        tco_sign.stop_gradient = True
        tco_in_loss = 0.5 * abs_tco_geo_diff * abs_tco_geo_diff * tco_sign + \
                      (abs_tco_geo_diff - 0.5) * (1.0 - tco_sign)
        tco_out_loss = l_tco_norm_split * tco_in_loss
        tco_loss = torch.sum(tco_out_loss * l_tco_score * l_tco_mask) / \
                   (torch.sum(l_tco_score * l_tco_mask) + 1e-5)

        # total loss

        total_loss = score_loss * self.score_lw + border_loss * self.border_lw + \
                     tvo_loss * self.tvo_lw + tco_loss * self.tco_lw

        metrics = {'loss_total': total_loss, "loss_score": score_loss, \
                  "loss_border": border_loss, 'loss_tvo': tvo_loss, 'loss_tco': tco_loss}
        return total_loss,metrics
