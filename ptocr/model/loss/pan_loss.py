#-*- coding:utf-8 _*-
"""
@author:fxw
@file: pan_loss.py
@time: 2020/08/10
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from .basical_loss import DiceLoss,Agg_loss,Dis_loss,ohem_batch

class PANLoss(nn.Module):
    def __init__(self,kernel_rate=0.5,agg_dis_rate=0.25,eps=1e-6):
        super(PANLoss,self).__init__()
        self.kernel_rate = kernel_rate
        self.agg_dis_rate = agg_dis_rate
        self.dice_loss = DiceLoss(eps)
        self.agg_loss = Agg_loss()
        self.dis_loss = Dis_loss()

    def GetKernelLoss(self,pre_text,pre_kernel,gt_kernel,train_mask):
        mask0 = pre_text.data.cpu().numpy()
        mask1 = train_mask.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = Variable(selected_masks)
        if torch.cuda.is_available():
            selected_masks = selected_masks.cuda()
        loss_kernel = self.dice_loss(pre_kernel, gt_kernel, selected_masks)
        return loss_kernel

    def GetTextLoss(self,pre_text,gt_text,train_mask):
        selected_masks = ohem_batch(pre_text, gt_text, train_mask)
        selected_masks = Variable(selected_masks)
        if torch.cuda.is_available():
            selected_masks = selected_masks.cuda()
        loss_text = self.dice_loss(pre_text, gt_text, selected_masks)
        return loss_text

    def forward(self,pred_bach,gt_batch):
        pre_text = torch.sigmoid(pred_bach['pre_text'])
        pre_kernel = torch.sigmoid(pred_bach['pre_kernel'])
        gt_text =  gt_batch['gt_text']
        gt_text_key =  gt_batch['gt_text_key']
        gt_kernel =  gt_batch['gt_kernel']
        gt_kernel_key =  gt_batch['gt_kernel_key']
        train_mask = gt_batch['train_mask']
        similarity_vector = pred_bach['similarity_vector']
        
        pre_text_select = (pre_text > 0.5).float()
        pre_kernel_select = (pre_kernel > 0.5).float()
        gt_text_key = gt_text_key*pre_text_select
        gt_kernel_key = gt_kernel_key*pre_kernel_select
        

        loss_kernel = self.GetKernelLoss(pre_text,pre_kernel,gt_kernel,train_mask)
        loss_text = self.GetTextLoss(pre_text,gt_text,train_mask)
        loss_agg = self.agg_loss.cal_agg_batch(similarity_vector,gt_kernel_key, gt_text_key,train_mask)
        loss_dis = self.dis_loss.cal_Ldis_batch(similarity_vector,gt_kernel_key,train_mask)
        loss = loss_text + self.kernel_rate*loss_kernel + self.agg_dis_rate*(loss_agg + loss_dis)
        metrics = dict(loss_text=loss_text)
        metrics['loss_kernel'] = loss_kernel
        metrics['loss_agg'] = loss_agg
        metrics['loss_dis'] = loss_dis
        return loss,metrics

