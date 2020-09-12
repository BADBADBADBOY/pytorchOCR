#-*- coding:utf-8 _*-
"""
@author:fxw
@file: pse_loss.py
@time: 2020/08/10
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from .basical_loss import DiceLoss,ohem_batch

class PSELoss(nn.Module):
    def __init__(self,text_tatio=0.7,eps=1e-6):
        super(PSELoss,self).__init__()
        self.text_tatio = text_tatio
        self.dice_loss = DiceLoss(eps)

    def GetKernelLoss(self, pre_text, pre_kernel, gt_kernel, train_mask):
        mask0 = pre_text.data.cpu().numpy()
        mask1 = train_mask.data.cpu().numpy()
        selected_masks = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
        selected_masks = torch.from_numpy(selected_masks).float()
        selected_masks = Variable(selected_masks)
        if torch.cuda.is_available():
            selected_masks = selected_masks.cuda()
        loss_kernels = []
        for i in range(pre_kernel.shape[1]):
            loss_kernel = self.dice_loss(torch.sigmoid(pre_kernel[:,i]), gt_kernel[:,i], selected_masks)
            loss_kernels.append(loss_kernel)
        return sum(loss_kernels)/len(loss_kernels)

    def GetTextLoss(self, pre_text, gt_text, train_mask):
        selected_masks = ohem_batch(pre_text, gt_text, train_mask)
        selected_masks = Variable(selected_masks)
        if torch.cuda.is_available():
            selected_masks = selected_masks.cuda()
        loss_text = self.dice_loss(pre_text, gt_text, selected_masks)
        return loss_text

    def forward(self, pred_bach,gt_batch):
        pre_text = torch.sigmoid(pred_bach['pre_text'])
        pre_kernel = pred_bach['pre_kernel']
        gt_text = gt_batch['gt_text']
        gt_kernel = gt_batch['gt_kernel']
        train_mask = gt_batch['train_mask']

        loss_text = self.GetTextLoss(pre_text,gt_text,train_mask)
        loss_kernel = self.GetKernelLoss(pre_text,pre_kernel,gt_kernel,train_mask)
        loss = self.text_tatio*loss_text + (1 - self.text_tatio)*loss_kernel
        metrics = dict(loss_text=loss_text)
        metrics['loss_kernel'] = loss_kernel
        return loss,metrics