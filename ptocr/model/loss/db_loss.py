#-*- coding:utf-8 _*-
"""
@author:fxw
@file: db_loss.py
@time: 2020/08/10
"""
import torch
import torch.nn as nn
from .basical_loss import MaskL1Loss,BalanceCrossEntropyLoss,DiceLoss,FocalCrossEntropyLoss,MulClassLoss




class DBLoss(nn.Module):
    def __init__(self, l1_scale=10, bce_scale=1,eps=1e-6):
        super(DBLoss, self).__init__()
        self.dice_loss = DiceLoss(eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()
        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, pred_bach, gt_batch):
        bce_loss = self.bce_loss(pred_bach['binary'][:,0], gt_batch['gt'], gt_batch['mask'])
        metrics = dict(loss_bce=bce_loss)
        if 'thresh' in pred_bach:
            l1_loss, l1_metric = self.l1_loss(pred_bach['thresh'][:,0], gt_batch['thresh_map'], gt_batch['thresh_mask'])
            dice_loss = self.dice_loss(pred_bach['thresh_binary'][:,0], gt_batch['gt'], gt_batch['mask'])
            metrics['loss_thresh'] = dice_loss
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
            metrics.update(**l1_metric)
        else:
            loss = bce_loss
        return loss, metrics
    
class DBLossMul(nn.Module):
    def __init__(self, n_class,l1_scale=10, bce_scale=1, class_scale=1,eps=1e-6):
        super(DBLossMul, self).__init__()
        self.dice_loss = DiceLoss(eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()
        self.class_loss = MulClassLoss()
        self.l1_scale = l1_scale
        self.bce_scale = bce_scale
        self.class_scale = class_scale
        self.n_class = n_class

    def forward(self, pred_bach, gt_batch):
        bce_loss = self.bce_loss(pred_bach['binary'][:,0], gt_batch['gt'], gt_batch['mask'])
        class_loss = self.class_loss(pred_bach['binary_class'] ,gt_batch['gt_class'],self.n_class)
        metrics = dict(loss_bce=bce_loss)
        if 'thresh' in pred_bach:
            l1_loss, l1_metric = self.l1_loss(pred_bach['thresh'][:,0], gt_batch['thresh_map'], gt_batch['thresh_mask'])
            dice_loss = self.dice_loss(pred_bach['thresh_binary'][:,0], gt_batch['gt'], gt_batch['mask'])
            metrics['loss_thresh'] = dice_loss
            metrics['loss_class'] = class_loss
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale + class_loss * self.class_scale
            metrics.update(**l1_metric)
        else:
            loss = bce_loss
        return loss, metrics