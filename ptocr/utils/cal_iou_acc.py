#-*- coding:utf-8 _*-
"""
@author:fxw
@file: cal_iou_acc.py
@time: 2020/08/13
"""
import torch
import numpy as np

def cal_binary_score(binarys, gt_binarys, training_masks, running_metric_binary, thresh=0.5):
    training_masks = training_masks.data.cpu().numpy()
    pred_binary = binarys.data.cpu().numpy() * training_masks
    pred_binary[pred_binary <= thresh] = 0
    pred_binary[pred_binary > thresh] = 1
    pred_binary = pred_binary.astype(np.int32)
    gt_binary = gt_binarys.data.cpu().numpy() * training_masks
    gt_binary = gt_binary.astype(np.int32)
    running_metric_binary.update(gt_binary, pred_binary)
    score_binary, _ = running_metric_binary.get_scores()
    return score_binary

def cal_text_score(texts, gt_texts, training_masks, running_metric_text, thresh=0.5):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = torch.sigmoid(texts).data.cpu().numpy() * training_masks
    pred_text[pred_text <= thresh] = 0
    pred_text[pred_text > thresh] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text

def cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel, thresh=0.5):
    mask = (gt_texts * training_masks).data.cpu().numpy()
    kernel = kernels[:, -1, :, :]
    gt_kernel = gt_kernels[:, -1, :, :]
    pred_kernel = torch.sigmoid(kernel).data.cpu().numpy()
    pred_kernel[pred_kernel <= thresh] = 0
    pred_kernel[pred_kernel >  thresh] = 1
    pred_kernel = (pred_kernel * mask).astype(np.int32)
    gt_kernel = gt_kernel.data.cpu().numpy()
    gt_kernel = (gt_kernel * mask).astype(np.int32)
    running_metric_kernel.update(gt_kernel, pred_kernel)
    score_kernel, _ = running_metric_kernel.get_scores()
    return score_kernel

def cal_PAN_PSE(kernels, gt_kernels,texts ,gt_texts, training_masks, running_metric_text,running_metric_kernel):
    if(len(kernels.shape)==3):
        kernels = kernels.unsqueeze(1)
        gt_kernels = gt_kernels.unsqueeze(1)
    score_kernel = cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel)
    score_text = cal_text_score(texts, gt_texts, training_masks, running_metric_text)
    acc = (score_text['Mean Acc'] + score_kernel['Mean Acc'])/2
    iou = (score_text['Mean IoU'] + score_kernel['Mean IoU'])/2
    return iou,acc

def cal_DB(texts ,gt_texts, training_masks, running_metric_text):
    score_text = cal_binary_score(texts.squeeze(1), gt_texts.squeeze(1), training_masks.squeeze(1), running_metric_text)
    acc = score_text['Mean Acc']
    iou = score_text['Mean IoU']
    return iou,acc


