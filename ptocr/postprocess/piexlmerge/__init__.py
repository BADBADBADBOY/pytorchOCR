
import os
import cv2
import torch
import time
import subprocess
import numpy as np



BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))

    
def pse(outputs,config):
    
    from .pixelmerge import pse_cpp,get_points,get_num
    
    score = torch.sigmoid(outputs[:, 0, :, :])
    outputs = (torch.sign(outputs - config['postprocess']['binary_th']) + 1) / 2

    text = outputs[:, 0, :, :]
    kernels = outputs[:, 0:config['base']['classes'], :, :] * text

    score = score.data.cpu().numpy()[0].astype(np.float32)
    kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

    pred = pse_cpp(kernels,config['postprocess']['min_kernel_area'] / (config['postprocess']['scale'] * config['postprocess']['scale']))
    pred = np.array(pred).astype(np.uint8)
    label_num = np.max(pred) + 1
    label_points = get_points(pred, score, label_num)
    
    label_values = []
    for label_idx in range(1, label_num):
        label_values.append(label_idx)
    
#     label_values = []
#     label_sum = get_num(pred, label_num)
#     for label_idx in range(1, label_num):
#         if label_sum[label_idx] < config['postprocess']['min_kernel_area']:
#             continue
#         label_values.append(label_idx)
    
        
    return pred,label_points,label_values
    
    
def pan(preds, config):
    
    from .pixelmerge import pan_cpp, get_points, get_num

    pred = (torch.sign(preds[0, 0:2, :, :]-config['postprocess']['bin_th']) + 1) / 2
    score = torch.sigmoid(preds[0]).cpu().numpy().astype(np.float32)
    text = pred[0] # text
    kernel = (pred[1] * text).cpu().numpy() # kernel
    text = text.cpu().numpy()

#     score = torch.sigmoid(preds[0]).cpu().numpy().astype(np.float32)
#     pred_t = torch.sigmoid(preds[0, 0:2, :, :])
#     text = pred_t[0]> 0.8
#     kernel = (pred_t[1]> 0.8)* text
#     text = text.cpu().numpy()
#     kernel = kernel.cpu().numpy()

    similarity_vectors = preds[0,2:].permute((1, 2, 0)).cpu().numpy().astype(np.float32)
    

    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    label_values = []
    label_sum = get_num(label, label_num)
    for label_idx in range(1, label_num):
        if label_sum[label_idx] < config['postprocess']['min_kernel_area']:
            continue
        label_values.append(label_idx)

    pred = pan_cpp(text.astype(np.uint8), similarity_vectors, label, label_num, config['postprocess']['dis_thresh'])
    pred = pred.reshape(text.shape)

    label_points = get_points(pred, score, label_num)
    
    return pred,label_points,label_values
    
    



