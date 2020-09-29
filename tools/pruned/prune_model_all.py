"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: prune.py
@time: 2020/6/27 10:23

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
sys.path.append('./')
import yaml
from ptocr.utils.util_function import create_module,load_model,resize_image
import ptocr
import torch
import torch.nn as  nn
import numpy as np
import collections
import torchvision.transforms as transforms
import cv2

import argparse
import math
from PIL import Image
from torch.autograd import Variable

def prune(args):
    
    stream = open(args.config, 'r', encoding='utf-8')
    config = yaml.load(stream,Loader=yaml.FullLoader)
    
    img = cv2.imread(args.img_file)
    img = resize_image(img,config['base']['algorithm'],config['testload']['test_size'],stride=config['testload']['stride'])
    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).cuda()
    img = Variable(img).unsqueeze(0)

    model = create_module(config['architectures']['model_function'])(config).cuda()
    model = load_model(model,args.checkpoint)
    
    model.eval()
    print(model)
    with torch.no_grad():
        out = model(img)
    cv2.imwrite('ori_model_result.jpg',out[0,0].cpu().numpy()*255)
    cut_percent = args.cut_percent
    base_num =  args.base_num

    bn_weights = []
    for m in model.modules():
        if (isinstance(m, nn.BatchNorm2d)):
            bn_weights.append(m.weight.data.abs().clone())
    bn_weights = torch.cat(bn_weights, 0)

    sort_result, sort_index = torch.sort(bn_weights)

    thresh_index = int(cut_percent * bn_weights.shape[0])

    if (thresh_index == bn_weights.shape[0]):
        thresh_index = bn_weights.shape[0] - 1

    prued = 0
    prued_mask = []
    bn_index = []
    conv_index = []
    remain_channel_nums = []
    for k, m in enumerate(model.modules()):
        if (isinstance(m, nn.BatchNorm2d)):
            bn_weight = m.weight.data.clone()
            mask = bn_weight.abs().gt(sort_result[thresh_index])
            remain_channel = mask.sum()

            if (remain_channel == 0):
                remain_channel = 1
                mask[int(torch.argmax(bn_weight))] = 1

            v = 0
            n = 1
            if (remain_channel % base_num != 0):
                if (remain_channel > base_num):
                    while (v < remain_channel):
                        n += 1
                        v = base_num * n
                    if (remain_channel - (v - base_num) < v - remain_channel):
                        remain_channel = v - base_num
                    else:
                        remain_channel = v
                    if (remain_channel > bn_weight.size()[0]):
                        remain_channel = bn_weight.size()[0]
                    remain_channel = torch.tensor(remain_channel)
                    result, index = torch.sort(bn_weight)
                    mask = bn_weight.abs().ge(result[-remain_channel])

            remain_channel_nums.append(int(mask.sum()))
            prued_mask.append(mask)
            bn_index.append(k)
            prued += mask.shape[0] - mask.sum()
        elif (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
            conv_index.append(k)
    conv_index.remove(227)
    conv_index.remove(236)
    print('remain_channel_nums',remain_channel_nums)
    print('total_prune_ratio:', float(prued) / bn_weights.shape[0])
    print('bn_index',bn_index)
    print('conv_index',conv_index)

#     import pdb
#     pdb.set_trace()

    new_model = create_module(config['architectures']['model_function'])(config).cuda()

    keys = {}
    tag = 0
    for k, m in enumerate(new_model.modules()):
        if(isinstance(m,ptocr.model.backbone.det_mobilev3.Block)):
            keys[tag]=k
            tag+=1
    print(keys)
    #### step 1
    mg_1 = np.array([-3,7,16])
    block_idx = keys[0]
    tag = 0
    for idx in mg_1+block_idx:
        if (tag == 0):
            msk = prued_mask[bn_index.index(idx)]
        else:
            msk = msk | prued_mask[bn_index.index(idx)]
        tag+=1
        print('step1',idx)
    print(msk.sum())
    for idx in mg_1+block_idx:
        prued_mask[bn_index.index(idx)] = msk
    msk_1 = msk.clone()

    #### step 2
    block_idx2 = np.array([keys[1],keys[2]])
    mg_2 = 7
    tag = 0
    for idx in mg_2+block_idx2:
        print('step2',idx)
        if(tag==0):
            msk = prued_mask[bn_index.index(idx)]
        else:
            msk = msk|prued_mask[bn_index.index(idx)]
        tag += 1
    for idx in mg_2+block_idx2:
        prued_mask[bn_index.index(idx)] = msk
    print(msk.sum())
    msk_2 = msk.clone()
    
    ####step 3
    block_idx3s = [keys[3],keys[4],keys[5]]
    mg_3 = np.array([7,16])
    tag = 0
    for block_idx3 in block_idx3s:
        for idx in block_idx3+mg_3:
            print('step3',idx)
            if (tag == 0):
                msk = prued_mask[bn_index.index(idx)]
            else:
                msk = msk | prued_mask[bn_index.index(idx)]
            tag += 1
    for block_idx3 in block_idx3s:
        for idx in block_idx3+mg_3:
            prued_mask[bn_index.index(idx)] = msk
    print(msk.sum())
    msk_3 = msk.clone()

    ####step 4_1
    block_idx4_all = []

    block_idx4 = keys[6]

    mg_4 = np.array([7,16])
    block_idx4_all.extend((block_idx4+mg_4).tolist())

    ####step 4_2
    block_idx4 = keys[7]
    mg_4 = np.array([7,16])
    block_idx4_all.extend((block_idx4+mg_4).tolist())
    tag = 0

    for idx in block_idx4_all:
        print('step4',idx)
        if(tag==0):
            msk = prued_mask[bn_index.index(idx)]
        else:
            msk = msk | prued_mask[bn_index.index(idx)]
        tag += 1

    for idx in block_idx4_all:
         prued_mask[bn_index.index(idx)] = msk
    print(msk.sum())
    msk_4 = msk.clone()
    
    ####step 5
    block_idx5s = [keys[8],keys[9],keys[10]]
    mg_5 = np.array([7,16])
    tag = 0
    for block_idx5 in block_idx5s:
        for idx in block_idx5+mg_5:
            if (tag == 0):
                msk = prued_mask[bn_index.index(idx)]
            else:
                msk = msk | prued_mask[bn_index.index(idx)]
            tag += 1

    for block_idx5 in block_idx5s:
        for idx in block_idx5+mg_5:
            prued_mask[bn_index.index(idx)] = msk
    print(msk.sum())
    msk_5 = msk.clone()
    
    group_index = []
    spl_index = []
    for i in range(11):
        block_idx6 = keys[i]
        tag = 0
        mg_6 = np.array([2,5])
        for idx in mg_6+block_idx6:
            if(tag==0):
                msk = prued_mask[bn_index.index(idx)]
            else:
                msk = msk | prued_mask[bn_index.index(idx)]
            tag+=1
        for idx in mg_6 + block_idx6:
            prued_mask[bn_index.index(idx)] = msk
        if(i==6):
            spl_index.extend([block_idx6+9,block_idx6-2])
        group_index.append(block_idx6+4)
    

    count_conv = 0
    count_bn = 0
    conv_in_mask = [torch.ones(3)]
    conv_out_mask = []
    bn_mask = []
    tag = 0
    for k, m in enumerate(new_model.modules()):
        if(tag>187 ):
            continue
        else:
            if isinstance(m,nn.Conv2d):

                if(tag in group_index):
                    m.groups = int(prued_mask[bn_index.index(tag+1)].sum())
                m.out_channels = int(prued_mask[count_conv].sum())
                conv_out_mask.append(prued_mask[count_conv])
                if(count_conv>0):
                    if (tag == spl_index[0]):
                        m.in_channels = int(prued_mask[bn_index.index(spl_index[1])].sum())
                        conv_in_mask.append(prued_mask[bn_index.index(spl_index[1])])
                    else:
                        m.in_channels = int(prued_mask[count_conv-1].sum())
                        conv_in_mask.append(prued_mask[count_conv-1])

                count_conv+=1
            elif isinstance(m,nn.BatchNorm2d):
                m.num_features = int(prued_mask[count_bn].sum())
                bn_mask.append(prued_mask[count_bn])
                count_bn+=1
        tag+=1
        
    
    head_bn_merge_idx1 = np.array([189,193,197,201])
    tag = 0
    for idx in head_bn_merge_idx1:
        if(tag==0):
            _msk1 = prued_mask[bn_index.index(idx)]
        else:
            _msk1 = _msk1 | prued_mask[bn_index.index(idx)]
        tag += 1    
    
    head_bn_merge_idx2 = np.array([205,209,213,217])
    num_ch = 0
    head_mask_1 = []
    for idx in head_bn_merge_idx2:
        num_ch += int(prued_mask[bn_index.index(idx)].sum())
        head_mask_1.append(prued_mask[bn_index.index(idx)])

        
    print(new_model)
    
    head_bn_merge_idx2 = head_bn_merge_idx2 - 1
    
    bn_i = 0
    conv_i = 0
    model_i = 0
    scale = [188,192,196,200]
    scale_mask = [msk_5,msk_4,msk_3,msk_2]
    import copy
    prued_mask_bk = copy.deepcopy(prued_mask)
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if(model_i>187):
            if isinstance(m0, nn.Conv2d) or isinstance(m0, nn.ConvTranspose2d):
                if(model_i in scale):
                    index = scale.index(model_i)
                    m1.in_channels = int(scale_mask[index].sum())
                    m1.out_channels = int(_msk1.sum())
                    idx0 = np.squeeze(np.argwhere(np.asarray(scale_mask[index].cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(_msk1.cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w = m0.weight.data[:, idx0, :, :].clone()
                    m1.weight.data = w[idx1, :, :, :].clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data[idx0].clone()
                elif(model_i in head_bn_merge_idx2.tolist()):
                   
                    m1.in_channels = int(_msk1.sum())
                    index = head_bn_merge_idx2.tolist().index(model_i)
                    m1.out_channels = int(head_mask_1[index].sum())
                    
                    idx1 = np.squeeze(np.argwhere(np.asarray(head_mask_1[index].cpu().numpy())))
                    idx0 = np.squeeze(np.argwhere(np.asarray(_msk1.cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w = m0.weight.data[:, idx0, :, :].clone()
                    m1.weight.data = w[idx1, :, :, :].clone()
#                     merge_weight_mask.append(m1.weight.data)
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data[idx0].clone()
                elif(model_i==221 or model_i==230):
                    
                    merge_mask = torch.cat(head_mask_1,0)
                    m1.in_channels = num_ch
                    index = bn_index.index(model_i+1)
                    m1.out_channels = int(prued_mask[index].sum())
                    
                    idx1 = np.squeeze(np.argwhere(np.asarray(prued_mask[index].cpu().numpy())))
                    idx0 = np.squeeze(np.argwhere(np.asarray(merge_mask.cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w = m0.weight.data[:, idx0, :, :].clone()
                    m1.weight.data = w[idx1, :, :, :].clone()
                    
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data[idx0].clone()

                elif(model_i==224 or model_i==233):
                   
                    m1.in_channels = int(prued_mask[bn_index.index(model_i-2)].sum())
                    m1.out_channels = int(prued_mask[bn_index.index(model_i+1)].sum())
                    idx0 = np.squeeze(np.argwhere(np.asarray(prued_mask[bn_index.index(model_i+1)].cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(prued_mask[bn_index.index(model_i-2)].cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w = m0.weight.data[:, idx0, :, :].clone()
                    m1.weight.data = w[idx1, :, :, :].clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data[idx0].clone()
                elif(model_i==227 or model_i==236):                
#                     import pdb
#                     pdb.set_trace()
                    m1.in_channels = int(prued_mask[bn_index.index(model_i-2)].sum())
                    idx1 = np.squeeze(np.argwhere(np.asarray(prued_mask[bn_index.index(model_i-2)].cpu().numpy())))
                    idx0 = np.squeeze(np.argwhere(np.asarray(torch.ones(1).cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w = m0.weight.data[:, idx0, :, :].clone()
                    m1.weight.data = w[idx1, :, :, :].clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data[idx0].clone()
                else:
                    m1.weight.data = m0.weight.data.clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data.clone()
                        
            elif isinstance(m0, nn.BatchNorm2d):
                
                if(model_i in  [189,193,197,201]):
                    m1.num_features = int(_msk1.sum())
                    idx1 = np.squeeze(np.argwhere(np.asarray(_msk1.cpu().numpy())))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    m1.weight.data = m0.weight.data[idx1].clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data[idx1].clone()
                    m1.running_mean = m0.running_mean[idx1].clone()
                    m1.running_var = m0.running_var[idx1].clone()
                
                else:
                    
                    index = bn_index.index(model_i)                    
                    m1.num_features = prued_mask[index].sum()
                    idx1 = np.squeeze(np.argwhere(np.asarray(prued_mask[index].cpu().numpy())))
                    
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    m1.weight.data = m0.weight.data[idx1].clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data[idx1].clone()
                    m1.running_mean = m0.running_mean[idx1].clone()
                    m1.running_var = m0.running_var[idx1].clone()
                
                    
        else:
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(bn_mask[bn_i].cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1].clone()
                if m1.bias is not None:
                    m1.bias.data = m0.bias.data[idx1].clone()
                m1.running_mean = m0.running_mean[idx1].clone()
                m1.running_var = m0.running_var[idx1].clone()
                bn_i += 1
            elif isinstance(m0, nn.Conv2d):
                if (isinstance(conv_in_mask[conv_i], list)):
                    idx0 = np.squeeze(np.argwhere(np.asarray(torch.cat(conv_in_mask[conv_i], 0).cpu().numpy())))
                else:
                    idx0 = np.squeeze(np.argwhere(np.asarray(conv_in_mask[conv_i].cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(conv_out_mask[conv_i].cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                if(model_i in group_index):
                    m1.weight.data = m0.weight.data[idx1, :, :, :].clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.clone()
                else:
                    w = m0.weight.data[:, idx0, :, :].clone()
                    m1.weight.data = w[idx1, :, :, :].clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data[idx0].clone()
                conv_i += 1
        model_i+=1

    print('model after pruned')
    print(new_model)
    
    new_model.eval()
    with torch.no_grad():
        out = new_model(img)

    cv2.imwrite('pruned_model_result.jpg',out[0,0].cpu().numpy()*255)

    save_obj = {'prued_mask': prued_mask, 'bn_index': bn_index}
    torch.save(save_obj, os.path.join(args.save_prune_model_path, 'pruned_dict.dict'))
    torch.save(new_model.state_dict(), os.path.join(args.save_prune_model_path, 'pruned_dict.pth'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--base_num', nargs='?', type=int, default = 2,
                        help='Base after Model Channel Clipping')
    parser.add_argument('--cut_percent', nargs='?', type=float, default=0.5,
                        help='Model channel clipping scale')
    parser.add_argument('--config', default='./config/det_DB_mobilev3.yaml',
                        type=str, metavar='PATH',
                        help='config path')
    parser.add_argument('--checkpoint', default='./checkpoint/ag_DB_bb_mobilenet_v3_small_he_DB_Head_bs_16_ep_1200_mobile_slim_all/DB_best.pth.tar',
                        type=str, metavar='PATH',
                        help='ori model path')
    parser.add_argument('--save_prune_model_path', default='./checkpoint/ag_DB_bb_mobilenet_v3_small_he_DB_Head_bs_16_ep_1200_mobile_slim_all/pruned/', type=str, metavar='PATH',
                        help='pruned model path')
    parser.add_argument('--img_file',
                        default='/src/notebooks/detect_text/icdar2015/ch4_test_images/img_108.jpg',
                        type=str,
                        help='')
    args = parser.parse_args()
    prune(args)
