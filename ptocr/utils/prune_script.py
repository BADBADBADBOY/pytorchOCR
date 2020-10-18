import torch
import torch.nn as nn
import numpy as np
import copy
from .util_function import load_model
import ptocr

def updateBN(model,args):
    for indedx,m in enumerate(model.modules()):
#         if(indedx>187):
#             break
        if isinstance(m, nn.BatchNorm2d):
            if hasattr(m.weight, 'data'):
                m.weight.grad.data.add_(args.sr_lr*torch.sign(m.weight.data)) #L1正则


def get_pruned_model_backbone(model,new_model,prued_mask,bn_index):
    keys = {}
    tag = 0
    for k, m in enumerate(new_model.modules()):
        if (isinstance(m, ptocr.model.backbone.det_mobilev3.Block)):
            keys[tag] = k
            tag += 1

    #### step 1
    mg_1 = np.array([-3, 7, 16])
    block_idx = keys[0]
    tag = 0
    for idx in mg_1 + block_idx:
        if (tag == 0):
            msk = prued_mask[bn_index.index(idx)]
        else:
            msk = msk | prued_mask[bn_index.index(idx)]
        tag += 1

    for idx in mg_1 + block_idx:
        prued_mask[bn_index.index(idx)] = msk
    msk_1 = msk.clone()

    #### step 2
    block_idx2 = np.array([keys[1], keys[2]])
    mg_2 = 7
    tag = 0
    for idx in mg_2 + block_idx2:
        if (tag == 0):
            msk = prued_mask[bn_index.index(idx)]
        else:
            msk = msk | prued_mask[bn_index.index(idx)]
        tag += 1
    for idx in mg_2 + block_idx2:
        prued_mask[bn_index.index(idx)] = msk
    msk_2 = msk.clone()

    ####step 3
    block_idx3s = [keys[3], keys[4], keys[5]]
    mg_3 = np.array([7, 16])
    tag = 0
    for block_idx3 in block_idx3s:
        for idx in block_idx3 + mg_3:
            if (tag == 0):
                msk = prued_mask[bn_index.index(idx)]
            else:
                msk = msk | prued_mask[bn_index.index(idx)]
            tag += 1
    for block_idx3 in block_idx3s:
        for idx in block_idx3 + mg_3:
            prued_mask[bn_index.index(idx)] = msk
    msk_3 = msk.clone()

    ####step 4_1
    block_idx4_all = []

    block_idx4 = keys[6]

    mg_4 = np.array([7, 16])
    block_idx4_all.extend((block_idx4 + mg_4).tolist())

    ####step 4_2
    block_idx4 = keys[7]
    mg_4 = np.array([7, 16])
    block_idx4_all.extend((block_idx4 + mg_4).tolist())
    tag = 0

    for idx in block_idx4_all:
        if (tag == 0):
            msk = prued_mask[bn_index.index(idx)]
        else:
            msk = msk | prued_mask[bn_index.index(idx)]
        tag += 1

    for idx in block_idx4_all:
        prued_mask[bn_index.index(idx)] = msk
    msk_4 = msk.clone()

    ####step 5
    block_idx5s = [keys[8], keys[9], keys[10]]
    mg_5 = np.array([7, 16])
    tag = 0
    for block_idx5 in block_idx5s:
        for idx in block_idx5 + mg_5:
            if (tag == 0):
                msk = prued_mask[bn_index.index(idx)]
            else:
                msk = msk | prued_mask[bn_index.index(idx)]
            tag += 1

    for block_idx5 in block_idx5s:
        for idx in block_idx5 + mg_5:
            prued_mask[bn_index.index(idx)] = msk
    msk_5 = msk.clone()

    group_index = []
    spl_index = []
    for i in range(11):
        block_idx6 = keys[i]
        tag = 0
        mg_6 = np.array([2, 5])
        for idx in mg_6 + block_idx6:
            if (tag == 0):
                msk = prued_mask[bn_index.index(idx)]
            else:
                msk = msk | prued_mask[bn_index.index(idx)]
            tag += 1
        for idx in mg_6 + block_idx6:
            prued_mask[bn_index.index(idx)] = msk
        if (i == 6):
            spl_index.extend([block_idx6 + 9, block_idx6 - 2])
        group_index.append(block_idx6 + 4)

    count_conv = 0
    count_bn = 0
    conv_in_mask = [torch.ones(3)]
    conv_out_mask = []
    bn_mask = []
    tag = 0
    for k, m in enumerate(new_model.modules()):
        if (tag > 187):
            break
        if isinstance(m, nn.Conv2d):

            if (tag in group_index):
                m.groups = int(prued_mask[bn_index.index(tag + 1)].sum())
            m.out_channels = int(prued_mask[count_conv].sum())
            conv_out_mask.append(prued_mask[count_conv])
            if (count_conv > 0):
                if (tag == spl_index[0]):
                    m.in_channels = int(prued_mask[bn_index.index(spl_index[1])].sum())
                    conv_in_mask.append(prued_mask[bn_index.index(spl_index[1])])
                else:
                    m.in_channels = int(prued_mask[count_conv - 1].sum())
                    conv_in_mask.append(prued_mask[count_conv - 1])

            count_conv += 1
        elif isinstance(m, nn.BatchNorm2d):
            m.num_features = prued_mask[count_bn].sum()
            bn_mask.append(prued_mask[count_bn])
            count_bn += 1
        tag += 1

    bn_i = 0
    conv_i = 0
    model_i = 0
    scale = [188, 192, 196, 200]
    scale_mask = [msk_5, msk_4, msk_3, msk_2]
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if (model_i > 187):
            if isinstance(m0, nn.Conv2d):
                if (model_i in scale):
                    index = scale.index(model_i)
                    m1.in_channels = int(scale_mask[index].sum())
                    idx0 = np.squeeze(np.argwhere(np.asarray(scale_mask[index].cpu().numpy())))
                    idx1 = np.squeeze(np.argwhere(np.asarray(torch.ones(96).cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    if idx1.size == 1:
                        idx1 = np.resize(idx1, (1,))
                    w = m0.weight.data[:, idx0, :, :].clone()
                    m1.weight.data = w[idx1, :, :, :].clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data[idx1].clone()

                else:
                    m1.weight.data = m0.weight.data.clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data.clone()

            elif isinstance(m0, nn.BatchNorm2d):
                m1.weight.data = m0.weight.data.clone()
                if m1.bias is not None:
                    m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
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
                if (model_i in group_index):
                    m1.weight.data = m0.weight.data[idx1, :, :, :].clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.clone()
                else:
                    w = m0.weight.data[:, idx0, :, :].clone()
                    m1.weight.data = w[idx1, :, :, :].clone()
                    if m1.bias is not None:
                        m1.bias.data = m0.bias.data[idx1].clone()
                conv_i += 1
        model_i += 1
    return new_model


def get_pruned_model_total(model,new_model,prued_mask,bn_index):
    
#     new_model = create_module(config['architectures']['model_function'])(config).cuda()

    keys = {}
    tag = 0
    for k, m in enumerate(new_model.modules()):
        if(isinstance(m,ptocr.model.backbone.det_mobilev3.Block)):
            keys[tag]=k
            tag+=1
#     print(keys)
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
#         print('step1',idx)
#     print(msk.sum())
    for idx in mg_1+block_idx:
        prued_mask[bn_index.index(idx)] = msk
    msk_1 = msk.clone()

    #### step 2
    block_idx2 = np.array([keys[1],keys[2]])
    mg_2 = 7
    tag = 0
    for idx in mg_2+block_idx2:
#         print('step2',idx)
        if(tag==0):
            msk = prued_mask[bn_index.index(idx)]
        else:
            msk = msk|prued_mask[bn_index.index(idx)]
        tag += 1
    for idx in mg_2+block_idx2:
        prued_mask[bn_index.index(idx)] = msk
#     print(msk.sum())
    msk_2 = msk.clone()
    
    ####step 3
    block_idx3s = [keys[3],keys[4],keys[5]]
    mg_3 = np.array([7,16])
    tag = 0
    for block_idx3 in block_idx3s:
        for idx in block_idx3+mg_3:
#             print('step3',idx)
            if (tag == 0):
                msk = prued_mask[bn_index.index(idx)]
            else:
                msk = msk | prued_mask[bn_index.index(idx)]
            tag += 1
    for block_idx3 in block_idx3s:
        for idx in block_idx3+mg_3:
            prued_mask[bn_index.index(idx)] = msk
#     print(msk.sum())
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
#         print('step4',idx)
        if(tag==0):
            msk = prued_mask[bn_index.index(idx)]
        else:
            msk = msk | prued_mask[bn_index.index(idx)]
        tag += 1

    for idx in block_idx4_all:
         prued_mask[bn_index.index(idx)] = msk
#     print(msk.sum())
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
#     print(msk.sum())
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

        
#     print(new_model)
    
    head_bn_merge_idx2 = head_bn_merge_idx2 - 1
    
    bn_i = 0
    conv_i = 0
    model_i = 0
    scale = [188,192,196,200]
    scale_mask = [msk_5,msk_4,msk_3,msk_2]
    
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
        
    return new_model

def load_prune_model(model,model_path,pruned_model_dict_path,d_type = 'total'):
    _load = torch.load(pruned_model_dict_path)
    prued_mask = _load['prued_mask']
    bn_index = _load['bn_index']
    if d_type=='total':
        prune_model = get_pruned_model_total(model, model, prued_mask, bn_index)
    else:
        prune_model = get_pruned_model_backbone(model, model, prued_mask, bn_index)
    prune_model = load_model(prune_model,model_path)
    return prune_model

