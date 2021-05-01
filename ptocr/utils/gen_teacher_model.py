#-*- coding:utf-8 _*-
"""
@author:fxw
@file: gen_teacher_model.py
@time: 2020/10/15
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from ptocr.utils.util_function import create_module,load_model


class DiceLoss(nn.Module):
    def __init__(self,eps=1e-6):
        super(DiceLoss,self).__init__()
        self.eps = eps
    def forward(self,pre_score,gt_score,train_mask):
        pre_score = pre_score.contiguous().view(pre_score.size()[0], -1)
        gt_score = gt_score.contiguous().view(gt_score.size()[0], -1)
        train_mask = train_mask.contiguous().view(train_mask.size()[0], -1)

        pre_score = pre_score * train_mask
        gt_score = gt_score * train_mask

        a = torch.sum(pre_score * gt_score, 1)
        b = torch.sum(pre_score * pre_score, 1) + self.eps
        c = torch.sum(gt_score * gt_score, 1) + self.eps
        d = (2 * a) / (b + c)
        dice_loss = torch.mean(d)
        return 1 - dice_loss
    
def GetTeacherModel(args):
    config = yaml.load(open(args.t_config, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    model = create_module(config['architectures']['model_function'])(config)
    model = load_model(model,args.t_model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    return model

class DistilLoss(nn.Module):
    def __init__(self):
        
        super(DistilLoss, self).__init__()  
        self.mse = nn.MSELoss()
        self.diceloss = DiceLoss()
        self.ignore = ['thresh']
        
    def forward(self, s_map, t_map):
        loss = 0
        for key in s_map.keys():
            if(key in self.ignore):
                continue
            loss += self.diceloss(s_map[key],t_map[key],torch.ones(t_map[key].shape).cuda())
        return loss
    
    
