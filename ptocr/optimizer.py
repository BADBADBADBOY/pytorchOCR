#-*- coding:utf-8 _*-
"""
@author:fxw
@file: optimizer.py
@time: 2020/08/11
"""
import torch

def AdamDecay(config,parameters):
    optimizer = torch.optim.Adam(parameters, lr=config['optimizer']['base_lr'],
                                 betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
                                weight_decay=config['optimizer']['weight_decay'])
    return optimizer

def SGDDecay(config,parameters):
    optimizer = torch.optim.SGD(parameters, lr=config['optimizer']['base_lr'],
                                 momentum=config['optimizer']['momentum'],
                               weight_decay=config['optimizer']['weight_decay'])
    return optimizer


def RMSPropDecay(config,parameters):
    optimizer = torch.optim.RMSprop(parameters, lr=config['optimizer']['base_lr'], 
                                    alpha=config['optimizer']['alpha'], 
                                    weight_decay=config['optimizer']['weight_decay'],
                                    momentum=config['optimizer']['momentum'])
    return optimizer


def lr_poly(base_lr, epoch, max_epoch=1200, factor=0.9):
    return base_lr*((1-float(epoch)/max_epoch)**(factor))


def SGDR(lr_max,lr_min,T_cur,T_m,ratio=0.3):
    """
    :param lr_max: 最大学习率
    :param lr_min: 最小学习率
    :param T_cur: 当前的epoch或iter
    :param T_m: 隔多少调整的一次
    :param ratio: 最大学习率衰减比率
    :return:
    """
    if T_cur % T_m == 0 and T_cur != 0:
        lr_max = lr_max - lr_max * ratio
    lr = lr_min+1/2*(lr_max-lr_min)*(1+math.cos((T_cur%T_m/T_m)*math.pi))
    return lr,lr_max


def adjust_learning_rate_poly(config, optimizer, epoch):
    lr = lr_poly(config['optimizer']['base_lr'], epoch,
                 config['base']['n_epoch'], config['optimizer_decay']['factor'])
    optimizer.param_groups[0]['lr'] = lr
    
def adjust_learning_rate_sgdr(config, optimizer, epoch):
    lr,lr_max = SGDR(config['optimizer']['lr_max'],config['optimizer']['lr_min'],epoch,config['optimizer']['T_m'],config['optimizer']['ratio'])
    optimizer.param_groups[0]['lr'] = lr
    config['optimizer']['lr_max'] = lr_max
    
def adjust_learning_rate(config, optimizer, epoch):
    if epoch in config['optimizer_decay']['schedule']:
        adjust_lr = optimizer.param_groups[0]['lr'] * config['optimizer_decay']['gama']
        for param_group in optimizer.param_groups:
            param_group['lr'] = adjust_lr
            
def adjust_learning_rate_center(config, optimizer, epoch):
    if epoch in config['optimizer_decay_center']['schedule']:
        adjust_lr = optimizer.param_groups[0]['lr'] * config['optimizer_decay_center']['gama']
        for param_group in optimizer.param_groups:
            param_group['lr'] = adjust_lr