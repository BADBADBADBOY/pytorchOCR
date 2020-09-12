#-*- coding:utf-8 _*-
"""
@author:fxw
@file: optimizer.py
@time: 2020/08/11
"""
import torch

def AdamDecay(config,model):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['base_lr'],
                                 betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
                                weight_decay=config['optimizer']['weight_decay'])
    return optimizer

def SGDDecay(config,model):
    optimizer = torch.optim.SGD(model.parameters(), lr=config['optimizer']['base_lr'],
                                 momentum=config['optimizer']['momentum'],
                               weight_decay=config['optimizer']['weight_decay'])
    return optimizer


def RMSPropDecay(config,model):
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['optimizer']['base_lr'], 
                                    alpha=config['optimizer']['alpha'], 
                                    weight_decay=config['optimizer']['weight_decay'],
                                    momentum=config['optimizer']['momentum'])
    return optimizer


def lr_poly(base_lr, epoch, max_epoch=1200, factor=0.9):
    return base_lr*((1-float(epoch)/max_epoch)**(factor))

def adjust_learning_rate_poly(config, optimizer, epoch):
    lr = lr_poly(config['optimizer']['base_lr'], epoch,
                 config['base']['n_epoch'], config['optimizer_decay']['factor'])
    optimizer.param_groups[0]['lr'] = lr

def adjust_learning_rate(config, optimizer, epoch):
    if epoch in config['optimizer_decay']['schedule']:
        config['optimizer']['base_lr'] =config['optimizer']['base_lr'] * config['optimizer_decay']['gama']
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['optimizer']['base_lr']