# -*- coding:utf-8 _*-
"""
@author:fxw
@file: det_train.py
@time: 2020/08/07
"""
import sys
sys.path.append('./')
import cv2
import torch
import time
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
np.seterr(divide='ignore', invalid='ignore')
import yaml
import torch.utils.data
from ptocr.utils.util_function import create_module, create_loss_bin, \
set_seed,save_checkpoint,create_dir
from ptocr.utils.metrics import runningScore
from ptocr.utils.logger import Logger
from ptocr.utils.util_function import create_process_obj,merge_config,AverageMeter
from ptocr.dataloader.RecLoad.CRNNProcess import alignCollate
import copy 

GLOBAL_WORKER_ID = None
GLOBAL_SEED = 2020


torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)



def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)
    
# def backward_hook(self,grad_input, grad_output):
#     for g in grad_input:
#         g[g != g] = 0   # replace all nan/inf in gradients to zero


def ModelTrain(train_data_loader,LabelConverter,model,center_model, criterion, optimizer,center_criterion,optimizer_center,center_flag,loss_bin, config, epoch):
    batch_time = AverageMeter()
    end = time.time()
    for batch_idx, data in enumerate(train_data_loader):
#         model.register_backward_hook(backward_hook)
        if(data is None):
            continue
        imgs,labels = data 
        pre_batch = {}
        gt_batch = {}
        
        if torch.cuda.is_available():
            imgs = imgs.cuda() 
        preds,feau = model(imgs)
                   
        preds = preds.permute(1, 0, 2)

        labels,labels_len = LabelConverter.encode(labels,preds.size(0))
        preds_size = Variable(torch.IntTensor([preds.size(0)] * config['trainload']['batch_size']))
        pre_batch['preds'],pre_batch['preds_size'] = preds,preds_size
        gt_batch['labels'],gt_batch['labels_len'] = labels,labels_len
        #########
        if config['loss']['use_ctc_weight']:

            len_index = torch.softmax(preds,-1).max(2)[1].transpose(0,1)>0
            len_flag = torch.cat([labels_len.cuda().long().unsqueeze(0),len_index.sum(1).unsqueeze(0)],0)
            ctc_loss_weight = len_flag.max(0)[0].float()/len_flag.min(0)[0].float()
            
            ctc_loss_weight[ctc_loss_weight==torch.tensor(np.inf).cuda()]=2.0
            gt_batch['ctc_loss_weight'] = ctc_loss_weight
        
        ctc_loss = criterion(pre_batch, gt_batch).cuda() 
        metrics = {}
        metrics['loss_total'] = 0.0
        metrics['loss_center'] = 0.0
        if center_criterion is not None and center_flag is True:
            center_model.eval()
            #####
            feautures = preds.clone()
            with torch.no_grad():
                center_preds,center_feau = center_model(imgs)
                center_preds = center_preds.permute(1, 0, 2)

            center_preds = torch.softmax(center_preds,-1)
            confs, center_preds = center_preds.max(2)
            center_preds = center_preds.squeeze(1).transpose(1, 0).contiguous()
            confs = confs.transpose(1, 0).contiguous()

#             confs = []
#             for i in range(center_preds.shape[0]):
#                 conf = []
#                 for j in range(len(center_preds[i])):
#                     conf.append(probs[i,j,center_preds[i][j]])
#                 confs.append(conf)
#             confs = torch.Tensor(confs).cuda()

            b,t = center_preds.shape
    
#             feautures = feautures.transpose(1, 0).contiguous()
            feautures = center_feau[0].transpose(1, 0).contiguous()
            
#             import pdb
#             pdb.set_trace()
            ### 去重复
            repeat_index = (center_preds[:,:-1] == center_preds[:,1:])
            center_preds[:,:-1][repeat_index] = 0

            confs = confs.view(-1)
            center_preds = center_preds.view(-1)
            feautures = feautures.view(b*t,-1)


            index = (center_preds>0) & (confs>config['loss']['label_score'])
            center_preds = center_preds[index]
            feautures = feautures[index]

            center_loss = center_criterion(feautures,center_preds)*config['loss']['weight_center']

            loss = ctc_loss + center_loss
            
            
            metrics['loss_total'] = loss.item()
            metrics['loss_ctc'] = ctc_loss.item()
            metrics['loss_center'] = center_loss.item()

            #####
            optimizer_center.zero_grad()
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            for param in center_criterion.parameters():
                param.grad.data *= (1. / config['loss']['weight_center'])
            optimizer_center.step()
        else:
            loss = ctc_loss
            metrics['loss_ctc'] = ctc_loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for key in loss_bin.keys():
            loss_bin[key].loss_add(metrics[key])
        batch_time.update(time.time() - end)
        end = time.time()
        if (batch_idx % config['base']['show_step'] == 0):
            log = '({}/{}/{}/{}) | ' \
                .format(epoch, config['base']['n_epoch'], batch_idx, len(train_data_loader))
            bin_keys = list(loss_bin.keys())

            for i in range(len(bin_keys)):
                log += bin_keys[i] + ':{:.4f}'.format(loss_bin[bin_keys[i]].loss_mean()) + ' | '
            log += 'lr:{:.8f}'.format(optimizer.param_groups[0]['lr'])+ ' | '
            log+='batch_time:{:.2f} s'.format(batch_time.avg)+ ' | '
            log+='total_time:{:.2f} min'.format(batch_time.avg * batch_idx / 60.0)+ ' | '
            log+='ETA:{:.2f} min'.format(batch_time.avg*(len(train_data_loader)-batch_idx)/60.0)
            print(log)
    loss_write = []
    for key in list(loss_bin.keys()):
        loss_write.append(loss_bin[key].loss_mean())
    return loss_write,loss_bin['loss_ctc'].loss_mean()


def ModelEval(test_data_loader,LabelConverter,model,criterion,config):
    bar = tqdm(total=len(test_data_loader))
    loss_avg = []
    n_correct = 0
    for batch_idx, (imgs, labels) in enumerate(test_data_loader):
        bar.update(1)
        pre_batch = {}
        gt_batch = {}
        if torch.cuda.is_available():
            imgs = imgs.cuda() 
        with torch.no_grad():
            preds,feau = model(imgs)
            preds = preds.permute(1, 0, 2)
            
        labels_class, labels_len = LabelConverter.encode(labels,preds.size(0))
        preds_size = Variable(torch.IntTensor([preds.size(0)] * config['testload']['batch_size']))
        pre_batch['preds'],pre_batch['preds_size'] = preds,preds_size
        gt_batch['labels'],gt_batch['labels_len'] = labels_class,labels_len
        
        if config['loss']['use_ctc_weight']:
            len_index = torch.softmax(preds,-1).max(2)[1].transpose(0,1)>0
            len_flag = torch.cat([labels_len.cuda().long().unsqueeze(0),len_index.sum(1).unsqueeze(0)],0)
            ctc_loss_weight = len_flag.max(0)[0].float()/len_flag.min(0)[0].float()
            ctc_loss_weight[ctc_loss_weight==torch.tensor(np.inf).cuda()]=2.0
            gt_batch['ctc_loss_weight'] = ctc_loss_weight
        
        cost = criterion(pre_batch, gt_batch)
        loss_avg.append(cost.item())
        _, preds = preds.max(2)
        preds = preds.squeeze(1)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = LabelConverter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, labels):
            if pred == target:
                n_correct += 1
    raw_preds = LabelConverter.decode(preds.data, preds_size.data, raw=True)[:config['base']['show_num']]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    val_acc = n_correct / float(len(test_data_loader) * config['testload']['batch_size'])
    val_loss = np.mean(loss_avg)
    print('Test loss: %f, accuray: %f' % (val_loss, val_acc))  
    return val_acc,val_loss

def TrainValProgram(config):
    
    config = yaml.load(open(args.config, 'r', encoding='utf-8'),Loader=yaml.FullLoader)
    config = merge_config(config,args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config['base']['gpu_id']

    create_dir(config['base']['checkpoints'])
    checkpoints = os.path.join(config['base']['checkpoints'],
                               "ag_%s_bb_%s_he_%s_bs_%d_ep_%d_%s" % (config['base']['algorithm'],
                                                      config['backbone']['function'].split(',')[-1],
                                                      config['head']['function'].split(',')[-1],
                                                      config['trainload']['batch_size'],
                                                      config['base']['n_epoch'],
                                                      args.log_str))
    create_dir(checkpoints)
    
    LabelConverter = create_module(config['label_transform']['function'])(config)
    config['base']['classes'] = len(LabelConverter.alphabet)
    model = create_module(config['architectures']['model_function'])(config)
    criterion = create_module(config['architectures']['loss_function'])(config)
    train_dataset = create_module(config['trainload']['function'])(config)
    test_dataset = create_module(config['testload']['function'])(config)
    optimizer = create_module(config['optimizer']['function'])(config, model)
    optimizer_decay = create_module(config['optimizer_decay']['function'])
    
    if config['loss']['use_center']:
#         center_criterion = create_module(config['loss']['center_function'])(config['base']['classes'],config['base']['classes'])
        center_criterion = create_module(config['loss']['center_function'])(config['base']['classes'],config['base']['hiddenchannel'])
        
        optimizer_center = torch.optim.Adam(center_criterion.parameters(), lr= config['loss']['center_lr'])
        optimizer_decay_center = create_module(config['optimizer_decay_center']['function'])
    else:
        center_criterion = None
        optimizer_center=None
    
    
    train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['trainload']['batch_size'],
            shuffle=True,
            num_workers=config['trainload']['num_workers'],
            worker_init_fn = worker_init_fn,
            collate_fn = alignCollate(),
            drop_last=True,
            pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['testload']['batch_size'],
        shuffle=False,
        num_workers=config['testload']['num_workers'],
        collate_fn = alignCollate(),
        drop_last=True,
        pin_memory=True)

    loss_bin = create_loss_bin(config['base']['algorithm'],use_center=config['loss']['use_center'])

    if torch.cuda.is_available():
        if (len(config['base']['gpu_id'].split(',')) > 1):
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        criterion = criterion.cuda()
        
    start_epoch = 0
    val_acc = 0
    val_loss = 0
    best_acc = 0
    
    print(model)
    
    if config['base']['restore']:
        print('Resuming from checkpoint.')
        assert os.path.isfile(config['base']['restore_file']), 'Error: no checkpoint file found!'
        checkpoint = torch.load(config['base']['restore_file'])
        start_epoch = checkpoint['epoch']
#         model.load_state_dict(checkpoint['state_dict'])
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            state = model.state_dict()
            for key in state.keys():
                state[key] = checkpoint['state_dict'][key[7:]]
            model.load_state_dict(state)
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        if not config['loss']['use_center']:
            log_write = Logger(os.path.join(checkpoints, 'log.txt'), title=config['base']['algorithm'], resume=True)
        if config['loss']['use_center']:
            if os.path.exists(os.path.join(checkpoints, 'log_center.txt')):
                log_write = Logger(os.path.join(checkpoints, 'log_center.txt'), title=config['base']['algorithm'], resume=True)
            else:
                log_write = Logger(os.path.join(checkpoints, 'log_center.txt'), title=config['base']['algorithm'])
                title = list(loss_bin.keys())
                title.extend(['val_loss','test_acc','best_acc'])
                log_write.set_names(title)
    else:
        print('Training from scratch.')
        log_write = Logger(os.path.join(checkpoints, 'log.txt'), title=config['base']['algorithm'])
        title = list(loss_bin.keys())
        title.extend(['val_loss','test_acc','best_acc'])
        log_write.set_names(title)
    center_flag = False
    center_model = None
    if config['base']['finetune']:
        start_epoch = 0
        optimizer.param_groups[0]['lr'] = 0.0001
        center_flag = True
        center_model = copy.deepcopy(model)
    for epoch in range(start_epoch,config['base']['n_epoch']):
        model.train()
        optimizer_decay(config, optimizer, epoch)
        if config['loss']['use_center']:
            optimizer_decay_center(config, optimizer_center, epoch)
        loss_write,loss_flag = ModelTrain(train_data_loader,LabelConverter, model,center_model, criterion, optimizer, center_criterion,optimizer_center,center_flag,loss_bin, config, epoch)
#         if loss_flag < config['loss']['min_score']:
#             center_flag = True
        if(epoch >= config['base']['start_val']):
            model.eval()
            val_acc,val_loss = ModelEval(test_data_loader,LabelConverter, model,criterion ,config)
            print('val_acc:',val_acc,'val_loss',val_loss)
            if (val_acc > best_acc):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lr': config['optimizer']['base_lr'],
                    'optimizer': optimizer.state_dict(),
                    'best_acc': val_acc
                }, checkpoints, config['base']['algorithm'] + '_best' + '.pth.tar')
                best_acc = val_acc
                

        loss_write.extend([val_loss,val_acc,best_acc])
        log_write.append(loss_write)
        for key in loss_bin.keys():
            loss_bin[key].loss_clear()
        if epoch%config['base']['save_epoch'] ==0:
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lr': config['optimizer']['base_lr'],
            'optimizer': optimizer.state_dict(),
            'best_acc': 0
        },checkpoints,config['base']['algorithm']+'_'+str(epoch)+'.pth.tar')


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--log_str', help='log title')
    args = parser.parse_args()

    
    TrainValProgram(args)