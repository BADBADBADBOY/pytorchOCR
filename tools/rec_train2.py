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
import re

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


def ModelEval(test_data_loaders,LabelConverter,model,criterion,config):
    loss_all= []
    acc_all = []
    for test_data_loader in test_data_loaders:
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
                

            _,_, labels_class = LabelConverter.test_encode(labels)

            pre_batch['pred'],gt_batch['gt'] = preds,labels_class


            if config['loss']['use_ctc_weight']:
                len_index = torch.softmax(preds,-1).max(2)[1].transpose(0,1)>0
                len_flag = torch.cat([labels_len.cuda().long().unsqueeze(0),len_index.sum(1).unsqueeze(0)],0)
                ctc_loss_weight = len_flag.max(0)[0].float()/len_flag.min(0)[0].float()
                ctc_loss_weight[ctc_loss_weight==torch.tensor(np.inf).cuda()]=2.0
                gt_batch['ctc_loss_weight'] = ctc_loss_weight

            cost,_ = criterion(pre_batch, gt_batch)
            loss_avg.append(cost.item())
            _, preds = preds.max(2)

            sim_preds = LabelConverter.decode(preds.data)
            for pred, target in zip(sim_preds, labels):
                target = ''.join(re.findall('[0-9a-zA-Z]+',target)).lower()
                if pred == target:
                    n_correct += 1
        val_acc = n_correct / float(len(test_data_loader) * config['valload']['batch_size'])
        val_loss = np.mean(loss_avg)
        loss_all.append(val_loss)
        acc_all.append(val_acc)
    print('Test acc:' ,acc_all)
    return loss_all,acc_all

def ModelTrain(train_data_loader,test_data_loader,LabelConverter,model,center_model, criterion, optimizer,center_criterion,optimizer_center,center_flag,loss_bin, config,optimizer_decay,optimizer_decay_center,log_write,checkpoints):
    batch_time = AverageMeter()
    end = time.time()
    best_acc = 0
    dataloader_bin = []
    fid = open('test.txt','w+')
    for i in range(len(train_data_loader)):
        dataloader_bin.append(enumerate(train_data_loader[i]))
    
    for iters in range(config['base']['start_iters'],config['base']['max_iters']):
        model.train()
        optimizer_decay(config, optimizer, iters)
        if config['loss']['use_center']:
            optimizer_decay_center(config, optimizer_center, iters)
        
        imgs = []
        labels = []
        try:
            for i in range(len(train_data_loader)):
                index,(img,label) = next(dataloader_bin[i])
                imgs.append(img)
                labels.extend(label)
            imgs = torch.cat(imgs,0)
        except:
            for i in range(len(train_data_loader)):
                dataloader_bin[i] = enumerate(train_data_loader[i])
            continue

        pre_batch = {}
        gt_batch = {}

        _, _, labels = LabelConverter.train_encode(labels)

        if torch.cuda.is_available():
            imgs = imgs.cuda() 
        preds,feau = model(imgs)
        

        pre_batch['pred'],gt_batch['gt'] = preds,labels.long().cuda()
        #########
        
        
        
        if config['loss']['use_ctc_weight']:
#             print('use')

            len_index = torch.softmax(preds,-1).max(2)[1].transpose(0,1)>0
            len_flag = torch.cat([labels_len.cuda().long().unsqueeze(0),len_index.sum(1).unsqueeze(0)],0)
            ctc_loss_weight = len_flag.max(0)[0].float()/len_flag.min(0)[0].float()
            
            ctc_loss_weight[ctc_loss_weight==torch.tensor(np.inf).cuda()]=2.0
            gt_batch['ctc_loss_weight'] = ctc_loss_weight
        
        loss,_ = criterion(pre_batch, gt_batch)
        
#         import pdb
#         pdb.set_trace()
        
        metrics = {}
        metrics['loss_fc'] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for key in loss_bin.keys():
            loss_bin[key].loss_add(metrics[key])
        batch_time.update(time.time() - end)
        end = time.time()
        if (iters % config['base']['show_step'] == 0):
            log = '({}/{}) | ' \
                .format(iters,config['base']['max_iters'])
            bin_keys = list(loss_bin.keys())

            for i in range(len(bin_keys)):
                log += bin_keys[i] + ':{:.4f}'.format(loss_bin[bin_keys[i]].loss_mean()) + ' | '
            log += 'lr:{:.8f}'.format(optimizer.param_groups[0]['lr'])+ ' | '
            log+='batch_time:{:.2f} s'.format(batch_time.avg)+ ' | '
            log+='total_time:{:.2f} min'.format(batch_time.avg * iters / 60.0)+ ' | '
            log+='ETA:{:.2f} min'.format(batch_time.avg*(config['base']['max_iters']-iters)/60.0)
            print(log)
            
        
        if(iters % config['base']['eval_iter']==0 and iters!=0):
            
            loss_write = []
            for key in list(loss_bin.keys()):
                loss_write.append(loss_bin[key].loss_mean())
            
            model.eval()
            val_loss,acc_all = ModelEval(test_data_loader,LabelConverter, model,criterion ,config)
            val_acc = np.mean(acc_all)
            if (val_acc > best_acc):
                save_checkpoint({
                    'iters': iters + 1,
                    'state_dict': model.state_dict(),
                    'lr': config['optimizer']['base_lr'],
                    'optimizer': optimizer.state_dict(),
                    'best_acc': val_acc
                }, checkpoints, config['base']['algorithm'] + '_best' + '.pth.tar')
                best_acc = val_acc
            acc_all.append(val_acc)
            acc_all = [str(x) for x in acc_all]
            fid.write(str(','.join(acc_all))+'\n')
            fid.flush()
            loss_write.extend([0,0,0])
            log_write.append(loss_write)
            for key in loss_bin.keys():
                loss_bin[key].loss_clear()
            if iters %config['base']['eval_iter'] ==0:
                save_checkpoint({
                'iters': iters + 1,
                'state_dict': model.state_dict(),
                'lr': config['optimizer']['base_lr'],
                'optimizer': optimizer.state_dict(),
                'best_acc': 0
            },checkpoints,config['base']['algorithm']+'_'+str(iters)+'.pth.tar')
            
            
                
            





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
                                                      config['base']['max_iters'],
                                                      args.log_str))
    create_dir(checkpoints)
    
    LabelConverter = create_module(config['label_transform']['function'])(config)
    
    model = create_module(config['architectures']['model_function'])(config)
    criterion = create_module(config['architectures']['loss_function'])(config)
    train_data_loader = create_module(config['trainload']['function'])(config)
    test_data_loader = create_module(config['valload']['function'])(config)
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
        optimizer_decay_center = None
    

    

    loss_bin = create_loss_bin(config['base']['algorithm'],use_center=config['loss']['use_center'])

    if torch.cuda.is_available():
        if (len(config['base']['gpu_id'].split(',')) > 1):
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        criterion = criterion.cuda()
        

    
    print(model)
    
    ### model.head.lstm_2.embedding = nn.Linear(in_features=512, out_features=2000, bias=True)
    
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
    
    
    loss_write,loss_flag = ModelTrain(train_data_loader,test_data_loader,LabelConverter, model,center_model, criterion, optimizer, center_criterion,optimizer_center,center_flag,loss_bin, config,optimizer_decay,optimizer_decay_center,log_write,checkpoints)

    


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--log_str', help='log title')
    args = parser.parse_args()

    
    TrainValProgram(args)