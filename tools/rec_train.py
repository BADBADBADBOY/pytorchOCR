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
from ptocr.utils.logger import Logger,TrainLog
from ptocr.utils.util_function import create_process_obj,merge_config,AverageMeter,restore_training
from ptocr.dataloader.RecLoad.CRNNProcess import alignCollate
import copy 

### 设置随机种子
GLOBAL_SEED = 2020
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

def ModelTrain(train_data_loader,LabelConverter, model,criterion,optimizer, train_log,loss_dict, config, epoch):
    for batch_idx, ( imgs,labels) in enumerate(train_data_loader):
        pre_batch = {}
        gt_batch = {}
        
        if torch.cuda.is_available():
            imgs = imgs.cuda() 
        preds,feau = model(imgs)      
        preds = preds.permute(1, 0, 2) 
        
        #########
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
        
        loss = criterion(pre_batch, gt_batch).cuda() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        metrics = {}
        metrics['ctc_loss'] = loss.item()
        
        for key in loss_dict.keys():
            loss_dict[key].update(metrics[key])

        if (batch_idx % config['base']['show_step'] == 0):
            log = '({}/{}/{}/{}) | ' \
                .format(epoch, config['base']['n_epoch'], batch_idx, len(train_data_loader))
            keys = list(loss_dict.keys())
            for i in range(len(keys)):
                log += keys[i] + ':{:.4f}'.format(loss_dict[keys[i]].avg) + ' | '
            log += 'lr:{:.8f}'.format(optimizer.param_groups[0]['lr'])
            train_log.info(log)

def ModelEval(val_data_loader,LabelConverter, model,criterion,train_log,loss_dict,config):
    
    bar = tqdm(total=len(val_data_loader))
    val_loss = AverageMeter()
    n_correct = AverageMeter()
    
    for batch_idx, (imgs, labels) in enumerate(val_data_loader):
        bar.update(1)
        
        pre_batch = {}
        gt_batch = {}
        
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            
        with torch.no_grad():
            preds,feau = model(imgs)
            preds = preds.permute(1, 0, 2)
            
        labels_class, labels_len = LabelConverter.encode(labels,preds.size(0))
        preds_size = Variable(torch.IntTensor([preds.size(0)] * config['valload']['batch_size']))
        pre_batch['preds'],pre_batch['preds_size'] = preds,preds_size
        gt_batch['labels'],gt_batch['labels_len'] = labels_class,labels_len
        
        if config['loss']['use_ctc_weight']:
            len_index = torch.softmax(preds,-1).max(2)[1].transpose(0,1)>0
            len_flag = torch.cat([labels_len.cuda().long().unsqueeze(0),len_index.sum(1).unsqueeze(0)],0)
            ctc_loss_weight = len_flag.max(0)[0].float()/len_flag.min(0)[0].float()
            ctc_loss_weight[ctc_loss_weight==torch.tensor(np.inf).cuda()]=2.0
            gt_batch['ctc_loss_weight'] = ctc_loss_weight
        
        cost = criterion(pre_batch, gt_batch)
        val_loss.update(cost.item())
        
        _, preds = preds.max(2)
        preds = preds.squeeze(1).transpose(1, 0).contiguous().view(-1)
        
        sim_preds = LabelConverter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, labels):
            if pred == target:
                n_correct.update(1)
    raw_preds = LabelConverter.decode(preds.data, preds_size.data, raw=True)[:config['base']['show_num']]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        train_log.info('recog example %-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    val_acc = n_correct.sum / float(len(val_data_loader) * config['valload']['batch_size'])
    train_log.info('val loss: %f, val accuray: %f' % (val_loss.avg, val_acc))  
    return val_acc


def TrainValProgram(args):
    
    config = yaml.load(open(args.config, 'r', encoding='utf-8'),Loader=yaml.FullLoader)
    config = merge_config(config,args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config['base']['gpu_id']

    create_dir(config['base']['checkpoints'])
    checkpoints = os.path.join(config['base']['checkpoints'],"ag_%s_bb_%s_he_%s_bs_%d_ep_%d_%s" % (config['base']['algorithm'],
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
    train_dataset = create_module(config['trainload']['function'])(config,'train')
    val_dataset = create_module(config['valload']['function'])(config,'val')
    optimizer = create_module(config['optimizer']['function'])(config, model.parameters())
    optimizer_decay = create_module(config['optimizer_decay']['function'])
    if os.path.exists(os.path.join(checkpoints,'train_log.txt')):
        os.remove(os.path.join(checkpoints,'train_log.txt'))
    train_log = TrainLog(os.path.join(checkpoints,'train_log.txt'))
    train_log.info(model)

    train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['trainload']['batch_size'],
            shuffle=True,
            num_workers=config['trainload']['num_workers'],
            collate_fn = alignCollate(),
            drop_last=True,
            pin_memory=True)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['valload']['batch_size'],
        shuffle=False,
        num_workers=config['valload']['num_workers'],
        collate_fn = alignCollate(),
        drop_last=True,
        pin_memory=True)
    
    loss_dict = {}
    for title in config['loss']['loss_title']:
        loss_dict[title] = AverageMeter()

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

    if config['base']['restore']:
        train_log.info('Resuming from checkpoint.')
        assert os.path.isfile(config['base']['restore_file']), 'Error: no checkpoint file found!'
        model,optimizer,start_epoch,best_acc = restore_training(config['base']['restore_file'],model,optimizer)
       
    for epoch in range(start_epoch,config['base']['n_epoch']):
        model.train()
        optimizer_decay(config, optimizer, epoch)
        ModelTrain(train_data_loader,LabelConverter, model,criterion,optimizer, train_log,loss_dict, config, epoch)
        if(epoch >= config['base']['start_val']):
            model.eval()
            val_acc = ModelEval(val_data_loader,LabelConverter, model,criterion,train_log,loss_dict,config)
            if (val_acc > best_acc):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lr': config['optimizer']['base_lr'],
                    'optimizer': optimizer.state_dict(),
                    'best_acc': val_acc
                }, checkpoints, config['base']['algorithm'] + '_best' + '.pth.tar')
                best_acc = val_acc
        train_log.info('best_acc:' + str(best_acc))        
        for key in loss_dict.keys():
            loss_dict[key].reset()
            
        if epoch % config['base']['save_epoch'] == 0:
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