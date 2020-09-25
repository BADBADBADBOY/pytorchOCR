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
import os
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
from ptocr.utils.util_function import create_process_obj
from ptocr.dataloader.RecLoad.CRNNProcess import alignCollate

GLOBAL_WORKER_ID = None
GLOBAL_SEED = 2020

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)
    
def backward_hook(self,grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0   # replace all nan/inf in gradients to zero


def ModelTrain(train_data_loader,LabelConverter,model, criterion, optimizer, loss_bin, config, epoch):
    for batch_idx, data in enumerate(train_data_loader):
        model.register_backward_hook(backward_hook)
        if(data is None):
            continue
        imgs,labels = data 
        pre_batch = {}
        gt_batch = {}
        
        if torch.cuda.is_available():
            imgs = imgs.cuda() 
        preds = model(imgs)
        
        labels,labels_len = LabelConverter.encode(labels,preds.size(0))
        preds_size = Variable(torch.IntTensor([preds.size(0)] * config['trainload']['batch_size']))
        pre_batch['preds'],pre_batch['preds_size'] = preds,preds_size
        gt_batch['labels'],gt_batch['labels_len'] = labels,labels_len
        
        loss = criterion(pre_batch, gt_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for key in loss_bin.keys():
            loss_bin[key].loss_add(loss.item())

        if (batch_idx % config['base']['show_step'] == 0):
            log = '({}/{}/{}/{}) | ' \
                .format(epoch, config['base']['n_epoch'], batch_idx, len(train_data_loader))
            bin_keys = list(loss_bin.keys())

            for i in range(len(bin_keys)):
                log += bin_keys[i] + ':{:.4f}'.format(loss_bin[bin_keys[i]].loss_mean()) + ' | '
            log += 'lr:{:.8f}'.format(optimizer.param_groups[0]['lr'])
            print(log)
    loss_write = []
    for key in list(loss_bin.keys()):
        loss_write.append(loss_bin[key].loss_mean())
    return loss_write


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
            preds = model(imgs)
        labels_class, labels_len = LabelConverter.encode(labels,preds.size(0))
        preds_size = Variable(torch.IntTensor([preds.size(0)] * config['testload']['batch_size']))
        pre_batch['preds'],pre_batch['preds_size'] = preds,preds_size
        gt_batch['labels'],gt_batch['labels_len'] = labels_class,labels_len
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
    os.environ["CUDA_VISIBLE_DEVICES"] = config['base']['gpu_id']

    create_dir(config['base']['checkpoints'])
    checkpoints = os.path.join(config['base']['checkpoints'],
                               "ag_%s_bb_%s_he_%s_bs_%d_ep_%d" % (config['base']['algorithm'],
                                                      config['backbone']['function'].split(',')[-1],
                                                      config['head']['function'].split(',')[-1],
                                                      config['trainload']['batch_size'],
                                                      config['base']['n_epoch']))
    create_dir(checkpoints)
    
    LabelConverter = create_module(config['label_transform']['function'])(config)
    config['base']['classes'] = len(LabelConverter.alphabet)
    model = create_module(config['architectures']['model_function'])(config)
    criterion = create_module(config['architectures']['loss_function'])(config)
    train_dataset = create_module(config['trainload']['function'])(config)
    test_dataset = create_module(config['testload']['function'])(config)
    optimizer = create_module(config['optimizer']['function'])(config, model)
    optimizer_decay = create_module(config['optimizer_decay']['function'])
    
    
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

    loss_bin = create_loss_bin(config['base']['algorithm'])

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
        print('Resuming from checkpoint.')
        assert os.path.isfile(config['base']['restore_file']), 'Error: no checkpoint file found!'
        checkpoint = torch.load(config['base']['restore_file'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        log_write = Logger(os.path.join(checkpoints, 'log.txt'), title=config['base']['algorithm'], resume=True)
    else:
        print('Training from scratch.')
        log_write = Logger(os.path.join(checkpoints, 'log.txt'), title=config['base']['algorithm'])
        title = list(loss_bin.keys())
        title.extend(['val_loss','test_acc','best_acc'])
        log_write.set_names(title)
    
    for epoch in range(start_epoch,config['base']['n_epoch']):
        model.train()
        optimizer_decay(config, optimizer, epoch)
        loss_write = ModelTrain(train_data_loader,LabelConverter, model, criterion, optimizer, loss_bin, config, epoch)
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
    stream = open('./config/rec_CRNN_vgg16_bn.yaml', 'r', encoding='utf-8')
#     stream = open('./config/rec_CRNN_mobilev3.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream,Loader=yaml.FullLoader)
    TrainValProgram(config)