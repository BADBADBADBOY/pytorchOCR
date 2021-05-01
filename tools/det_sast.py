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
np.seterr(divide='ignore', invalid='ignore')
import yaml
import torch.utils.data
from ptocr.utils.util_function import create_module, create_loss_bin, \
set_seed,save_checkpoint,create_dir
from ptocr.utils.metrics import runningScore
from ptocr.utils.logger import Logger
from ptocr.utils.cal_iou_acc import cal_DB,cal_PAN_PSE
from tools.cal_rescall.script import cal_recall_precison_f1
from tools.cal_rescall.cal_det import cal_det_metrics
from ptocr.dataloader.DetLoad.SASTProcess_ori import alignCollate
from ptocr.utils.util_function import create_process_obj
torch.backends.cudnn.enabled = False


GLOBAL_WORKER_ID = None
GLOBAL_SEED = 2020

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def ModelTrain(train_data_loader, model, criterion, optimizer, loss_bin, config, epoch):
    if(config['base']['algorithm']=='DB' or config['base']['algorithm']=='SAST'):
        running_metric_text = runningScore(2)
    else:
        running_metric_text = runningScore(2)
        running_metric_kernel = runningScore(2)
    for batch_idx, data in enumerate(train_data_loader):
        if(data is None):
            continue
        pre_batch, gt_batch = model(data)
        loss, metrics = criterion(pre_batch, gt_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cv2.imwrite('pre.jpg',pre_batch['f_score'][0,0].cpu().detach().numpy()*255)
        for key in loss_bin.keys():
            if (key in metrics.keys()):
                loss_bin[key].loss_add(metrics[key].item())
            else:
                loss_bin[key].loss_add(loss.item())
        if(config['base']['algorithm']=='DB'):
            iou,acc = cal_DB(pre_batch['binary'], gt_batch['gt'], gt_batch['mask'], running_metric_text)
        elif(config['base']['algorithm']=='SAST'):
            iou, acc = cal_DB(pre_batch['f_score'], gt_batch['input_score'], gt_batch['input_mask'], running_metric_text)
        else:
            iou,acc = cal_PAN_PSE(pre_batch['pre_kernel'], gt_batch['gt_kernel'], pre_batch['pre_text'], gt_batch['gt_text'],
                                  gt_batch['train_mask'], running_metric_text,running_metric_kernel)

        if (batch_idx % config['base']['show_step'] == 0):
            log = '({}/{}/{}/{}) | ' \
                .format(epoch, config['base']['n_epoch'], batch_idx, len(train_data_loader))
            bin_keys = list(loss_bin.keys())

            for i in range(len(bin_keys)):
                log += bin_keys[i] + ':{:.4f}'.format(loss_bin[bin_keys[i]].loss_mean()) + ' | '

            log +=  'ACC:{:.4f}'.format(acc) + ' | '
            log +=  'IOU:{:.4f}'.format(iou) + ' | '
            log += 'lr:{:.8f}'.format(optimizer.param_groups[0]['lr'])
            print(log)
    loss_write = []
    for key in list(loss_bin.keys()):
        loss_write.append(loss_bin[key].loss_mean())
    loss_write.extend([acc,iou])
    return loss_write


def ModelEval(test_dataset, test_data_loader, model, imgprocess, checkpoints,config):
    bar = tqdm(total=len(test_data_loader))
    for batch_idx, (imgs, ori_imgs) in enumerate(test_data_loader):
        bar.update(1)
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        with torch.no_grad():
            out = model(imgs)
        cv2.imwrite('pre_score.jpg',out['f_score'][0,0].cpu().detach().numpy()*255)
        scales = []
        for i in range(out['f_score'].shape[0]):
            if(config['base']['algorithm']=='SAST'):
                scale = ((out['f_score'].shape[2]*4)/ori_imgs[i].shape[0],(out['f_score'].shape[3]*4)/ori_imgs[i].shape[1] ,ori_imgs[i].shape[0],ori_imgs[i].shape[1])
            else:
                scale = (ori_imgs[i].shape[1] * 1.0 / out.shape[3], ori_imgs[i].shape[0] * 1.0 / out.shape[2])
            scales.append(scale)
        out = create_process_obj(config['base']['algorithm'], out)
        bbox_batch, score_batch = imgprocess(out, scales)
        
        if(config['base']['algorithm']=='SAST'):
            out = out['f_score']
            
        for i in range(len(bbox_batch)):
            bboxes = bbox_batch[i]
            img_show = ori_imgs[i].numpy().copy()
            idx = i + out.shape[0] * batch_idx
            image_name = test_dataset.img_list[idx].split('/')[-1].split('.')[0]  # windows use \\ not /
            with open(os.path.join(checkpoints, 'val', 'res_txt', 'res_' + image_name + '.txt'), 'w+',
                      encoding='utf-8') as fid_res:
                for bbox in bboxes:
                    bbox = bbox.reshape(-1, 2).astype(np.int)
                    img_show = cv2.drawContours(img_show, [bbox], -1, (0, 255, 0), 1)
                    bbox_str = [str(x) for x in bbox.reshape(-1)]
                    bbox_str = ','.join(bbox_str) + '\n'
                    fid_res.write(bbox_str)
            cv2.imwrite(os.path.join(checkpoints, 'val', 'res_img', image_name + '.jpg'), img_show)
    bar.close()
#     result_dict = cal_recall_precison_f1(config['testload']['test_gt_path'],os.path.join(checkpoints, 'val', 'res_txt'))
    result_dict = cal_det_metrics(config['testload']['test_gt_path'],os.path.join(checkpoints, 'val', 'res_txt'))
    return result_dict['recall'],result_dict['precision'],result_dict['hmean']

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

    model = create_module(config['architectures']['model_function'])(config)
    criterion = create_module(config['architectures']['loss_function'])(config)
    train_dataset = create_module(config['trainload']['function'])(config)
    test_dataset = create_module(config['testload']['function'])(config)
    optimizer = create_module(config['optimizer']['function'])(config, model)
    optimizer_decay = create_module(config['optimizer_decay']['function'])
    img_process = create_module(config['postprocess']['function'])(config)
   
    train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['trainload']['batch_size'],
            shuffle=True,
            num_workers=config['trainload']['num_workers'],
            worker_init_fn = worker_init_fn,
            collate_fn=alignCollate(train_dataset),
            drop_last=True,
            pin_memory=True)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['testload']['batch_size'],
        shuffle=False,
        num_workers=config['testload']['num_workers'],
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
    rescall, precision, hmean = 0,0,0
    best_rescall,best_precision ,best_hmean= 0,0,0
    
    if config['base']['restore']:
        print('Resuming from checkpoint.')
        assert os.path.isfile(config['base']['restore_file']), 'Error: no checkpoint file found!'
        checkpoint = torch.load(config['base']['restore_file'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_rescall = checkpoint['rescall']
        best_precision = checkpoint['precision']
        best_hmean = checkpoint['hmean']
        log_write = Logger(os.path.join(checkpoints, 'log.txt'), title=config['base']['algorithm'], resume=True)
    else:
        print('Training from scratch.')
        log_write = Logger(os.path.join(checkpoints, 'log.txt'), title=config['base']['algorithm'])
        title = list(loss_bin.keys())
        title.extend(['piexl_acc','piexl_iou','t_rescall','t_precision','t_hmean','b_rescall','b_precision','b_hmean'])
        log_write.set_names(title)
    
    for epoch in range(start_epoch,config['base']['n_epoch']):
#         train_dataset.gen_train_img()
        model.train()
        optimizer_decay(config, optimizer, epoch)
        loss_write = ModelTrain(train_data_loader, model, criterion, optimizer, loss_bin, config, epoch)
        if(epoch >= config['base']['start_val']):
            create_dir(os.path.join(checkpoints,'val'))
            create_dir(os.path.join(checkpoints,'val','res_img'))
            create_dir(os.path.join(checkpoints,'val','res_txt'))
            model.eval()
            rescall,precision,hmean = ModelEval(test_dataset, test_data_loader, model, img_process, checkpoints,config)
            print('rescall:',rescall,'precision',precision,'hmean',hmean)
            if (hmean > best_hmean):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lr': config['optimizer']['base_lr'],
                    'optimizer': optimizer.state_dict(),
                    'hmean': hmean,
                    'rescall': rescall,
                    'precision': precision
                }, checkpoints, config['base']['algorithm'] + '_best' + '.pth.tar')
                best_hmean = hmean
                best_precision = precision
                best_rescall = rescall

        loss_write.extend([rescall,precision,hmean,best_rescall,best_precision,best_hmean])
        log_write.append(loss_write)
        for key in loss_bin.keys():
            loss_bin[key].loss_clear()
        if epoch%config['base']['save_epoch'] ==0:
            save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lr': config['optimizer']['base_lr'],
            'optimizer': optimizer.state_dict(),
        },checkpoints,config['base']['algorithm']+'_'+str(epoch)+'.pth.tar')


if __name__ == "__main__":
#     stream = open('./config/det_SAST_resnet50_ori_dataload.yaml', 'r', encoding='utf-8')
    stream = open('./config/det_SAST_resnet50_3_3_ori_dataload.yaml', 'r', encoding='utf-8')
    
    config = yaml.load(stream,Loader=yaml.FullLoader)
    TrainValProgram(config)