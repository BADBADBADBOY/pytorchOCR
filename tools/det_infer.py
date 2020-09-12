#-*- coding:utf-8 _*-
"""
@author:fxw
@file: det_infer.py
@time: 2020/08/20
"""
import os
import sys
sys.path.append('./')
import cv2
import torch
import argparse
import yaml
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from ptocr.utils.util_function import create_module,resize_image
from ptocr.utils.util_function import create_process_obj,create_dir,load_model

def config_load(args):
    stream = open(args.config, 'r', encoding='utf-8')
    config = yaml.load(stream,Loader=yaml.FullLoader)
    config['infer']['model_path'] = args.model_path
    config['infer']['path'] = args.img_path
    config['infer']['save_path'] = args.result_save_path
    return config


class TestProgram():
    def __init__(self,config):
        super(TestProgram,self).__init__()
        self.congig = config
        model = create_module(config['architectures']['model_function'])(config)
        img_process = create_module(config['postprocess']['function'])(config)
        model = load_model(model,config['infer']['model_path'])
        if torch.cuda.is_available():
            model = model.cuda()
        self.model = model
        self.img_process = img_process
        self.model.eval()

    def infer_img(self,ori_img):
        img = resize_image(ori_img,self.congig['base']['algorithm'],self.congig['testload']['test_size'],stride=self.congig['testload']['stride'])
        img = Image.fromarray(img).convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0)
        if torch.cuda.is_available():
            img = img.cuda()
            
        with torch.no_grad():
            out = self.model(img)

        if (config['base']['algorithm'] == 'SAST'):
            scale = ((out['f_score'].shape[2]*4)/ori_img.shape[0],(out['f_score'].shape[3]*4)/ori_img.shape[1] ,ori_img.shape[0],ori_img.shape[1])
        else:
            scale = (ori_img.shape[1] * 1.0 / out.shape[3], ori_img.shape[0] * 1.0 / out.shape[2])
        out = create_process_obj(self.congig['base']['algorithm'],out)
        bbox_batch, score_batch = self.img_process(out, [scale])
        return bbox_batch,score_batch

def InferOneImg(bin,img,image_name,save_path):
    img_show = img.copy()
    bbox_batch, score_batch = bin.infer_img(img)
    with open(os.path.join(save_path, 'result_txt', 'res_' + image_name + '.txt'), 'w+', encoding='utf-8') as fid_res:
        for bbox in bbox_batch[0]:
            bbox = bbox.reshape(-1, 2).astype(np.int)
            img_show = cv2.drawContours(img_show, [bbox], -1, (0, 255, 0), 1)
            bbox_str = [str(x) for x in bbox.reshape(-1)]
            bbox_str = ','.join(bbox_str) + '\n'
            fid_res.write(bbox_str)
    cv2.imwrite(os.path.join(save_path, 'result_img', image_name + '.jpg'), img_show)

def InferImage(config):
    path = config['infer']['path']
    save_path = config['infer']['save_path']
    test_bin = TestProgram(config)
    create_dir(save_path)
    create_dir(os.path.join(save_path,'result_img'))
    create_dir(os.path.join(save_path,'result_txt'))
    if os.path.isdir(path):
        files = os.listdir(path)
        bar = tqdm(total=len(files))
        for file in files:
            bar.update(1)
            image_name = file.split('.')[0]
            img_path = os.path.join(path,file)
            img = cv2.imread(img_path)
            InferOneImg(test_bin, img, image_name, save_path)
        bar.close()

    else:
        image_name = path.split('/')[-1].split('.')[0]
        img = cv2.imread(path)
        InferOneImg(test_bin, img, image_name, save_path)


if __name__ == "__main__":
    
#     stream = open('./config/det_PSE_mobilev3.yaml', 'r', encoding='utf-8')
#     stream = open('./config/det_PSE_resnet50.yaml', 'r', encoding='utf-8')
#     stream = open('./config/det_PAN_mobilev3.yaml', 'r', encoding='utf-8')
#     stream = open('./config/det_DB_mobilev3.yaml', 'r', encoding='utf-8')
#     stream = open('./config/det_SAST_resnet50.yaml', 'r', encoding='utf-8')
#     stream = open('./config/det_DB_resnet50.yaml', 'r', encoding='utf-8')
    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--model_path', nargs='?', type=str, default=None)
    parser.add_argument('--img_path', nargs='?', type=str, default=None)
    parser.add_argument('--result_save_path', nargs='?', type=str, default=None)
    args = parser.parse_args()
    config = config_load(args)
    InferImage(config)