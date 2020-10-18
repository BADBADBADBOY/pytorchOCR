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
import yaml
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from ptocr.utils.util_function import create_module,resize_image
from ptocr.utils.util_function import create_process_obj,create_dir,load_model


class TestProgram():
    def __init__(self,config):
        super(TestProgram,self).__init__()
        
        self.converter = create_module(config['label_transform']['function'])(config)
        config['base']['classes'] = len(self.converter.alphabet)
        model = create_module(config['architectures']['model_function'])(config)
        model = load_model(model,config['infer']['model_path'])
        if torch.cuda.is_available():
            model = model.cuda()
        self.model = model
        self.congig = config
        self.model.eval()

    def infer_img(self,ori_img):
        img = resize_image(ori_img,self.congig['base']['algorithm'],32)
        img = Image.fromarray(img).convert('RGB')
        if(self.congig['base']['is_gray']):
            img = img.convert('L')
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)
        img = img.unsqueeze(0)
        if torch.cuda.is_available():
            img = img.cuda()
            
        with torch.no_grad():
            preds = self.model(img)
        preds_size = torch.IntTensor([preds.size(0)])   
        _, preds = preds.max(2)
        preds = preds.squeeze(1)
        preds = preds.contiguous().view(-1)
        sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
        
        return sim_preds

def InferImage(config):
    path = config['infer']['path']
    save_path = config['infer']['save_path']
    test_bin = TestProgram(config)
    if os.path.isdir(path):
        files = os.listdir(path)
        bar = tqdm(total=len(files))
        for file in files:
            bar.update(1)
            image_name = file.split('.')[0]
            img_path = os.path.join(path,file)
            img = cv2.imread(img_path)
            rec_char = test_bin.infer_img(img)
            print(rec_char)
        bar.close()

    else:
        image_name = path.split('/')[-1].split('.')[0]
        img = cv2.imread(path)
        rec_char = test_bin.infer_img(img)
        print(rec_char)


if __name__ == "__main__":
    stream = open('./config/rec_CRNN_ori.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream,Loader=yaml.FullLoader)
    InferImage(config)