#-*- coding:utf-8 _*-
"""
@author:fxw
@file: PANProcess.py
@time: 2020/08/11
"""
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from .transform_img import Random_Augment
from .MakeSegMap import MakeSegMap
import torchvision.transforms as transforms
from ptocr.utils.util_function import resize_image

class PANProcessTrain(data.Dataset):
    def __init__(self,config):
        super(PANProcessTrain,self).__init__()
        self.crop_shape = config['base']['crop_shape']
        self.TSM = Random_Augment(self.crop_shape)
        self.MSM = MakeSegMap(algorithm=config['base']['algorithm'],shrink_ratio = config['base']['shrink_ratio'])
        img_list, label_list = self.get_base_information(config['trainload']['train_file'])
        self.img_list = img_list
        self.label_list = label_list

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def get_bboxes(self,gt_path):
        polys = []
        tags = []
        with open(gt_path, 'r', encoding='utf-8') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '')
                gt = line.split(',')
                if "###" in gt[-1]:
                    tags.append(True)
                else:
                    tags.append(False)
                # box = [int(gt[i]) for i in range(len(gt)//2*2)]
                box = [int(gt[i]) for i in range(8)]
                polys.append(box)
        return np.array(polys), tags

    def get_base_information(self,train_txt_file):
        label_list = []
        img_list = []
        with open(train_txt_file,'r',encoding='utf-8') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                img_list.append(line[0])
                result = self.get_bboxes(line[1])
                label_list.append(result)
        return img_list,label_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index])
        polys, dontcare = self.label_list[index]

        img, polys = self.TSM.random_scale_pan(img, polys)
        img, polys = self.TSM.random_rotate(img, polys)
        img, polys = self.TSM.random_flip(img, polys)
        img,gt_text,gt_text_key,gt_kernel,gt_kernel_key,train_mask = self.MSM.process(img, polys, dontcare)

        imgs = [img, gt_text, gt_text_key, gt_kernel, gt_kernel_key, train_mask]
        imgs = self.TSM.random_crop_pan(imgs)
        img, gt_text, gt_text_key, gt_kernel, gt_kernel_key, train_mask = imgs[0], imgs[1], imgs[2], imgs[3], imgs[
            4], imgs[5]

        img = Image.fromarray(img).convert('RGB')
        img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        img = self.TSM.normalize_img(img)


        gt_text = torch.from_numpy(gt_text).float()
        gt_text_key = torch.from_numpy(gt_text_key).float()
        gt_kernel = torch.from_numpy(gt_kernel).float()
        gt_kernel_key = torch.from_numpy(gt_kernel_key).float()
        train_mask = torch.from_numpy(train_mask).float()

        return img,gt_text, gt_text_key, gt_kernel, gt_kernel_key, train_mask


class PANProcessTest():
    def __init__(self, config):
        super(PANProcessTest, self).__init__()
        self.img_list = self.get_img_files(config['testload']['test_file'])
        self.TSM = Random_Augment(config['base']['crop_shape'])
        self.test_size = config['testload']['test_size']
        self.config = config

    def get_img_files(self, test_txt_file):
        img_list = []
        with open(test_txt_file, 'r', encoding='utf-8') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.strip('\n')
                img_list.append(line)
        return img_list
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, index):
        ori_img = cv2.imread(self.img_list[index])
        img = resize_image(ori_img,self.config['base']['algorithm'], self.test_size,stride = self.config['testload']['stride'])
        img = Image.fromarray(img).convert('RGB')
        img = self.TSM.normalize_img(img)
        return img, ori_img

