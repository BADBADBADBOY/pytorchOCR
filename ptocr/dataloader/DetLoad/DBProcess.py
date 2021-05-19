#-*- coding:utf-8 _*-
"""
@author:fxw
@file: DBProcess.py
@time: 2020/08/11
"""
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from .MakeBorderMap import MakeBorderMap
from .transform_img import Random_Augment
from .MakeSegMap import MakeSegMap
import torchvision.transforms as transforms
from ptocr.utils.util_function import resize_image

class DBProcessTrain(data.Dataset):
    def __init__(self,config):
        super(DBProcessTrain,self).__init__()
        self.crop_shape = config['base']['crop_shape']
        self.MBM = MakeBorderMap()
        self.TSM = Random_Augment(self.crop_shape)
        self.MSM = MakeSegMap(shrink_ratio = config['base']['shrink_ratio'])
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
                if "#" in gt[-1]:
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
        
        img = Image.open(self.img_list[index]).convert('RGB')
        img = np.array(img)[:,:,::-1]
        
        polys, dontcare = self.label_list[index]

        img, polys = self.TSM.random_scale(img, polys, self.crop_shape[0])
        img, polys = self.TSM.random_rotate(img, polys)
        img, polys = self.TSM.random_flip(img, polys)
        img, polys, dontcare = self.TSM.random_crop_db(img, polys, dontcare)
        img, gt, gt_mask = self.MSM.process(img, polys, dontcare)
        img, thresh_map, thresh_mask = self.MBM.process(img, polys, dontcare)

        img = Image.fromarray(img).convert('RGB')
        img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        img = self.TSM.normalize_img(img)


        gt = torch.from_numpy(gt).float()
        gt_mask = torch.from_numpy(gt_mask).float()
        thresh_map = torch.from_numpy(thresh_map).float()
        thresh_mask = torch.from_numpy(thresh_mask).float()

        return img,gt,gt_mask,thresh_map,thresh_mask

class DBProcessTrainMul(data.Dataset):
    def __init__(self,config):
        super(DBProcessTrainMul,self).__init__()
        self.crop_shape = config['base']['crop_shape']
        self.MBM = MakeBorderMap()
        self.TSM = Random_Augment(self.crop_shape)
        self.MSM = MakeSegMap(shrink_ratio = config['base']['shrink_ratio'])
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
        classes = []
        with open(gt_path, 'r', encoding='utf-8') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '')
                gt = line.split(',')
                if "#" in gt[-1]:
                    tags.append(True)
                    classes.append(-2)
                else:
                    tags.append(False)
                    classes.append(int(gt[-1]))
                # box = [int(gt[i]) for i in range(len(gt)//2*2)]
                box = [int(gt[i]) for i in range(8)]
                polys.append(box)
        return np.array(polys), tags, classes

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
        
        img = Image.open(self.img_list[index]).convert('RGB')
        img = np.array(img)[:,:,::-1]
        
        polys, dontcare, classes = self.label_list[index]

        img, polys = self.TSM.random_scale(img, polys, self.crop_shape[0])
        img, polys = self.TSM.random_rotate(img, polys)
        img, polys = self.TSM.random_flip(img, polys)
        img, polys, classes,dontcare = self.TSM.random_crop_db_mul(img, polys,classes, dontcare)
        img, gt, classes,gt_mask = self.MSM.process_mul(img, polys, classes,dontcare)
        img, thresh_map, thresh_mask = self.MBM.process(img, polys, dontcare)

        img = Image.fromarray(img).convert('RGB')
        img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        img = self.TSM.normalize_img(img)


        gt = torch.from_numpy(gt).float()
        gt_classes = torch.from_numpy(classes).long()
        gt_mask = torch.from_numpy(gt_mask).float()
        thresh_map = torch.from_numpy(thresh_map).float()
        thresh_mask = torch.from_numpy(thresh_mask).float()

        return img,gt,gt_classes,gt_mask,thresh_map,thresh_mask

class DBProcessTest(data.Dataset):
    def __init__(self,config):
        super(DBProcessTest,self).__init__()
        self.img_list = self.get_img_files(config['testload']['test_file'])
        self.TSM = Random_Augment(config['base']['crop_shape'])
        self.test_size = config['testload']['test_size']
        self.config =config

    def get_img_files(self,test_txt_file):
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
        return img,ori_img