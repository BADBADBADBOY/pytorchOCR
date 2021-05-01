#-*- coding:utf-8 _*-
"""
@author:fxw
@file: CRNNProcess.py
@time: 2021/03/23
"""

import lmdb
import torch
import six,re,glob,os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from ptocr.dataloader.RecLoad.DataAgument import transform_img_shape,transform_image_one
from ptocr.utils.util_function import create_module


def get_img(path,is_gray = False):
    try:
        img = Image.open(path).convert('RGB')
    except:
        print(path)
        img = np.zeros(3,320,32)
        img = Image.fromarray(img).convert('RGB')
    if(is_gray):
        img = img.convert('L')
    return img

class alignCollate(object):
    def __init__(self, ):
        pass
    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images, 0)
        return images, labels

class CRNNProcessTrainLmdb(Dataset):
    def __init__(self, config,lmdb_path):
        self.env = lmdb.open(
            lmdb_path,
            max_readers=config['trainload']['num_workers'],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode('utf-8')))
            self.nSamples = nSamples

        self.config = config
        self.transform_label = create_module(config['transform']['function'])
        self.bg_img = []
        for path in glob.glob(os.path.join(config['trainload']['bg_path'],'*')):
            self.bg_img.append(path)


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:

            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = Image.open(buf).convert('RGB')
            except IOError:
                print('IO image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode('utf-8'))

        if self.config['base']['is_gray']:
            img = img.convert('L')

        img = np.array(img)
        if isinstance(label,bytes):
            label = label.decode()

        label = self.transform_label(label,char_type=self.config['transform']['char_type'],t_type=self.config['transform']['t_type'])

         
        try:
            bg_index = np.random.randint(0,len(self.bg_img))
            bg_img = np.array(get_img(self.bg_img[bg_index]))
            img = transform_img_shape(img,self.config['base']['img_shape'])
            img = transform_image_one(img,bg_img,self.config['base']['img_shape'])
            img = transform_img_shape(img,self.config['base']['img_shape'])
        except:
            print('Corrupted image for %d' % index)
            img = np.zeros((32,100)).astype(np.uint8)

        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)

        return (img, label)

def GetDataLoad(config,data_type='train'):
    if data_type == 'train':
        lmdb_path_list = config['trainload']['train_file']
        ratio = config['trainload']['batch_ratio']
    elif data_type == 'val':
        lmdb_path_list = config['valload']['val_file']

    num = len(lmdb_path_list)
    train_data_loaders = []
    sum_num = 0
    for i in range(num):
        if data_type=='train':
            if i==num-1:
                batch_size = config['trainload']['batch_size'] - sum_num
            else:
                batch_size = int(config['trainload']['batch_size']*ratio[i])//2*2
                sum_num+=batch_size
            dataset = CRNNProcessTrainLmdb(config, lmdb_path_list[i])
            num_workers = config['trainload']['num_workers']
            shuffle = True
        elif data_type == 'val':
            batch_size = 1
            num_workers = config['valload']['num_workers']
            dataset = CRNNProcessTrainLmdb(config, lmdb_path_list[i])
            shuffle = False

        train_data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=alignCollate(),
            drop_last=True,
            pin_memory=True)
        train_data_loaders.append(train_data_loader)
    return train_data_loaders

def GetValDataLoad(config):
    
    val_dir = config['valload']['dir']
    root= config['valload']['root']

    num = len(val_dir)
    data_loaders = []
    sum_num = 0
    for i in range(num):
        
        batch_size = 1
        num_workers = config['valload']['num_workers']
        config['valload']['test_file'] = os.path.join(root,val_dir[i],'val_train.txt')
        dataset = CRNNProcessTest(config)
        shuffle = False

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=alignCollate(),
            drop_last=True,
            pin_memory=True)
        data_loaders.append(data_loader)
    return data_loaders

class CRNNProcessTest(Dataset):
    def __init__(self, config):
        super(CRNNProcessTest,self).__init__()
        with open(config['valload']['test_file'],'r',encoding='utf-8') as fid:
            self.label_list = []
            self.image_list = []

            for line in fid.readlines():
                line = line.strip('\n').replace('\ufeff','').split('\t') 
                self.label_list.append(line[1])
                self.image_list.append(line[0])

        self.config = config

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        img = get_img(self.image_list[index],is_gray=self.config['base']['is_gray'])
        img = transform_img_shape(img,self.config['base']['img_shape'])
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)
        label = self.label_list[index]
        return (img,label)
    
    
# if __name__ == "__main__":

#     config = {}
#     config['base'] = {}
#     config['base']['img_shape'] = [32,100]
#     config['trainload'] = {}
#     config['valload'] = {}
#     config['trainload']['bg_path']='./bg_img/'
#     config['base']['is_gray'] = True
#     config['trainload']['train_file'] =['/src/notebooks/MyworkData/EnglishCrnnData/train_lmdb/SynthText/','/src/notebooks/MyworkData/EnglishCrnnData/train_lmdb/MJSynth']

#     config['valload']['val_file'] = ['D:\BaiduNetdiskDownload\data\evaluation\CUTE80',
#                                      'D:\BaiduNetdiskDownload\data\evaluation\IC03_860',
#                                      'D:\BaiduNetdiskDownload\data\evaluation\IC03_867']
#     config['trainload']['num_workers'] = 0
#     config['valload']['num_workers'] = 0
#     config['trainload']['batch_size'] = 128
#     config['trainload']['batch_ratio'] = [0.33, 0.33, 0.33]
#     config['transform']={}

#     config['transform']['char_type'] = 'En'
#     config['transform']['t_type']='lower'
#     config['transform']['function'] = 'ptocr.dataloader.RecLoad.DataAgument,transform_label'
#     import time
#     train_data_loaders = GetDataLoad(config,data_type='train')
# #     import pdb
# #     pdb.set_trace()

# #     t = time.time()
# #     for idx,data1 in enumerate(train_data_loaders[0]):
# #         pass
# #     print(time.time()-t)

#     data1 = enumerate(train_data_loaders[0])
#     for i in range(100):
#         try:
#             t = time.time()
#             index,(data,label) = next(data1)
#             print(time.time()-t)
#             print(label)
#         except:
#             print('end')
    
        