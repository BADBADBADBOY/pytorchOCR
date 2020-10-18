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
import onnxruntime
import torchvision.transforms as transforms
from ptocr.utils.util_function import create_module,resize_image_batch
from ptocr.utils.util_function import create_process_obj,create_dir,load_model
from script.onnx_to_tensorrt import build_engine,allocate_buffers,do_inference

def config_load(args):
    stream = open(args.config, 'r', encoding='utf-8')
    config = yaml.load(stream,Loader=yaml.FullLoader)
    config['infer']['model_path'] = args.model_path
    config['infer']['path'] = args.img_path
    config['infer']['save_path'] = args.result_save_path
    config['infer']['onnx_path'] = args.onnx_path
    config['infer']['trt_path'] = args.trt_path
    config['testload']['add_padding'] = args.add_padding
    config['testload']['test_size'] = args.max_size
    config['testload']['batch_size'] = args.batch_size
    return config


def get_batch_files(path,img_files,batch_size=3):
    img_files = np.array(img_files)
    num = len(img_files)//batch_size
    batch_imgs = []
    batch_img_names = []
    for i in range(num):
        files = img_files[batch_size*i:batch_size*(i+1)]
        img = [cv2.imread(os.path.join(path,img_file)) for img_file in files]
        img_names = [img_file.split('.')[0] for img_file in files]
        batch_imgs.append(img)
        batch_img_names.append(img_names)
    files = img_files[batch_size*(num):len(img_files)]
    if(len(files)!=0):
        img = [cv2.imread(os.path.join(path, img_file)) for img_file in files]
        img_names = [img_file.split('.')[0] for img_file in files]
        batch_imgs.append(img)
        batch_img_names.append(img_names)
    return batch_imgs,batch_img_names


def get_img(ori_imgs,config):
    imgs = []
    scales = []
    for ori_img in ori_imgs:
        img,scale = resize_image_batch(ori_img,config['base']['algorithm'],config['testload']['test_size'],add_padding = config['testload']['add_padding'])
        img = Image.fromarray(img).convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img).unsqueeze(0)
        imgs.append(img)
        scales.append(scale)
    return torch.cat(imgs,0),scales

class TestProgram():
    def __init__(self,config):
        super(TestProgram,self).__init__()
        self.config = config
        if(config['infer']['trt_path'] is not None):
            engine = build_engine(config['infer']['onnx_path'],config['infer']['trt_path'],config['testload']['batch_size'])
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(engine)
            self.context = engine.create_execution_context()
            self.infer_type = 'trt_infer'
            
        elif(config['infer']['onnx_path'] is not None):
            self.model = onnxruntime.InferenceSession(config['infer']['onnx_path'])
            self.infer_type = 'onnx_infer'
        else:
            model = create_module(config['architectures']['model_function'])(config)
            model = load_model(model,config['infer']['model_path'])
            if torch.cuda.is_available():
                model = model.cuda()
            self.model = model
            self.model.eval()
            self.infer_type = 'torch_infer'
        
        img_process = create_module(config['postprocess']['function'])(config)
        self.img_process = img_process
        

    def infer_img(self,ori_imgs):
        img,scales = get_img(ori_imgs,self.config)
        if(self.infer_type == 'torch_infer'):
            if torch.cuda.is_available():
                img = img.cuda()   
            with torch.no_grad():
                out = self.model(img)
        elif(self.infer_type == 'onnx_infer'):
            out = self.model.run(['out'], {'input': img.numpy()})
            out = torch.Tensor(out[0])
        else:
            self.inputs[0].host = img.numpy()
            output = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream,batch_size=int(self.config['testload']['batch_size']))
            test_size = self.config['testload']['test_size']
            if self.config['base']['algorithm']=='DB':
                out = output[0].reshape(int(args.batch_size),1,test_size,test_size)
            elif self.config['base']['algorithm']=='PAN':
                out = output[0].reshape(int(args.batch_size),6,test_size,test_size)
            elif self.config['base']['algorithm']=='PSE':
                out = output[0].reshape(int(args.batch_size),7,test_size,test_size)
            
            out = torch.Tensor(out)
        
#         import pdb
#         pdb.set_trace()
        
        if isinstance(out,dict):
            img_num = out['f_score'].shape[0]
        else:
            img_num = out.shape[0]

        if(self.config['base']['algorithm']=='SAST'):
            scales = [(scale[0],scale[1],ori_imgs[i].shape[0],ori_imgs[i].shape[1]) for scale in scales]

        out = create_process_obj(self.config['base']['algorithm'],out)
        bbox_batch, score_batch = self.img_process(out, scales)
        return bbox_batch,score_batch

def InferOneImg(bin,img,image_name,save_path): 
    bbox_batch, score_batch = bin.infer_img(img)
    for i in range(len(bbox_batch)):
        img_show = img[i].copy()
        with open(os.path.join(save_path, 'result_txt', 'res_' + image_name[i] + '.txt'), 'w+', encoding='utf-8') as fid_res:
            bboxes = bbox_batch[i]
            for bbox in bboxes:
                bbox = bbox.reshape(-1, 2).astype(np.int)
                img_show = cv2.drawContours(img_show, [bbox], -1, (0, 255, 0), 1)
                bbox_str = [str(x) for x in bbox.reshape(-1)]
                bbox_str = ','.join(bbox_str) + '\n'
                fid_res.write(bbox_str)
        cv2.imwrite(os.path.join(save_path, 'result_img', image_name[i] + '.jpg'), img_show)

def InferImage(config):
    path = config['infer']['path']
    save_path = config['infer']['save_path']
    test_bin = TestProgram(config)
    create_dir(save_path)
    create_dir(os.path.join(save_path,'result_img'))
    create_dir(os.path.join(save_path,'result_txt'))
    if os.path.isdir(path):
        files = os.listdir(path)
        batch_imgs,batch_img_names = get_batch_files(path,files,batch_size=config['testload']['batch_size'])
#         print(batch_img_names)
        bar = tqdm(total=len(batch_imgs))
        for i in range(len(batch_imgs)):
            bar.update(1)
            InferOneImg(test_bin, batch_imgs[i],batch_img_names[i], save_path)
        bar.close()

    else:
        image_name = path.split('/')[-1].split('.')[0]
        img = cv2.imread(path)
        InferOneImg(test_bin, [img], [image_name], save_path)


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
    parser.add_argument('--onnx_path', nargs='?', type=str, default=None)
    parser.add_argument('--trt_path', nargs='?', type=str, default=None)
    parser.add_argument('--img_path', nargs='?', type=str, default=None)
    parser.add_argument('--result_save_path', nargs='?', type=str, default=None)
    parser.add_argument('--max_size', nargs='?', type=int, default=None)
    parser.add_argument('--batch_size', nargs='?', type=int, default=None)
    parser.add_argument('--add_padding', action='store_true', default=False)
    args = parser.parse_args()
    config = config_load(args)
    InferImage(config)