#-*- coding:utf-8 _*-
"""
@author:fxw
@file: det_infer.py
@time: 2020/08/20
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import sys
sys.path.append('./')
import cv2
import torch
import yaml
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from ptocr.utils.util_function import create_module,resize_image,resize_image_crnn
from ptocr.utils.util_function import create_process_obj,create_dir,load_model
from get_acc_english import get_test_acc

def is_chinese(check_str):
    flag = False
    for ch in check_str.decode('utf-8'):
        if u'\u4e00' >= ch or ch >= u'\u9fff':
            flag = True
    return flag

def is_number_letter(check_str):
    if(check_str in '0123456789' or check_str in 'abcdefghijklmnopgrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        return True
    else:
        return False




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
        
#         img = resize_image(ori_img,self.congig['base']['algorithm'],32,4)
        img = resize_image_crnn(ori_img)
#         cv2.imwrite('result.jpg',img)
        if(img.shape[0]!=self.congig['base']['img_shape'][0]):
            return ''
        
        img = Image.fromarray(img).convert('RGB')
        if(self.congig['base']['is_gray']):
            img = img.convert('L')
#         img = img.resize((100, 32),Image.ANTIALIAS)
        image_for_show = np.array(img.convert('RGB')).copy()
        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)
        img = img.unsqueeze(0)

        if torch.cuda.is_available():
            img = img.cuda()
            
        with torch.no_grad():
            preds,feau= self.model(img)
            preds = preds.permute(1, 0, 2)
            
        time_step = preds.shape[0]
        image_width = img.shape[3]
        step_ = image_width//time_step
        
        ### 输出相似字结果
#         k_num = 3
#         p,idx = torch.softmax(preds,-1).squeeze().topk(k_num,-1)
#         p = p[idx[:,0]>0].cpu().numpy()
#         for i in range(k_num):
#             index = idx[:,i][idx[:,0]>0]
#             result = np.array(list(self.converter.alphabet))[index.cpu().numpy()-1].tolist()
#             if(i==0):
#                 print('识别结果：',result,p[:,i])
#             else:
#                 print('相似字：',result,p[:,i])

        preds_size = torch.IntTensor([preds.size(0)])   
        _, preds = preds.max(2)
        preds = preds.squeeze(1)
        preds = preds.contiguous().view(-1)

        sim_preds = self.converter.decode(preds.data, preds_size.data, raw=False)
        
#         raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
#         start_step = 0
#         word_po = []
#         for i in range(len(raw_pred)-1):
#             if(raw_pred[i]!='-' and raw_pred[i]!=raw_pred[i+1]):
# #                 image_for_show = cv2.rectangle(image_for_show,(start_step+step_,0),(start_step+step_, img.shape[2]-1),(0,0,255))
#                 start_step+=step_
#                 word_po.append(start_step+step_)
#             else:
#                 start_step+=step_
                
# #         word_po = np.array(word_po)
# #         word_po_flag = np.zeros(len(word_po))
# #         word_po_flag[1:] = word_po[:-1]
# #         word_len = sorted(word_po-word_po_flag)[len(word_po)//2]
#         word_s_e = []
#         for i in range(len(word_po)):
#             if(is_number_letter(sim_preds[i])):
#                 word_len = self.congig['base']['img_shape'][0]//4
#             else:
#                 word_len = self.congig['base']['img_shape'][0]//2
#             word_s_e.append([word_po[i]-word_len,word_po[i]+word_len])
# #             image_for_show = cv2.rectangle(image_for_show,(int(word_po[i]-word_len),1),(int(word_po[i]+word_len), img.shape[2]-1),(255,0,255))
#             image_for_show = cv2.line(image_for_show,(int(word_po[i]),1),(int(word_po[i]), img.shape[2]-1),(255,0,255))
#         cv2.imwrite('result.jpg',image_for_show)
        
        return sim_preds

def InferImage(config):
    path = config['infer']['path']
    save_path = config['infer']['save_path']
    test_bin = TestProgram(config)
    fid = open('re_result.txt','w+',encoding='utf-8')
    if os.path.isdir(path):
        files = os.listdir(path)
        bar = tqdm(total=len(files))
        for file in files:
            print(file)
            bar.update(1)
            image_name = file.split('.')[0]
            img_path = os.path.join(path,file)
            img = cv2.imread(img_path)
            rec_char = test_bin.infer_img(img)
            fid.write(file+'\t'+rec_char+'\n')
            print(rec_char)
        bar.close()

    else:
        image_name = path.split('/')[-1].split('.')[0]
        img = cv2.imread(path)
        rec_char = test_bin.infer_img(img)
        print(rec_char)
        
def InferImageAcc(config):
    root_path = config['infer']['path']
    save_path = config['infer']['save_path']
    test_bin = TestProgram(config)
    acc_fid = open('acc.txt','w+',encoding='utf-8')
    acc_all = 0
    for _dir in os.listdir(root_path):
        path = os.path.join(root_path,_dir,'image')
        gt_file = os.path.join(root_path,_dir,'val.txt')
#         print(gt_file)
        fid = open(_dir+'.txt','w+',encoding='utf-8')
        if os.path.isdir(path):
            files = os.listdir(path)
            bar = tqdm(total=len(files))
            for file in files:
#                 print(file)
                bar.update(1)
                image_name = file.split('.')[0]
                img_path = os.path.join(path,file)
                img = cv2.imread(img_path)
                rec_char = test_bin.infer_img(img)
                fid.write(file+'\t'+rec_char+'\n')
#                 print(rec_char)
            bar.close()

        else:
            image_name = path.split('/')[-1].split('.')[0]
            img = cv2.imread(path)
            rec_char = test_bin.infer_img(img)
            print(rec_char)
        fid.close()
        acc = get_test_acc(_dir+'.txt',gt_file)
        acc_all+=acc
        acc_fid.write(_dir+':'+'\t'+str(acc)+'\n')
    print('mean acc:',acc_all/9.0)   
    acc_fid.close()
    

if __name__ == "__main__":
    
    stream = open('./config/rec_CRNN_mobilev3_small_english_all.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream,Loader=yaml.FullLoader)
    InferImageAcc(config)

#     stream = open('./config/rec_CRNN_resnet_english.yaml', 'r', encoding='utf-8')
#     config = yaml.load(stream,Loader=yaml.FullLoader)
#     InferImage(config)