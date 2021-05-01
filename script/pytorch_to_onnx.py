import sys
sys.path.append('./')
import yaml
import math 
import argparse
import torch.nn as nn
import torch
import cv2
import numpy as np
import onnx
import time
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms
from ptocr.utils.util_function import create_module,load_model


crnn_max_length = 280

def transform_img_shape(img,img_shape):
    img= np.array(img)
    H,W = img_shape
    h,w = img.shape[:2]
    new_w = int((h/H)*w)
    if(new_w > W):
        img = cv2.resize(img,(W,H))
    else:
        img = cv2.resize(img,(new_w,H))
        img = cv2.copyMakeBorder(img, 0, 0,
                      0, W-new_w, cv2.BORDER_CONSTANT, value=(0,0,0))
    return img


def resize_image(img,algorithm,side_len=1536,add_padding=True):
    
    if algorithm == 'CRNN':
        img = transform_img_shape(img,[32,crnn_max_length])
        return img
    
    if(algorithm=='SAST'):
        stride = 128
    else:
        stride = 32
    height, width, _ = img.shape
    flag = None
    if height > width:
        flag = True
        new_height = side_len
        new_width = int(math.ceil(new_height / height * width / stride) * stride)
    else:
        flag = False
        new_width = side_len
        new_height = int(math.ceil(new_width / width * height / stride) * stride)
    resized_img = cv2.resize(img, (new_width, new_height))
    if add_padding is True:
        if flag:
            padded_image = cv2.copyMakeBorder(resized_img, 0, 0,
                          0, side_len-new_width, cv2.BORDER_CONSTANT, value=(0,0,0))
        else:
            padded_image = cv2.copyMakeBorder(resized_img, 0, side_len-new_height,
                          0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    else:
        return resized_img
    return padded_image


def gen_onnx(args):
    stream = open(args.config, 'r', encoding='utf-8')
    config = yaml.load(stream,Loader=yaml.FullLoader)

    model = create_module(config['architectures']['model_function'])(config)

    model = model.cuda()
    model = load_model(model,args.model_path)
    model.eval()

    print('load model ok.....')
    

    img = cv2.imread(args.img_path)
    img = resize_image(img,args.algorithm,side_len=args.max_size,add_padding=args.add_padding)
    w,h = img.shape[:2]
    print(w,h)
    img1 = Image.fromarray(img).convert('RGB')
    img1 = transforms.ToTensor()(img1)
    img1 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img1).unsqueeze(0).cuda()

    s = time.time()
    with torch.no_grad():
        out = model(img1)
    print('cost time:',time.time()-s)
    if isinstance(out,dict):
        out = out['f_score']
    
    cv2.imwrite('./onnx/ori_output.jpg',out[0,0].cpu().detach().numpy()*255)

    output_onnx = args.save_path
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input"]
    # output_names = ["hm" , "wh"  , "reg"]
    output_names = ["out"]
    inputs = torch.randn(args.batch_size, 3,w,h).cuda()
    torch_out = torch.onnx._export(model, inputs, output_onnx, export_params=True, verbose=False,do_constant_folding=False,keep_initializers_as_inputs=True,
                                   input_names=input_names, output_names=output_names)


    onnx_path = args.save_path
    session = onnxruntime.InferenceSession(onnx_path)
    # session.get_modelmeta()
    # input_name = session.get_inputs()[0].name
    # output_name = session.get_outputs()[0].name

    image = img / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    s = time.time()
    preds = session.run(['out'], {'input': image})
    preds = preds[0]
    print(time.time()-s)
    if isinstance(preds,dict):
        preds = preds['f_score']
    cv2.imwrite('./onnx/onnx_output.jpg',preds[0,0]*255)

    print('error_distance:',np.abs((out.cpu().detach().numpy()-preds)).mean())
    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--model_path', nargs='?', type=str, default=None)
    parser.add_argument('--img_path', nargs='?', type=str, default=None)
    parser.add_argument('--save_path', nargs='?', type=str, default=None)
    parser.add_argument('--batch_size', nargs='?', type=int, default=None)
    parser.add_argument('--max_size', nargs='?', type=int, default=None)
    parser.add_argument('--algorithm', nargs='?', type=str, default=None)
    parser.add_argument('--add_padding', action='store_true', default=False)
    
    args = parser.parse_args()
    gen_onnx(args)
    