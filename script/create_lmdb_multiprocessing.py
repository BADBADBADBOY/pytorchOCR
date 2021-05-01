import os
import lmdb
import cv2
import numpy as np
import argparse
import shutil
import sys
from tqdm import tqdm 
import time
import six
from PIL import Image
from multiprocessing import Process

def get_dict(char_list):
    char_dict={}
    for item in char_list:
        if item in char_dict.keys():
            char_dict[item]+=1
        else:
            char_dict[item]=1
    return char_dict

def checklmdb(args):
    env = lmdb.open(
            args.out,
            max_readers=2,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
    
    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode('utf-8')))
        print('Check lmdb ok!!!')
        print('lmdb Have {} samples'.format(nSamples))
        print('Print 5 samples:')

        for index in range(1,nSamples+1):
            if index>5:
                break
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))
            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode('utf-8')).decode()
            print(index,label)
    

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False
    
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k,v)

            
def write(imagePathList,labelList,env,start,end):
    cache = {}
    cnt = 1
    bar = tqdm(total=end-start)
    checkValid=True
    lexiconList=None
    for i in range(end-start):
        bar.update(1)
        imagePath = imagePathList[start+i]
        label = labelList[start+i]
        
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % (start+cnt)
        labelKey = 'label-%09d' % (start+cnt)

        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if (start+cnt) % 1000 == 0:
            writeCache(env, cache)
            cache = {}
        cnt += 1
    bar.close()
    writeCache(env, cache)
    
def createDataset(outputPath, imagePathList, labelList, num=1,lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    # If lmdb file already exists, remove it. Or the new data will add to it.
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
        os.makedirs(outputPath)
    else:
        os.makedirs(outputPath)

    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    index = []
    
    if nSamples%num==0:
        step = nSamples//num
        for i in range(num):
            index.append([i*step,(i+1)*step])
    else:
        step = nSamples//num
        for i in range(num):
            index.append([i*step,(i+1)*step])
        index.append([num*step,nSamples])
        
    p_list = []
    if nSamples%num==0:
        for i in range(num):
            p = Process(target=write,args=(imagePathList,labelList,env,index[i][0],index[i][1]))
            p_list.append(p) 
            p.start()
    else:
        for i in range(num+1):
            p = Process(target=write,args=(imagePathList,labelList,env,index[i][0],index[i][1]))
            p_list.append(p)
            p.start() 
    for p in p_list:
        p.join()      
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)


def read_data_from_file(file_path):
    image_path_list = []
    label_list = []
    with open(file_path,'r',encoding='utf-8') as fid:
        lines = fid.readlines()
        for line in lines:
            line = line.replace('\r', '').replace('\n', '').split('\t')
            image_path_list.append(line[0])
            label_list.append(line[1])

    return image_path_list, label_list

def show_demo(demo_number, image_path_list, label_list):
    print ('The first line is the path to image and the second line is the image label')
    print ('###########################################################################')
    for i in range(demo_number):
        print ('image: %s\nlabel: %s\n' % (image_path_list[i], label_list[i]))
    print ('###########################################################################')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type = str, required = True, help = 'lmdb data output path')
    parser.add_argument('--file', type = str,required = True, help = 'path to file which contains the image path and label')
    parser.add_argument('--num_process', type = int, required = True, help = 'num_process to do')
    args = parser.parse_args()

    if args.file is not None:
        image_path_list, label_list = read_data_from_file(args.file)
        show_demo(2, image_path_list, label_list)
        s_time = time.time()
        createDataset(args.out, image_path_list, label_list,num=args.num_process)
        print('cost_time:'+str(time.time()-s_time)+'s')
    else:
        print ('Please use --file to assign the input. Use -h to see more.')
        sys.exit()
    print('lmdb generate ok!!!!')
    checklmdb(args)
        




                
            