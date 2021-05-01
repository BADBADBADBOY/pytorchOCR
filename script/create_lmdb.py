import os
import lmdb
import cv2
import numpy as np
import argparse
import shutil
import sys
from tqdm import tqdm 

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

def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
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
    cnt = 1
    bar = tqdm(total=nSamples)
    for i in range(nSamples):
        bar.update(1)
        imagePath = imagePathList[i]
        label = labelList[i]
        
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
#             print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    bar.close()
    nSamples = cnt-1
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
    parser.add_argument('--file', type = str, help = 'path to file which contains the image path and label')
    args = parser.parse_args()
    
    if args.file is not None:
        image_path_list, label_list = read_data_from_file(args.file)
        createDataset(args.out, image_path_list, label_list)
        show_demo(2, image_path_list, label_list)
   
    else:
        print ('Please use --file to assign the input. Use -h to see more.')
        sys.exit()