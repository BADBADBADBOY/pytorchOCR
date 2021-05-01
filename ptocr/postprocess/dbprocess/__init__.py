
import os
import cv2
import torch
import time
import subprocess
import numpy as np



BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))

    
def cpp_boxes_from_bitmap(pred,bitmap,box_thresh=0.6,det_db_unclip_ratio=2.0):
    
    from .cppdbprocess import db_cpp
    bitmap = bitmap.astype(np.uint8)
    bboxes = db_cpp(pred,bitmap,box_thresh,det_db_unclip_ratio)
        
    return bboxes
    


    
    
    



