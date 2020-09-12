#-*- coding:utf-8 _*-
"""
@author:fxw
@file: PSEpostprocess.py
@time: 2020/08/13
"""

import numpy as np
import cv2
import torch
from .piexlmerge import pan

class PANPostProcess(object):

    def __init__(self, config):
        self.min_text_area = config['postprocess']['min_text_area']
        self.is_poly = config['postprocess']['is_poly']
        self.min_score = config['postprocess']['min_score']
        self.config = config

    def polygons_from_bitmap(self, pred):
        
        boxes = []
        scores = []
        pred,label_points,label_values = pan(pred,self.config)
        
        for label_value, label_point in label_points.items():
            if label_value not in label_values:
                continue
            score_i = label_point[0]
            label_point = label_point[2:]
            points = np.array(label_point, dtype=int).reshape(-1, 2)

            if points.shape[0] < self.min_text_area :
                continue

            if score_i < self.min_score:
                continue
                
            binary = np.zeros(pred.shape, dtype='uint8')
            binary[pred == label_value] = 1

            find_out = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if(len(find_out)==2):
                contours, _ = find_out
            else:
                _, contours, _ = find_out
            contour = contours[0]
            # epsilon = 0.01 * cv2.arcLength(contour, True)
            # box = cv2.approxPolyDP(contour, epsilon, True)
            box = contour

            if box.shape[0] <= 2:
                continue
            
            box = box * self.config['postprocess']['scale']
            box = box.astype('int32')
            boxes.append(box[:,0])
            scores.append(score_i)
        return boxes, scores

    def boxes_from_bitmap(self, pred):
        boxes = []
        scores = []
        pred,label_points,label_values = pan(pred,self.config)
        
        for label_value, label_point in label_points.items():
            if label_value not in label_values:
                continue
            score_i = label_point[0]
            label_point = label_point[2:]
            points = np.array(label_point, dtype=int).reshape(-1, 2)

            if points.shape[0] < self.min_text_area :
                continue

            if score_i < self.min_score:
                continue
            box = self.get_mini_boxes(points)*self.config['postprocess']['scale']
            boxes.append(box)
            scores.append(score_i)
        return boxes, scores

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return np.array(box)

    def __call__(self, pred, ratio_list):
        boxes_batch = []
        score_batch = []
        for batch_index in range(pred.shape[0]):
            pred_single = pred[batch_index].unsqueeze(0)
            if (self.is_poly):
                boxes, score = self.polygons_from_bitmap(pred_single)
                new_bboxes =[]
                if len(boxes) > 0:
                    ratio_w, ratio_h = ratio_list[batch_index]
                    for bbox in boxes:
                        bbox[:, 0] = bbox[:, 0] * ratio_w
                        bbox[:, 1] = bbox[:, 1] * ratio_h
                        new_bboxes.append(bbox)

                boxes_batch.append(new_bboxes)
                score_batch.append(score)
            else:
                boxes, score = self.boxes_from_bitmap(pred_single)
                if len(boxes) > 0:
                    boxes = np.array(boxes)
                    ratio_w, ratio_h = ratio_list[batch_index]
                    boxes[:, :, 0] = boxes[:, :, 0] * ratio_w
                    boxes[:, :, 1] = boxes[:, :, 1] * ratio_h

                boxes_batch.append(boxes)
                score_batch.append(score)
        return boxes_batch, score_batch

