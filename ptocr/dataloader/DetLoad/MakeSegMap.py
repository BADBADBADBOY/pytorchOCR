#-*- coding:utf-8 _*-
"""
@author:fxw
@file: MakeSegMap.py
@time: 2020/08/11
"""
#-*- coding:utf-8 _*-
"""
@author:fxw
@file: MakeSegMap.py
@time: 2020/04/28
"""

import cv2
import pyclipper
from shapely.geometry import Polygon
import numpy as np
import Polygon as plg

class MakeSegMap():
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    def __init__(self, algorithm='DB',min_text_size = 8,shrink_ratio = 0.4,is_training = True):
        self.min_text_size =min_text_size
        self.shrink_ratio = shrink_ratio
        self.is_training = is_training
        self.algorithm = algorithm
        
    def process(self, img,polys,dontcare):
        '''
        requied keys:
            image, polygons, ignore_tags, filename
        adding keys:
            mask
        '''
        h, w = img.shape[:2]
        if self.is_training:
            polys, dontcare = self.validate_polygons(
                polys, dontcare, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        if self.algorithm =='PAN':
            gt_text = np.zeros((h, w), dtype=np.float32)
            gt_text_key = np.zeros((h, w), dtype=np.float32)
            gt_kernel_key = np.zeros((h, w), dtype=np.float32)

        for i in range(len(polys)):
            polygon = polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if dontcare[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                dontcare[i] = True
            else:
                if self.algorithm == 'PAN':
                    cv2.fillPoly(gt_text, [polygon.astype(np.int32)], 1)
                    cv2.fillPoly(gt_text_key, [polygon.astype(np.int32)], i + 1)
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * \
                           (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polys[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(
                        np.int32)[np.newaxis, :, :], 0)
                    dontcare[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                if self.algorithm == 'PAN':
                    cv2.fillPoly(gt_kernel_key, [shrinked.astype(np.int32)], i + 1)
        if self.algorithm == 'PAN':
            return img,gt_text,gt_text_key,gt,gt_kernel_key,mask
        return img,gt,mask
    
    def process_mul(self, img,polys,classes,dontcare):
        '''
        requied keys:
            image, polygons, ignore_tags, filename
        adding keys:
            mask
        '''
        h, w = img.shape[:2]
        if self.is_training:
            polys, dontcare = self.validate_polygons(
                polys, dontcare, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        gt_class = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        
        if self.algorithm =='PAN':
            gt_text = np.zeros((h, w), dtype=np.float32)
            gt_text_key = np.zeros((h, w), dtype=np.float32)
            gt_kernel_key = np.zeros((h, w), dtype=np.float32)

        for i in range(len(polys)):
            polygon = polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if dontcare[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                dontcare[i] = True
            else:
                if self.algorithm == 'PAN':
                    cv2.fillPoly(gt_text, [polygon.astype(np.int32)], 1)
                    cv2.fillPoly(gt_text_key, [polygon.astype(np.int32)], i + 1)
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * \
                           (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polys[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(
                        np.int32)[np.newaxis, :, :], 0)
                    dontcare[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                cv2.fillPoly(gt_class, polygon.astype(np.int32)[np.newaxis, :, :], 1+classes[i])
                if self.algorithm == 'PAN':
                    cv2.fillPoly(gt_kernel_key, [shrinked.astype(np.int32)], i + 1)
        if self.algorithm == 'PAN':
            return img,gt_text,gt_text_key,gt,gt_kernel_key,mask
        return img,gt,gt_class,mask
    
    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] - polygon[i, 1])

        return edge / 2.


class MakeSegPSE():

    def __init__(self, kernel_num = 7,shrink_ratio = 0.4):
        self.kernel_num = kernel_num
        self.shrink_ratio = shrink_ratio

    def dist(self,a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def perimeter(self,bbox):
        peri = 0.0
        for i in range(bbox.shape[0]):
            peri += self.dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
        return peri

    def shrink(self,bboxes, rate, max_shr=20):
        rate = rate * rate
        shrinked_bboxes = []
        for bbox in bboxes:
            area = plg.Polygon(bbox).area()
            peri = self.perimeter(bbox)

            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox)[0]
            shrinked_bbox = np.array(shrinked_bbox)
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)

        return np.array(shrinked_bboxes)

    def process(self,img, bboxes, tags):

        bboxes = np.array(bboxes).astype(np.int)
        gt_text = np.zeros(img.shape[0:2], dtype='uint8').copy()
        training_mask = np.ones(img.shape[0:2], dtype='uint8').copy()

        if bboxes.shape[0] > 0:
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_text, [bboxes[i]], -1, 1, -1)
                if tags[i]:
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)
        gt_kernels = []
        for i in range(1, self.kernel_num):
            rate = 1.0 - (1.0 - self.shrink_ratio) / (self.kernel_num - 1) * i
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = self.shrink(bboxes, rate)
            for j in range(bboxes.shape[0]):
                cv2.drawContours(gt_kernel, [kernel_bboxes[j]], -1, 1, -1)
            gt_kernels.append(gt_kernel)
        return img, training_mask, gt_text, gt_kernels


