#-*- coding:utf-8 _*-
"""
@author:fxw
@file: DBpostprocess.py
@time: 2020/08/13
"""
import time
import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper
from .dbprocess import cpp_boxes_from_bitmap

class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self, config):
        self.thresh = config['postprocess']['thresh']
        self.box_thresh = config['postprocess']['box_thresh']
        self.max_candidates = config['postprocess']['max_candidates']
        self.is_poly = config['postprocess']['is_poly']
        self.unclip_ratio = config['postprocess']['unclip_ratio']
        self.min_size = config['postprocess']['min_size']

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        pred = pred
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box, unclip_ratio=2):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

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
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, pred, ratio_list):
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        score_batch = []
        for batch_index in range(pred.shape[0]):
            height, width = pred.shape[-2:]
            if (self.is_poly):
                tmp_boxes, tmp_scores = self.polygons_from_bitmap(
                    pred[batch_index], segmentation[batch_index], width, height)

                boxes = []
                score = []
                for k in range(len(tmp_boxes)):
                    if tmp_scores[k] > self.box_thresh:
                        boxes.append(tmp_boxes[k])
                        score.append(tmp_scores[k])
                if len(boxes) > 0:
                    ratio_w, ratio_h = ratio_list[batch_index]
                    for i in range(len(boxes)):
                        boxes[i] = np.array(boxes[i])
                        boxes[i][:, 0] = boxes[i][:, 0] * ratio_w
                        boxes[i][:, 1] = boxes[i][:, 1] * ratio_h

                boxes_batch.append(boxes)
                score_batch.append(score)
            else:

#                 tmp_boxes, tmp_scores = self.boxes_from_bitmap(
#                     pred[batch_index], segmentation[batch_index], width, height)


                tmp_boxes = cpp_boxes_from_bitmap(pred[batch_index], segmentation[batch_index],self.box_thresh,self.unclip_ratio)
                boxes = []
                score = []
                for k in range(len(tmp_boxes)):
#                     if tmp_scores[k] > self.box_thresh:
                    boxes.append(tmp_boxes[k])
#                         score.append(tmp_scores[k])
                if len(boxes) > 0:
                    boxes = np.array(boxes)

                    ratio_w, ratio_h = ratio_list[batch_index]
                    boxes[:, :, 0] = boxes[:, :, 0] * ratio_w
                    boxes[:, :, 1] = boxes[:, :, 1] * ratio_h

                boxes_batch.append(boxes)
                score_batch.append(score)
        return boxes_batch, score_batch
    
class DBPostProcessMul(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self, config):
        self.thresh = config['postprocess']['thresh']
        self.box_thresh = config['postprocess']['box_thresh']
        self.max_candidates = config['postprocess']['max_candidates']
        self.is_poly = config['postprocess']['is_poly']
        self.unclip_ratio = config['postprocess']['unclip_ratio']
        self.min_size = config['postprocess']['min_size']
        
    def polygons_from_bitmap(self, pred,classes, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''

        
        bitmap = _bitmap
        pred = pred
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap*255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            # _, sside = self.get_mini_boxes(contour)
            # if sside < self.min_size:
            #     continue
            score = self.box_score_fast(pred,classes, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue
            
            if points.shape[0] > 2:
                box = self.unclip(points, self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box ,type_class = box.reshape(-1, 2)
            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred,classes, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours, ), dtype=np.float32)
        type_classes = np.zeros((num_contours, ), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score,type_class = self.box_score_fast(pred,classes, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue
            box = self.unclip(points,self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
            type_classes[index] = type_class
        return boxes, scores,type_classes

    def unclip(self, box, unclip_ratio=2):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

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
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap,classes,_box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        classes = classes[ymin:ymax + 1, xmin:xmax + 1]
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0],np.argmax(np.bincount(classes.reshape(-1).astype(np.int32)))

    def __call__(self, pred,pred_class, ratio_list):
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        classes = pred_class[:, 0, :, :]

        boxes_batch = []
        score_batch = []
        type_classes_batch = [] 
        for batch_index in range(pred.shape[0]):
            height, width = pred.shape[-2:]
            if(self.is_poly):
                tmp_boxes, tmp_scores = self.polygons_from_bitmap(
                    pred[batch_index], classes[batch_index],segmentation[batch_index], width, height)

                boxes = []
                score = []
                for k in range(len(tmp_boxes)):
                    if tmp_scores[k] > self.box_thresh:
                        boxes.append(tmp_boxes[k])
                        score.append(tmp_scores[k])
                if len(boxes) > 0:
                    ratio_w, ratio_h = ratio_list[batch_index]
                    for i in range(len(boxes)):
                        boxes[i] = np.array(boxes[i])
                        boxes[i][:, 0] = boxes[i][:, 0] * ratio_w
                        boxes[i][:, 1] = boxes[i][:, 1] * ratio_h

                boxes_batch.append(boxes)
                score_batch.append(score)
            else:
                tmp_boxes, tmp_scores,type_classes = self.boxes_from_bitmap(
                    pred[batch_index], classes[batch_index],segmentation[batch_index], width, height)

                boxes = []
                score = []
                _classes = []
                for k in range(len(tmp_boxes)):
                    if tmp_scores[k] > self.box_thresh:
                        boxes.append(tmp_boxes[k])
                        score.append(tmp_scores[k])
                        _classes.append(type_classes[k])
                if len(boxes) > 0:
                    boxes = np.array(boxes)

                    ratio_w, ratio_h = ratio_list[batch_index]
                    boxes[:, :, 0] = boxes[:, :, 0] * ratio_w
                    boxes[:, :, 1] = boxes[:, :, 1] * ratio_h
                type_classes_batch.append(_classes)
                boxes_batch.append(boxes)
                score_batch.append(score)
        return boxes_batch,score_batch,type_classes_batch