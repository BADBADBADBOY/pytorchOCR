"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: SASTProcess.py
@time: 2020/8/23 14:16

"""

import math
import cv2
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from .transform_img import Random_Augment
from ptocr.utils.util_function import resize_image


def get_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    return img

class SASTProcessTrain(data.Dataset):
    """
    SAST process function for training
    """

    def __init__(self, config):
        self.TSM = Random_Augment(config['base']['crop_shape'],max_tries=20,
          min_crop_side_ratio=config['trainload']['min_crop_side_ratio'])
        self.img_set_dir = config['trainload']['train_file']
        image_shape = config['base']['crop_shape']
        self.input_size = image_shape[0]
        self.min_text_size = config['trainload']['min_text_size']
        self.max_text_size = image_shape[0]

        self.img_files = []
        self.gt_files = []
        with open(self.img_set_dir, 'r') as fid:
            lines = fid.readlines()
            for line in lines:
                img_file = line.strip('\n').split('\t')[0]
                gt_file = line.strip('\n').split('\t')[1]
                self.img_files.append(img_file)
                self.gt_files.append(gt_file)

    def adjust_point(self, poly):
        """
        adjust point order.
        """
        point_num = poly.shape[0]
        if point_num == 4:
            len_1 = np.linalg.norm(poly[0] - poly[1])
            len_2 = np.linalg.norm(poly[1] - poly[2])
            len_3 = np.linalg.norm(poly[2] - poly[3])
            len_4 = np.linalg.norm(poly[3] - poly[0])

            if (len_1 + len_3) * 1.5 < (len_2 + len_4):
                poly = poly[[1, 2, 3, 0], :]

        elif point_num > 4:
            vector_1 = poly[0] - poly[1]
            vector_2 = poly[1] - poly[2]
            cos_theta = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2) + 1e-6)
            theta = np.arccos(np.round(cos_theta, decimals=4))

            if abs(theta) > (70 / 180 * math.pi):
                index = list(range(1, point_num)) + [0]
                poly = poly[np.array(index), :]
        return poly

    def gen_min_area_quad_from_poly(self, poly):
        """
        Generate min area quad from poly.
        """
        point_num = poly.shape[0]
        min_area_quad = np.zeros((4, 2), dtype=np.float32)
        if point_num == 4:
            min_area_quad = poly
            center_point = np.sum(poly, axis=0) / 4
        else:
            rect = cv2.minAreaRect(poly.astype(np.int32))  # (center (x,y), (width, height), angle of rotation)
            center_point = rect[0]
            box = np.array(cv2.boxPoints(rect))

            first_point_idx = 0
            min_dist = 1e4
            for i in range(4):
                dist = np.linalg.norm(box[(i + 0) % 4] - poly[0]) + \
                       np.linalg.norm(box[(i + 1) % 4] - poly[point_num // 2 - 1]) + \
                       np.linalg.norm(box[(i + 2) % 4] - poly[point_num // 2]) + \
                       np.linalg.norm(box[(i + 3) % 4] - poly[-1])
                if dist < min_dist:
                    min_dist = dist
                    first_point_idx = i

            for i in range(4):
                min_area_quad[i] = box[(first_point_idx + i) % 4]

        return min_area_quad, center_point

    def shrink_quad_along_width(self, quad, begin_width_ratio=0., end_width_ratio=1.):
        """
        Generate shrink_quad_along_width.
        """
        ratio_pair = np.array([[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
        p0_1 = quad[0] + (quad[1] - quad[0]) * ratio_pair
        p3_2 = quad[3] + (quad[2] - quad[3]) * ratio_pair
        return np.array([p0_1[0], p0_1[1], p3_2[1], p3_2[0]])

    def shrink_poly_along_width(self, quads, shrink_ratio_of_width, expand_height_ratio=1.0):
        """
        shrink poly with given length.
        """
        upper_edge_list = []

        def get_cut_info(edge_len_list, cut_len):
            for idx, edge_len in enumerate(edge_len_list):
                cut_len -= edge_len
                if cut_len <= 0.000001:
                    ratio = (cut_len + edge_len_list[idx]) / edge_len_list[idx]
                    return idx, ratio

        for quad in quads:
            upper_edge_len = np.linalg.norm(quad[0] - quad[1])
            upper_edge_list.append(upper_edge_len)

        # length of left edge and right edge.
        left_length = np.linalg.norm(quads[0][0] - quads[0][3]) * expand_height_ratio
        right_length = np.linalg.norm(quads[-1][1] - quads[-1][2]) * expand_height_ratio

        shrink_length = min(left_length, right_length, sum(upper_edge_list)) * shrink_ratio_of_width
        # shrinking length
        upper_len_left = shrink_length
        upper_len_right = sum(upper_edge_list) - shrink_length

        left_idx, left_ratio = get_cut_info(upper_edge_list, upper_len_left)
        left_quad = self.shrink_quad_along_width(quads[left_idx], begin_width_ratio=left_ratio, end_width_ratio=1)
        right_idx, right_ratio = get_cut_info(upper_edge_list, upper_len_right)
        right_quad = self.shrink_quad_along_width(quads[right_idx], begin_width_ratio=0, end_width_ratio=right_ratio)

        out_quad_list = []
        if left_idx == right_idx:
            out_quad_list.append([left_quad[0], right_quad[1], right_quad[2], left_quad[3]])
        else:
            out_quad_list.append(left_quad)
            for idx in range(left_idx + 1, right_idx):
                out_quad_list.append(quads[idx])
            out_quad_list.append(right_quad)

        return np.array(out_quad_list), list(range(left_idx, right_idx + 1))

    def vector_angle(self, A, B):
        """
        Calculate the angle between vector AB and x-axis positive direction.
        """
        AB = np.array([B[1] - A[1], B[0] - A[0]])
        return np.arctan2(*AB)

    def theta_line_cross_point(self, theta, point):
        """
        Calculate the line through given point and angle in ax + by + c =0 form.
        """
        x, y = point
        cos = np.cos(theta)
        sin = np.sin(theta)
        return [sin, -cos, cos * y - sin * x]

    def line_cross_two_point(self, A, B):
        """
        Calculate the line through given point A and B in ax + by + c =0 form.
        """
        angle = self.vector_angle(A, B)
        return self.theta_line_cross_point(angle, A)

    def average_angle(self, poly):
        """
        Calculate the average angle between left and right edge in given poly.
        """
        p0, p1, p2, p3 = poly
        angle30 = self.vector_angle(p3, p0)
        angle21 = self.vector_angle(p2, p1)
        return (angle30 + angle21) / 2

    def line_cross_point(self, line1, line2):
        """
        line1 and line2 in  0=ax+by+c form, compute the cross point of line1 and line2
        """
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        d = a1 * b2 - a2 * b1

        if d == 0:
            # print("line1", line1)
            # print("line2", line2)
            print('Cross point does not exist')
            return np.array([0, 0], dtype=np.float32)
        else:
            x = (b1 * c2 - b2 * c1) / d
            y = (a2 * c1 - a1 * c2) / d

        return np.array([x, y], dtype=np.float32)

    def quad2tcl(self, poly, ratio):
        """
        Generate center line by poly clock-wise point. (4, 2)
        """
        ratio_pair = np.array([[0.5 - ratio / 2], [0.5 + ratio / 2]], dtype=np.float32)
        p0_3 = poly[0] + (poly[3] - poly[0]) * ratio_pair
        p1_2 = poly[1] + (poly[2] - poly[1]) * ratio_pair
        return np.array([p0_3[0], p1_2[0], p1_2[1], p0_3[1]])

    def poly2tcl(self, poly, ratio):
        """
        Generate center line by poly clock-wise point.
        """
        ratio_pair = np.array([[0.5 - ratio / 2], [0.5 + ratio / 2]], dtype=np.float32)
        tcl_poly = np.zeros_like(poly)
        point_num = poly.shape[0]

        for idx in range(point_num // 2):
            point_pair = poly[idx] + (poly[point_num - 1 - idx] - poly[idx]) * ratio_pair
            tcl_poly[idx] = point_pair[0]
            tcl_poly[point_num - 1 - idx] = point_pair[1]
        return tcl_poly

    def gen_quad_tbo(self, quad, tcl_mask, tbo_map):
        """
        Generate tbo_map for give quad.
        """
        # upper and lower line function: ax + by + c = 0;
        up_line = self.line_cross_two_point(quad[0], quad[1])
        lower_line = self.line_cross_two_point(quad[3], quad[2])

        quad_h = 0.5 * (np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[1] - quad[2]))
        quad_w = 0.5 * (np.linalg.norm(quad[0] - quad[1]) + np.linalg.norm(quad[2] - quad[3]))

        # average angle of left and right line.
        angle = self.average_angle(quad)

        xy_in_poly = np.argwhere(tcl_mask == 1)
        for y, x in xy_in_poly:
            point = (x, y)
            line = self.theta_line_cross_point(angle, point)
            cross_point_upper = self.line_cross_point(up_line, line)
            cross_point_lower = self.line_cross_point(lower_line, line)
            ##FIX, offset reverse
            upper_offset_x, upper_offset_y = cross_point_upper - point
            lower_offset_x, lower_offset_y = cross_point_lower - point
            tbo_map[y, x, 0] = upper_offset_y
            tbo_map[y, x, 1] = upper_offset_x
            tbo_map[y, x, 2] = lower_offset_y
            tbo_map[y, x, 3] = lower_offset_x
            tbo_map[y, x, 4] = 1.0 / max(min(quad_h, quad_w), 1.0) * 2
        return tbo_map

    def poly2quads(self, poly):
        """
        Split poly into quads.
        """
        quad_list = []
        point_num = poly.shape[0]

        # point pair
        point_pair_list = []
        for idx in range(point_num // 2):
            point_pair = [poly[idx], poly[point_num - 1 - idx]]
            point_pair_list.append(point_pair)

        quad_num = point_num // 2 - 1
        for idx in range(quad_num):
            # reshape and adjust to clock-wise
            quad_list.append((np.array(point_pair_list)[[idx, idx + 1]]).reshape(4, 2)[[0, 2, 3, 1]])

        return np.array(quad_list)

    def generate_tcl_label(self, hw, polys, tags, ds_ratio,
                           tcl_ratio=0.3, shrink_ratio_of_width=0.15):
        """
        Generate polygon.
        """
        h, w = hw
        h, w = int(h * ds_ratio), int(w * ds_ratio)
        polys = polys * ds_ratio

        score_map = np.zeros((h, w,), dtype=np.float32)
        tbo_map = np.zeros((h, w, 5), dtype=np.float32)
        training_mask = np.ones((h, w,), dtype=np.float32)
        direction_map = np.ones((h, w, 3)) * np.array([0, 0, 1]).reshape([1, 1, 3]).astype(np.float32)

        for poly_idx, poly_tag in enumerate(zip(polys, tags)):
            poly = poly_tag[0]
            tag = poly_tag[1]

            # generate min_area_quad
            min_area_quad, center_point = self.gen_min_area_quad_from_poly(poly)
            min_area_quad_h = 0.5 * (np.linalg.norm(min_area_quad[0] - min_area_quad[3]) +
                                     np.linalg.norm(min_area_quad[1] - min_area_quad[2]))
            min_area_quad_w = 0.5 * (np.linalg.norm(min_area_quad[0] - min_area_quad[1]) +
                                     np.linalg.norm(min_area_quad[2] - min_area_quad[3]))

            if min(min_area_quad_h, min_area_quad_w) < self.min_text_size * ds_ratio \
                    or min(min_area_quad_h, min_area_quad_w) > self.max_text_size * ds_ratio:
                continue

            if tag:
                # continue
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0.15)
            else:
                tcl_poly = self.poly2tcl(poly, tcl_ratio)
                tcl_quads = self.poly2quads(tcl_poly)
                poly_quads = self.poly2quads(poly)
                # stcl map
                stcl_quads, quad_index = self.shrink_poly_along_width(tcl_quads,
                                                  shrink_ratio_of_width=shrink_ratio_of_width,
                                                  expand_height_ratio=1.0 / tcl_ratio)
                # generate tcl map
                cv2.fillPoly(score_map, np.round(stcl_quads).astype(np.int32), 1.0)

                # generate tbo map
                for idx, quad in enumerate(stcl_quads):
                    quad_mask = np.zeros((h, w), dtype=np.float32)
                    quad_mask = cv2.fillPoly(quad_mask, np.round(quad[np.newaxis, :, :]).astype(np.int32), 1.0)
                    tbo_map = self.gen_quad_tbo(poly_quads[quad_index[idx]], quad_mask, tbo_map)
        return score_map, tbo_map, training_mask

    def generate_tvo_and_tco(self, hw, polys, tags, tcl_ratio=0.3, ds_ratio=0.25):
        """
        Generate tcl map, tvo map and tbo map.
        """
        h, w = hw
        h, w = int(h * ds_ratio), int(w * ds_ratio)
        polys = polys * ds_ratio
        poly_mask = np.zeros((h, w), dtype=np.float32)

        tvo_map = np.ones((9, h, w), dtype=np.float32)
        tvo_map[0:-1:2] = np.tile(np.arange(0, w), (h, 1))
        tvo_map[1:-1:2] = np.tile(np.arange(0, w), (h, 1)).T
        poly_tv_xy_map = np.zeros((8, h, w), dtype=np.float32)

        # tco map
        tco_map = np.ones((3, h, w), dtype=np.float32)
        tco_map[0] = np.tile(np.arange(0, w), (h, 1))
        tco_map[1] = np.tile(np.arange(0, w), (h, 1)).T
        poly_tc_xy_map = np.zeros((2, h, w), dtype=np.float32)

        poly_short_edge_map = np.ones((h, w), dtype=np.float32)

        for poly, poly_tag in zip(polys, tags):

            if poly_tag == True:
                continue

            # adjust point order for vertical poly
            poly = self.adjust_point(poly)

            # generate min_area_quad
            min_area_quad, center_point = self.gen_min_area_quad_from_poly(poly)
            min_area_quad_h = 0.5 * (np.linalg.norm(min_area_quad[0] - min_area_quad[3]) +
                                     np.linalg.norm(min_area_quad[1] - min_area_quad[2]))
            min_area_quad_w = 0.5 * (np.linalg.norm(min_area_quad[0] - min_area_quad[1]) +
                                     np.linalg.norm(min_area_quad[2] - min_area_quad[3]))

            # generate tcl map and text, 128 * 128
            tcl_poly = self.poly2tcl(poly, tcl_ratio)

            # generate poly_tv_xy_map
            for idx in range(4):
                cv2.fillPoly(poly_tv_xy_map[2 * idx],
                             np.round(tcl_poly[np.newaxis, :, :]).astype(np.int32),
                             float(min(max(min_area_quad[idx, 0], 0), w)))
                cv2.fillPoly(poly_tv_xy_map[2 * idx + 1],
                             np.round(tcl_poly[np.newaxis, :, :]).astype(np.int32),
                             float(min(max(min_area_quad[idx, 1], 0), h)))

            # generate poly_tc_xy_map
            for idx in range(2):
                cv2.fillPoly(poly_tc_xy_map[idx],
                             np.round(tcl_poly[np.newaxis, :, :]).astype(np.int32), float(center_point[idx]))

            # generate poly_short_edge_map
            cv2.fillPoly(poly_short_edge_map,
                         np.round(tcl_poly[np.newaxis, :, :]).astype(np.int32),
                         float(max(min(min_area_quad_h, min_area_quad_w), 1.0)))

            # generate poly_mask and training_mask
            cv2.fillPoly(poly_mask, np.round(tcl_poly[np.newaxis, :, :]).astype(np.int32), 1)

        tvo_map *= poly_mask
        tvo_map[:8] -= poly_tv_xy_map
        tvo_map[-1] /= poly_short_edge_map
        tvo_map = tvo_map.transpose((1, 2, 0))

        tco_map *= poly_mask
        tco_map[:2] -= poly_tc_xy_map
        tco_map[-1] /= poly_short_edge_map
        tco_map = tco_map.transpose((1, 2, 0))

        return tvo_map, tco_map

    def get_bboxes(self, gt_path):
        polys = []
        tags = []
        with open(gt_path, 'r', encoding='utf-8') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.replace('\ufeff', '').replace('\xef\xbb\xbf', '')
                gt = line.split(',')
                if "###" in gt[-1]:
                    tags.append(True)
                else:
                    tags.append(False)
                # box = [int(gt[i]) for i in range(len(gt)//2*2)]
                box = [int(gt[i]) for i in range(8)]
                polys.append(box)
        return np.array(polys), tags

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        img = get_img(self.img_files[index])
        polys, dontcare = self.get_bboxes(self.gt_files[index])

        img, polys = self.TSM.random_scale(img, polys, self.input_size)
        img, polys = self.TSM.random_rotate(img, polys)
        img, polys = self.TSM.random_flip(img, polys)
        img, text_polys, text_tags = self.TSM.random_crop_db(img, polys, dontcare)
        text_polys = np.array(text_polys)

        score_map, border_map, training_mask = self.generate_tcl_label((self.input_size, self.input_size),
                                                                       text_polys, text_tags, 0.25)

        # SAST head
        tvo_map, tco_map = self.generate_tvo_and_tco((self.input_size, self.input_size), text_polys, text_tags,
                                                     tcl_ratio=0.3, ds_ratio=0.25)
        
         # add gaussian blur
        if np.random.rand() < 0.1 * 0.5:
            ks = np.random.permutation(5)[0] + 1
            ks = int(ks / 2) * 2 + 1
            img = cv2.GaussianBlur(img, ksize=(ks, ks), sigmaX=0, sigmaY=0)
        # add brighter
        if np.random.rand() < 0.1 * 0.5:
            img = img * (1.0 + np.random.rand() * 0.5)
            img = np.clip(img, 0.0, 255.0)
        # add darker
        if np.random.rand() < 0.1 * 0.5:
            img = img * (1.0 - np.random.rand() * 0.5)
            img = np.clip(img, 0.0, 255.0)

        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
        img.save('img.jpg')
        
        cv2.imwrite('score.jpg',np.array(score_map)*255)
        
        
        img = self.TSM.normalize_img(img)

        score_map = torch.Tensor(score_map[np.newaxis, :, :])
        border_map = torch.Tensor(border_map.transpose((2, 0, 1)))
        training_mask = torch.Tensor(training_mask[np.newaxis, :, :])
        tvo_map = torch.Tensor(tvo_map.transpose((2, 0, 1)))
        tco_map = torch.Tensor(tco_map.transpose((2, 0, 1)))

        return img, score_map, border_map, training_mask, tvo_map, tco_map



class SASTProcessTest(data.Dataset):
    """
    SAST process function for test
    """

    def __init__(self, config):
        super(SASTProcessTest, self).__init__()
        self.img_list = self.get_img_files(config['testload']['test_file'])
        self.TSM = Random_Augment(config['base']['crop_shape'])
        self.test_size = config['testload']['test_size']
        self.config = config

    def get_img_files(self, test_txt_file):
        img_list = []
        with open(test_txt_file, 'r', encoding='utf-8') as fid:
            lines = fid.readlines()
            for line in lines:
                line = line.strip('\n')
                img_list.append(line)
        return img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        ori_img = get_img(self.img_list[index])
        img = resize_image(ori_img, self.config['base']['algorithm'], self.test_size,self.config['testload']['stride'])
        img = Image.fromarray(img).convert('RGB')
        img = self.TSM.normalize_img(img)
        return img, ori_img
