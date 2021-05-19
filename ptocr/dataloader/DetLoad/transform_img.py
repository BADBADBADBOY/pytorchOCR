#-*- coding:utf-8 _*-
"""
@author:fxw
@file: transform_img.py
@time: 2020/08/11
"""
import cv2
import numpy as np
import imgaug.augmenters as aug_img
import imgaug
import random
import math
import torchvision.transforms as transforms

def solve_polys(polys):
    len_max = 0
    for poly in polys:
        if (len(poly) // 2 > len_max):
            len_max = len(poly) // 2
    new_polys = []
    for poly in polys:
        new_poly = []
        if (len(poly) // 2 < len_max):
            new_poly.extend(poly)
            for i in range(len(poly) // 2, len_max):
                new_poly.extend([poly[0], poly[1]])
        else:
            new_poly = poly
        new_polys.append(new_poly)
    return np.array(new_polys), len_max


def scale_aligned(img, h_scale, w_scale):
    h, w = img.shape[0:2]
    h = int(h * h_scale + 0.5)
    w = int(w * w_scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


class RandomCropData():
    def __init__(self, max_tries=10, min_crop_side_ratio=0.1, crop_size=(640, 640)):
        self.size = crop_size
        self.min_crop_side_ratio = min_crop_side_ratio
        self.max_tries = max_tries

    def process(self, img, polys, dont_care):
        all_care_polys = []
        for i in range(len(dont_care)):
            if (dont_care[i] is False):
                all_care_polys.append(polys[i])
        crop_x, crop_y, crop_w, crop_h = self.crop_area(img, all_care_polys)
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        padimg = np.zeros(
            (self.size[1], self.size[0], img.shape[2]), img.dtype)
        padimg[:h, :w] = cv2.resize(
            img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
        img = padimg

        new_polys = []
        new_dotcare = []
        for i in range(len(polys)):
            poly = polys[i]
            poly = ((np.array(poly) -
                     (crop_x, crop_y)) * scale)
            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                new_polys.append(poly)
                new_dotcare.append(dont_care[i])

        return img, new_polys, new_dotcare
    
    def process_mul(self, img, polys, classes, dont_care):
        all_care_polys = []
        for i in range(len(dont_care)):
            if (dont_care[i] is False):
                all_care_polys.append(polys[i])
        crop_x, crop_y, crop_w, crop_h = self.crop_area(img, all_care_polys)
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        padimg = np.zeros(
            (self.size[1], self.size[0], img.shape[2]), img.dtype)
        padimg[:h, :w] = cv2.resize(
            img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
        img = padimg

        new_polys = []
        new_dotcare = []
        new_classes = []
        
        for i in range(len(polys)):
            poly = polys[i]
            poly = ((np.array(poly) -
                     (crop_x, crop_y)) * scale)
            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                new_polys.append(poly)
                new_dotcare.append(dont_care[i])
                new_classes.append(classes[i])

        return img, new_polys,new_classes, new_dotcare

    def is_poly_in_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, img, polys):
        h, w, _ = img.shape
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h


class Random_Augment():
    def __init__(self,crop_size, max_tries=10, min_crop_side_ratio=0.1):
        super(Random_Augment, self).__init__()
        self.random_crop_data = RandomCropData(crop_size = crop_size, max_tries=max_tries, min_crop_side_ratio=min_crop_side_ratio)
        self.crop_size = crop_size

    def normalize_img(self,img):
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return  img

    def augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints([imgaug.KeypointsOnImage(keypoints, shape=img_shape[:2])])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return np.array(poly)

    def random_rotate(self, img, polys, random_range=[-10,10]):
        angle = np.random.randint(random_range[0], random_range[1])
        aug_bin = aug_img.Sequential([aug_img.Affine(rotate=angle)])
        img = aug_bin.augment_image(img)
        new_polys = []
        for poly in polys:
            poly = self.augment_poly(aug_bin, img.shape, poly)
            poly = np.maximum(poly, 0)
            new_polys.append(poly)
        return img, new_polys

    def random_scale(self, img, polys, min_size):
        polys, len_max = solve_polys(polys)
        h, w = img.shape[0:2]
        new_polys = []
        for poly in polys:
            poly = np.asarray(poly)
            poly = poly / ([w * 1.0, h * 1.0] * len_max)
            new_polys.append(poly)
        new_polys = np.array(new_polys)
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
            img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        h, w = img.shape[0:2]
        random_scale = np.array([0.5,1.0,2.0,3.0])
        scale = np.random.choice(random_scale)
        if min(h, w) * scale <= min_size:
            scale = (min_size + 10) * 1.0 / min(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        new_polys = np.reshape(new_polys * ([img.shape[1], img.shape[0]] * len_max),
                               (new_polys.shape[0], polys.shape[1] // 2, 2))
        return img, new_polys
    
    def random_scale_pan(self, img, polys, short_size=736):
        
        polys, len_max = solve_polys(polys)
        h, w = img.shape[0:2]
        new_polys = []
        for poly in polys:
            poly = np.asarray(poly)
            poly = poly / ([w * 1.0, h * 1.0] * len_max)
            new_polys.append(poly)
        new_polys = np.array(new_polys)
        
        scale = np.random.choice(np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]))
        scale = (scale * short_size) / min(h, w)

        aspect = np.random.choice(np.array([0.9, 0.95, 1.0, 1.05, 1.1]))
        h_scale = scale * math.sqrt(aspect)
        w_scale = scale / math.sqrt(aspect)

        img = scale_aligned(img, h_scale, w_scale)
        
        new_polys = np.reshape(new_polys * ([img.shape[1], img.shape[0]] * len_max),
                               (new_polys.shape[0], polys.shape[1] // 2, 2))
        return img, new_polys
        
    def random_flip(self, img, polys):
        if (np.random.rand(1)[0] > 0.5):
            aug_bin = aug_img.Sequential([aug_img.Fliplr((1))])
            img = aug_bin.augment_image(img)
            new_polys = []
            for poly in polys:
                poly = self.augment_poly(aug_bin, img.shape, poly)
                poly = np.maximum(poly, 0)
                new_polys.append(poly)
        else:
            new_polys = polys
        return img, new_polys

    def random_crop_db(self, img, polys, dont_care):
        img, new_polys, new_dotcare = self.random_crop_data.process(img, polys, dont_care)
        return img, new_polys, new_dotcare
    
    def random_crop_db_mul(self, img, polys,classes, dont_care):
        img, new_polys, new_classes,new_dotcare = self.random_crop_data.process_mul(img, polys, classes,dont_care)
        return img, new_polys,new_classes, new_dotcare

    def random_crop_pse(self, imgs, ):
        h, w = imgs[0].shape[0:2]
        th, tw = self.crop_size
        if w == tw and h == th:
            return imgs

        if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
            tl = np.min(np.where(imgs[1] > 0), axis=1) - self.crop_size
            tl[tl < 0] = 0
            br = np.max(np.where(imgs[1] > 0), axis=1) - self.crop_size
            br[br < 0] = 0
            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)

            i = random.randint(tl[0], br[0])
            j = random.randint(tl[1], br[1])
        else:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)

        # return i, j, th, tw
        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
            else:
                imgs[idx] = imgs[idx][i:i + th, j:j + tw]
        return imgs
    
    def random_crop_pan(self,imgs,):
        h, w = imgs[0].shape[0:2]
        t_w, t_h = self.crop_size
        p_w, p_h = self.crop_size
        if w == t_w and h == t_h:
            return imgs

        t_h = t_h if t_h < h else h
        t_w = t_w if t_w < w else w

        if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
            # make sure to crop the text region
            tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
            tl[tl < 0] = 0
            br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
            br[br < 0] = 0
            br[0] = min(br[0], h - t_h)
            br[1] = min(br[1], w - t_w)

            i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
            j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            i = random.randint(0, h - t_h) if h - t_h > 0 else 0
            j = random.randint(0, w - t_w) if w - t_w > 0 else 0

        n_imgs = []
        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                s3_length = int(imgs[idx].shape[-1])
                img = imgs[idx][i:i + t_h, j:j + t_w, :]
                img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                           value=tuple(0 for i in range(s3_length)))
            else:
                img = imgs[idx][i:i + t_h, j:j + t_w]
                img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
            n_imgs.append(img_p)
        return n_imgs


