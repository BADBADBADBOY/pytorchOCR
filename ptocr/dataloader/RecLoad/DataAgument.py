# -*- coding:utf-8 _*-
"""
@author:fxw
@file: DataAgument.py
@time: 2019/06/06
"""
import cv2,re
import numpy as np
from skimage.util import random_noise
import os
from math import floor
from PIL import Image
import numpy as np
import random
import cv2


def Add_Padding(image, top, bottom, left, right, color):
    padded_image = cv2.copyMakeBorder(image, top, bottom,
                                      left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image


def cvtColor(img):
    """
    cvtColor
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * (1 if random.random() > 0.5000001 else -1)
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


class Operation(object):
    def __init__(self, probability):
        self.probability = probability

    def __str__(self):
        return self.__class__.__name__

    def perform_operation(self, images):
        raise RuntimeError("Illegal call to base class.")


class Distort(Operation):
    def __init__(self, probability, grid_width, grid_height, magnitude):
        Operation.__init__(self, probability)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = abs(magnitude)
        # TODO: Implement non-random magnitude.
        self.randomise_magnitude = True

    def perform_operation(self, images):
        w, h = images[0].size

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

        # For loop that generates polygons could be rewritten, but maybe harder to read?
        # polygons = [x1,y1, x1,y2, x2,y2, x2,y1 for x1,y1, x2,y2 in dimensions]

        # last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for a, b, c, d in polygon_indices:
            dx = random.randint(-self.magnitude, self.magnitude)
            dy = random.randint(-self.magnitude, self.magnitude)

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,x2, y2,x3 + dx, y3 + dy,x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1, x2 + dx, y2 + dy,x3, y3,x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,x2, y2,x3, y3,x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,x2, y2,x3, y3,x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        def do(image):

            return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images


def GetRandomDistortImage(images):
    # images need use Image.open()
    width = np.random.randint(2, 4)
    height = np.random.randint(2, 4)
    mag = np.random.randint(2, 5)
    d = Distort(probability=1, grid_width=width, grid_height=height, magnitude=mag)
    tansformed_images = d.perform_operation(images)
    tansformed_images = [np.array(item) for item in tansformed_images]
    return tansformed_images


def random_dilute(image, sele_value=50, set_value=20, num_ratio=[0.1, 0.2]):
    index = np.where(image < (image.min() + sele_value))
    tag = []
    for i in range(len(index[0])):
        tag.append([index[0][i], index[1][i]])
    np.random.shuffle(tag)
    tag = np.array(tag)
    total_num = len(tag[:, 0])
    num = int(total_num * np.random.choice(num_ratio, 1)[0])
    tag1 = tag[:num, 0]
    tag2 = tag[:num, 1]
    index = (tag1, tag2)
    start = image.min() + sele_value + set_value
    if (start >= 230):
        start = start - 50
    ra_value = min(np.random.randint(min(225,start), 230), 230)
    image[index] = ra_value
    return image


def RandomAddLine(image):
    a = np.random.randint(0, 15)
    line_color = (a, a, a)
    thickness = np.random.randint(1, 3)
    if (np.random.randint(0, 10) > 7):
        tag1 = np.random.randint(1, 10)
        tag2 = np.random.randint(1, 5)
        image = cv2.line(image,
                         (tag1, image.shape[0] - tag2),
                         (image.shape[1] - tag1, image.shape[0] - tag2),
                         color=line_color,
                         thickness=thickness,
                         lineType=cv2.LINE_AA)
    if (np.random.randint(0, 10) > 7):
        tag1 = np.random.randint(1, 10)
        tag2 = np.random.randint(1, 5)
        image = cv2.line(image,
                         (tag1, tag2),
                         (image.shape[1] - tag1, tag2),
                         color=line_color,
                         thickness=thickness,
                         lineType=cv2.LINE_AA)
    if (np.random.randint(0, 10) > 7):
        tag1 = np.random.randint(1, 5)
        tag2 = np.random.randint(0, 4)
        image = cv2.line(image,
                         (tag1, tag2),
                         (tag1, image.shape[0] - tag2),
                         color=line_color,
                         thickness=thickness,
                         lineType=cv2.LINE_AA)
    if (np.random.randint(0, 10) > 7):
        tag1 = np.random.randint(1, 5)
        tag2 = np.random.randint(0, 4)
        image = cv2.line(image,
                         (image.shape[1] - tag1, tag2),
                         (image.shape[1] - tag1, image.shape[0] - tag2),
                         color=line_color,
                         thickness=thickness,
                         lineType=cv2.LINE_AA)
    return image


class DataAugmentatonMore():
    def __init__(self, image):
        self.image = image

    def motion_blur(self, degree=5, angle=180):
        # degree建议：2 - 5
        # angle建议：0 - 360
        # 都为整数
        image = np.array(self.image)
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)

        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred_image = np.array(blurred, dtype=np.uint8)
        return blurred_image

    def gaussian_blur(self, k_size=7, sigmaX=0, sigmaY=0):
        # k_size越大越模糊，且为奇数，建议[1，3，5，7，9]
        blurred_image = cv2.GaussianBlur(self.image, ksize=(k_size, k_size),
                                         sigmaX=sigmaX, sigmaY=sigmaY)
        return blurred_image

    def Contrast_and_Brightness(self, alpha, beta=0):
        # alpha:调节亮度，越小越暗，越大越亮，等于1为原始亮度
        # 建议使用0.6-1.3
        blank = np.zeros(self.image.shape, self.image.dtype)
        # dst = alpha * img + beta * blank
        brighted_image = cv2.addWeighted(self.image, alpha, blank, 1 - alpha, beta)
        return brighted_image

    def Add_Padding(self, top, bottom, left, right, color):
        padded_image = cv2.copyMakeBorder(self.image, top, bottom,
                                          left, right, cv2.BORDER_CONSTANT, value=color)
        return padded_image

    def Add_gaussian_noise(self, mode='gaussian'):
        ##mode : 'gaussian' ,'salt' , 'pepper '
        noise_image = random_noise(self.image, mode=mode)
        return noise_image

    def Perspective(self, ratio=30, type='top'):
        h, w = self.image.shape[:2]
        pts1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
        if type == 'top':
            pts2 = np.float32([[ratio, ratio], [0, h], [w, h], [w - ratio, ratio]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(self.image, M, (w, h))
            dst = dst[ratio:, :]
        elif (type == 'left'):
            pts2 = np.float32([[ratio, ratio // 2], [ratio, h - ratio // 2], [w, h], [w, 0]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(self.image, M, (w, h))
            dst = dst[:, ratio:]
        elif (type == 'right'):
            pts2 = np.float32([[0, 0], [0, h], [w - ratio, h - ratio // 2], [w - ratio, ratio // 2]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(self.image, M, (w - ratio, h))
        else:
            pts2 = np.float32([[0, 0], [0 + ratio, h - ratio], [w - ratio, h - ratio], [w, 0]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(self.image, M, (w, h - ratio))
        return dst

    def resize_blur(self, ratio):
        image = cv2.resize(self.image, dsize=None, fy=ratio, fx=ratio)
        new_width = np.round((self.image.shape[0] / image.shape[0]) * image.shape[1]).astype(np.int)
        image = cv2.resize(image, dsize=(new_width, self.image.shape[0]))
        return image


def transform_img_shape(img, img_shape):
    img = np.array(img)
    H, W = img_shape
    h, w = img.shape[:2]
    new_w = int((float(H)/h) * w)
    if (new_w > W):
        img = cv2.resize(img, (W, H))
    else:
        img = cv2.resize(img, (new_w, H))
        img = Add_Padding(img, 0, 0, 0, W - new_w, color=(0, 0, 0))
    return img

def random_crop(image):
    h, w= image.shape[:2]
    top_min = 1
    top_max = h//5
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :]
    else:
        crop_img = crop_img[0:h - top_crop, :]
    return crop_img

def get_background_Amg(img,bg_img,img_shape=[32,200]):
    H,W = img_shape
    x = np.random.randint(0, img.shape[0] - H + 1)
    y = np.random.randint(0, img.shape[1] - W + 1)
    img_crop = bg_img[x:x + H, y:y + W]
    ratio = np.random.randint(1,4)/10.0+np.random.randint(0,10)/100.0
#     print(img.shape,img_crop.shape)
    img = np.array(Image.fromarray(img).convert('RGB'))
    dst = cv2.addWeighted(img, 1-ratio, img_crop, ratio, 2)
    dst = np.array(Image.fromarray(dst).convert('L'))
    return dst

def DataAugment(image,bg_img,img_shape):
    image = np.array(image)
    if (np.random.choice([True, False], 1)[0]):
        dataAu = DataAugmentatonMore(image)
        index = np.random.randint(0,11)
        if (index == 0):
            degree = np.random.randint(2, 6)
            angle = np.random.randint(0, 360)
            image = dataAu.motion_blur(degree, angle)
        elif (index == 1):
            id = np.random.randint(0, 4)
            k_size = [1, 3, 5, 7,9]
            image = dataAu.gaussian_blur(k_size[id])
        elif (index == 2):
            alpha = np.random.uniform(0.6, 1.3)
            image = dataAu.Contrast_and_Brightness(alpha)
        elif (index == 3):
            types = ['top', 'botttom']
            id = np.random.randint(0, 2)
            ratio = np.random.randint(0, image.shape[0]//3)
            image = dataAu.Perspective(ratio, types[id])
        elif (index == 4):
            ratio = np.random.uniform(0.35, 0.5)
            image = dataAu.resize_blur(ratio)
        elif(index==5):
            image = random_dilute(image)
        elif(index==6):
            image = Image.fromarray(image)
            image = GetRandomDistortImage([image])[0]
        elif(index==7):
            image = Image.fromarray(image).convert('RGB')
            image = np.array(image)
            image = cvtColor(image)
            image = Image.fromarray(image).convert('L')
            image = np.array(image)
        elif(index==8):
            image = RandomAddLine(image)
        elif(index==9):
            image = random_crop(image)
        elif(index==10):
            image = get_background_Amg(image,bg_img,img_shape)
        del dataAu,bg_img

    return image


def transform_label(label,char_type='En',t_type = 'lower'):
    if char_type == 'En':
        if t_type == 'lower':
            label = ''.join(re.findall('[0-9a-zA-Z]+', label)).lower()
        elif t_type == 'upper':
            label = ''.join(re.findall('[0-9a-zA-Z]+', label)).upper()
        else:
            label = ''.join(re.findall('[0-9a-zA-Z]+', label))
    elif(char_type=='Ch'):
        return label
    return label



