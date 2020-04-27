#-*-coding:utf-8-*-


import sys,os,cv2,time
import math
import  numpy as np
import glob

import random

import matplotlib.pyplot as plt

import image_augmentation as ia


mouth_width = 64
mouth_height = 64
ear_width = 64
ear_height = 64


def sample_resize(w,h):
    path = "/home/zxkj/zxkjCode/iau_dms_python/data/mouth/neg"
    files =  glob.glob(path+"/*.png")
    for file in files:
        image = cv2.imread(file)
        if image.shape[0] == h and image.shape[1] == w:
            pass
        elif float(image.shape[0])/image.shape[1] == float(h)/w:
            resized_image = cv2.resize(image, (w, h), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(file, resized_image)
        else:
            if float(image.shape[1])/image.shape[0]>float(w)/h:
                t = h*image.shape[1]//image.shape[0]
                resized_image = cv2.resize(image, (t, h), interpolation = cv2.INTER_CUBIC)
                canvas = resized_image[0:h,(t-w)//2:w+(t-w)//2]
                cv2.imwrite(file, canvas)
            else:
                t = w*image.shape[0]//image.shape[1]
                resized_image = cv2.resize(image, (w, t), interpolation = cv2.INTER_CUBIC)
                canvas = resized_image[(t-h)//2:(t-h)//2+h,0:w]
                cv2.imwrite(file, canvas)




def random_flip(image):
    t = random.randint(0,60)
    if t//30 == 1:
        return image
    elif t//10 == 0:
        return cv2.flip(image,0)
    elif t//10 ==1:
        return cv2.flip(image,1)
    else:
        return cv2.flip(image,-1)


def random_gray(image):
    t = random.randint(0,20)
    if t //10 ==0:
        return image
    else:
        img_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)
        return image


def random_augmentation(image):
    if random.random() <0.0:
        return image

    # 按照比例随机对图像进行镜像
    img_varied = random_flip(image)

    # 随机亮度
    if random.random() < 0.5:
        t = random.randint(-100,100)
        blank = np.zeros(image.shape,image.dtype)
        img_varied = cv2.addWeighted(img_varied,1,blank,1,t)

    # 随机对比度
    if random.random() < 0.5:
        t = random.randint(-100,100)
        blank = np.zeros(image.shape,image.dtype)
        img_varied = cv2.addWeighted(img_varied,1+t/100.0,blank,1-t/100.0,0)

    # 按照比例随机对图像进行HSV扰动
    if random.random() < 1:
        img_varied = ia.random_hsv_transform(
            img_varied,
            10,
            0.1,
            0.1)

    # 按照比例随机对图像进行Gamma扰动
    if random.random() < 0.5:
        img_varied = ia.random_gamma_transform(
            img_varied,
            2.0)
    return img_varied



# files = glob.glob("./data/image/mouth/pos/*.png")
# random.shuffle(files)
# c_image = cv2.imread(files[0])
# cv2.imshow("o",c_image)
# imgae = random_augmentation(c_image)
# cv2.imshow("a",imgae)
# cv2.waitKey(0)
















# # 按照比例随机对图像进行裁剪
    # if random.random() < 1:
    #     img_varied = ia.random_crop(
    #         img_varied,
    #         0.1,
    #         0.8)

    # # 按照比例随机对图像进行旋转
    # if random.random() < 1.0:
    #     img_varied = ia.random_rotate(
    #         img_varied,
    #         1.0,
    #         10.0)