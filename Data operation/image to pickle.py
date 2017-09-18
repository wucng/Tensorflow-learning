#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import glob
from PIL import Image
import numpy as np
import pickle
import gzip

"""
image to pickle
使用的数据：https://download.pytorch.org/tutorial/hymenoptera_data.zip
"""
img_pixel_h=60
img_pixel_w=60
channels=3
pkl_path='./'
# 将满足目录的所有.jpg文件的路径放置在image_filenames列表中
# image_filenames存放所有满足条件的jpg的路径
image_filenames = glob.glob("./hymenoptera_data/*/*/*.jpg")  # ==> <class 'list'>
testing_dataset=[]
training_dataset=[]

for i,image_filename in enumerate(image_filenames):
    image_name=image_filename.split("/")[-2]  # 做标签  linux '/' windows '\\'
    if image_name=="ants":label=0
    else:label=1
    try:
        image = Image.open(image_filename)
        image = image.convert('L')  # 转成灰度图
        image = image.resize((img_pixel_h, img_pixel_w))
    except:
        print(image_filename)
        continue
    image=np.array(image).flatten()
    data = np.append(image, label)[np.newaxis, :]
    if i%5==0:
        testing_dataset.append(data)
    else:
        training_dataset.append(data)

data_matrix = np.array(testing_dataset, dtype=np.float32)
data_matrix = data_matrix.reshape((-1, img_pixel_h * img_pixel_w * channels//3 + 1))
with gzip.open(pkl_path + '_' + str('test') + '.pkl', 'wb') as writer:  # 以压缩包方式创建文件，进一步压缩文件
    pickle.dump(data_matrix, writer)  # 数据存储成pickle文件
testing_dataset = []
data_matrix=[]

data_matrix = np.array(training_dataset, dtype=np.float32)
data_matrix = data_matrix.reshape((-1, img_pixel_h * img_pixel_w * channels//3 + 1))
with gzip.open(pkl_path + '_' + str('train') + '.pkl', 'wb') as writer:  # 以压缩包方式创建文件，进一步压缩文件
    pickle.dump(data_matrix, writer)  # 数据存储成pickle文件
training_dataset = []
data_matrix = []
