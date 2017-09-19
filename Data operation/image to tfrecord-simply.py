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
image to tfrecord  简化版
使用的数据：https://download.pytorch.org/tutorial/hymenoptera_data.zip
"""
def images_to_tfrecord(image_filenames,record_location,img_pixel_h=250,img_pixel_w=151,channels=3):
# record_location='./output/train'
    writer = None
    current_index = 0
    for i,image_filename in enumerate(image_filenames):

        if current_index % 10 == 0:
            if writer:
                writer.close()

            record_filename = "{record_location}-{current_index}.tfrecords".format(
                record_location=record_location,
                current_index=current_index)

            writer = tf.python_io.TFRecordWriter(record_filename)
        current_index += 1

        image_name=image_filename.split("/")[-2]  # 做标签  linux '/' windows '\\'
        # if image_name=="ants":label=0
        # else:label=1

        image_label = image_name.encode("utf-8")
        
        try:
            image = Image.open(image_filename)
            image = image.convert('L')  # 转成灰度图
            image = image.resize((img_pixel_h, img_pixel_w))
            # image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            image_bytes=np.array(image,np.float32).tobytes()
            # image_bytes = image.tobytes()  # 将图片转化为原生bytes
        except:
            print(image_filename)
            continue
        # image=np.array(image).flatten()

        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
            # 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
        }))
        writer.write(example.SerializeToString())
    writer.close()

if __name__=="__main__":
    # 将满足目录的所有.jpg文件的路径放置在image_filenames列表中
    # image_filenames存放所有满足条件的jpg的路径
    image_filenames = glob.glob("./hymenoptera_data/*/*/*.jpg")  # ==> <class 'list'>
    np.random.shuffle(image_filenames)  # 先随机打乱

    train_img_filenames=image_filenames[:int(len(image_filenames)*0.8)]
    test_img_filenames = image_filenames[int(len(image_filenames) * 0.8):]
    images_to_tfrecord(train_img_filenames,'./output/training-images/training-images')
    images_to_tfrecord(test_img_filenames, './output/testing-images/testing-images')
