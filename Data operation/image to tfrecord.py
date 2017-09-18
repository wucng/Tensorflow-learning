#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import glob
from itertools import groupby
from collections import defaultdict
from PIL import Image
import numpy as np

"""
image to tfrecord
使用的数据：https://download.pytorch.org/tutorial/hymenoptera_data.zip
http://blog.csdn.net/wc781708249/article/details/78013275
"""


# 将满足目录的所有.jpg文件的路径放置在image_filenames列表中
# image_filenames存放所有满足条件的jpg的路径
image_filenames = glob.glob("./hymenoptera_data/*/*/*.jpg")  # ==> <class 'list'>

sess = tf.InteractiveSession()

training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

# Split up the filename into its breed and corresponding filename. The breed is found by taking the directory name
# 将文件名分解为品种和相应的文件名（文件对应的路径），品种对应文件夹名称（作为标签）
image_filename_with_breed = map(lambda filename: (filename.split("/")[-2], filename), image_filenames)  # Linux "/" , windows "\\"

# Group each image by the breed which is the 0th element in the tuple returned above
for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
    # Enumerate each breed's image and send ~20% of the images to a testing set
    for i, breed_image in enumerate(breed_images):
        if i % 5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])  # dog_breed对应文件名，即标签，breed_image[1]对应jpg的路径
        else:
            training_dataset[dog_breed].append(breed_image[1])

    # Check that each breed includes at least 18% of the images for testing
    breed_training_count = len(training_dataset[dog_breed])
    breed_testing_count = len(testing_dataset[dog_breed])

    assert round(breed_testing_count / (breed_training_count + breed_testing_count),
                 2) > 0.18, "Not enough testing images."


# 图像--->tfrecode
def write_records_file(dataset, record_location):
    """
    Fill a TFRecords file with the images found in `dataset` and include their category.

    Parameters
    ----------
    dataset : dict(list)
      Dictionary with each key being a label for the list of image filenames of its value.
    record_location : str
      Location to store the TFRecord output.
    """
    writer = None

    # Enumerating the dataset because the current index is used to breakup the files if they get over 100
    # images to avoid a slowdown in writing.
    # 枚举dataset，因为当前索引用于对文件进行划分，每隔100幅图像，训练样本的信息就被写入到一个新的Tfrecode文件中，以加快操作的进程
    current_index = 0
    for breed, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if current_index % 10 == 0:
                if writer:
                    writer.close()

                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=record_location,
                    current_index=current_index)

                writer = tf.python_io.TFRecordWriter(record_filename)
            current_index += 1

            '''
            # 方法一，使用PIL
            try:
                image=Image.open(image_filename)
                image=image.convert('L') #转成灰度图
                image=image.resize((250,151))
            except:
                print(image_filename)
                continue

            image_bytes = sess.run(tf.cast(np.array(image), tf.uint8)).tobytes()
            '''

            # 方法二、使用tf.image.decode_jpeg
            # 在ImageNet的狗的图像中，有少量无法被Tensorflow识别的JPEG的图像，利用try/catch可以将这些图像忽略
            try:
                image_file = tf.read_file(image_filename)
                image = tf.image.decode_jpeg(image_file)

                # 转换成灰度图可以减少处理的计算量和内存占用，但这不是必须的
                grayscale_image = tf.image.rgb_to_grayscale(image)  # 转成灰度
                resized_image = tf.image.resize_images(grayscale_image, (250, 151))  # 图像大小固定为 250x151
# resized_image=tf.image.resize_image_with_crop_or_pad(grayscale_image,250,151)

                # 这里之所以使用tf.cast,是因为虽然尺寸更改后的图像数据类型是浮点型，但RGB尚未转换到[0,1)区间内
                image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            except:
                print(image_filename)
                continue

            # https://en.wikipedia.org/wiki/One-hot
            # 将标签按字符串存储较高效，推荐的做法是将其转换为整数索引或独热编码的秩1张量

            #
            '''
            image_label = tf.case({tf.equal(breed, tf.constant('n02085620-Chihuahua')): lambda: tf.constant(0),
                              tf.equal(breed, tf.constant('n02096051-Airedale')): lambda: tf.constant(1),
                              }, lambda: tf.constant(-1), exclusive=True)

            image_label = sess.run(image_label)
            '''
            image_label = breed.encode("utf-8")


            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                # 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_label])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            }))

            writer.write(example.SerializeToString())
    # if writer:
    writer.close()


if __name__ == "__main__":
    write_records_file(training_dataset, "./output/training-images/training-images")
    write_records_file(testing_dataset, "./output/testing-images/testing-images")
