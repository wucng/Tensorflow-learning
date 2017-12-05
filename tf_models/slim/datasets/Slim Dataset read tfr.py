#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

'''
https://github.com/tensorflow/models/tree/master/research/slim

您可以轻松地定义一个Slim Dataset，它存储指向数据文件的指针，
以及各种其他元数据，如类标签， train/test拆分以及如何解析TFExample protos。
'''

import tensorflow as tf
from datasets import flowers

slim = tf.contrib.slim

# Selects the 'validation' dataset.
DATA_DIR='./'
dataset = flowers.get_split('validation', DATA_DIR)

# Creates a TF-Slim DataProvider which reads the dataset in the background
# during both training and testing.
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])

print(type(image))
print(image.get_shape)
