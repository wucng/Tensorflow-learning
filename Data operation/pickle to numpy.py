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
pickle to numpy
使用的数据：https://download.pytorch.org/tutorial/hymenoptera_data.zip
"""


def read_and_decode(filename):
    with gzip.open(filename, 'rb') as pkl_file:  # 打开文件
        data = pickle.load(pkl_file)  # 加载数据

    return data

if __name__=="__main__":
    data=read_and_decode('_test.pkl')
    print(data.shape)
