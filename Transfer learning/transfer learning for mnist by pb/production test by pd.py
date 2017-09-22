#!/usr/bin/python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

"""
导入pb文件 部署生产测试
参考：http://blog.csdn.net/wc781708249/article/details/78043099
"""

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)


def load_pd(pd_path):
    output_graph_def = tf.GraphDef()
    with open(pd_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

with tf.Graph().as_default():
    load_pd("mnist.pb")
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        input_x = sess.graph.get_tensor_by_name("input_x:0")

        out_softmax = sess.graph.get_tensor_by_name("InceptionResnetV2/Logits/softmax:0")

        img_out_softmax = sess.run(out_softmax, feed_dict={input_x:np.reshape(mnist.test.images[:10],[-1,28,28,1])})
        prediction_labels = np.argmax(img_out_softmax, axis=1)
        print("label:", prediction_labels)
        # print('true label:', np.argmax(mnist.test.labels[:10], axis=1))
        print('true label:', mnist.test.labels[:10])