#!/usr/bin/python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

slim = tf.contrib.slim

""" transfer learning
导入pb文件 初始化输出层（最后一层）
参考：http://blog.csdn.net/wc781708249/article/details/78043099
"""

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
num_classes=10

def load_pd(pd_path):
    output_graph_def = tf.GraphDef()
    with open(pd_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

with tf.Graph().as_default():
    load_pd("mnist.pb")
    with tf.Session() as sess:
        # init = tf.global_variables_initializer()
        # sess.run(init)

        input_x = sess.graph.get_tensor_by_name("input_x:0")
        input_y = sess.graph.get_tensor_by_name("input_y:0")
        one_hot_labels = slim.one_hot_encoding(input_y, 10)
        net = sess.graph.get_tensor_by_name("InceptionResnetV2/fc:0")
        net = slim.dropout(net, 0.7, is_training=True,scope='Dropout')

        ## output layer ##
        logits = slim.fully_connected(net, num_classes, activation_fn=None,scope='Logits')
        prediction = tf.nn.softmax(logits, name='softmax')

        # the error between prediction and real data
        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=))  # loss
        cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(prediction, 1), input_y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(200):
            batch_xs, batch_ys = mnist.test.next_batch(200)
            batch_xs = np.reshape(batch_xs, [-1, 28, 28, 1])
            sess.run(train_step, feed_dict={input_x: batch_xs, input_y: batch_ys})
            if step % 20 == 0:
                print('step', step, 'acc',
                      accuracy.eval({input_x:batch_xs, input_y:batch_ys}))


        img_out_softmax = sess.run(prediction, feed_dict={input_x:np.reshape(mnist.train.images[:10],[-1,28,28,1])})
        prediction_labels = np.argmax(img_out_softmax, axis=1)
        print("label:", prediction_labels)
        # print('true label:', np.argmax(mnist.test.labels[:10], axis=1))
        print('true label:', mnist.train.labels[:10])