#!/usr/bin/python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.python.framework import ops

# ops.reset_default_graph()
"""
General cnn 一般cnn搭建方法

一般 image[h,w,c]-->label [1,1]

这里使用 image[h w c]-->label [h,w]

http://blog.csdn.net/wc781708249/article/details/78007593
http://blog.csdn.net/wc781708249/article/details/78013822
"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义一些参数
IMAGE_PIXELS=28
batch_size = 128
droup_out = 0.7
learn_rate = 0.1
num_steps = 1000
disp_step = 200

with tf.Graph().as_default() as graph:
    # mnist图像大小是28x28 对应的标签 28x28
    x=tf.placeholder(tf.float32,[None,IMAGE_PIXELS*IMAGE_PIXELS*1])
    y_=tf.placeholder(tf.float32,[None,IMAGE_PIXELS,IMAGE_PIXELS])
    keep=tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool, name='MODE')

    x_img=tf.reshape(x,[-1,IMAGE_PIXELS,IMAGE_PIXELS,1])

    def batch_norm_layer(inputT, is_training=True, scope=None):
        # Note: is_training is tf.placeholder(tf.bool) type
        return tf.cond(is_training,
                       lambda: batch_norm(inputT, is_training=True,
                                          center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                       lambda: batch_norm(inputT, is_training=False,
                                          center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                                          scope=scope, reuse = True))


    def conv2d(input, kernel_size, input_size, output_size, is_training, name):
        with tf.name_scope(name) as scope:
            with tf.variable_scope(name):
                # scope.reuse_variables()
                w = tf.get_variable('w', [kernel_size, kernel_size, input_size, output_size], tf.float32,
                                    initializer=tf.random_uniform_initializer) * 0.001
                b = tf.get_variable('b', [output_size], tf.float32, initializer=tf.random_normal_initializer) + 0.1
                conv = tf.nn.conv2d(input, w, [1, 1, 1, 1], padding="SAME")
                conv = tf.nn.bias_add(conv, b)
                conv = batch_norm_layer(conv, is_training, scope)
                conv = tf.nn.relu(conv)
        return conv

    def fc_layer(input,input_size,output_size,is_training,name):
        with tf.name_scope(name) as scope:
            with tf.variable_scope(name):
                w = tf.get_variable('w', [input_size, output_size], tf.float32,
                                    initializer=tf.random_uniform_initializer) * 0.001
                b = tf.get_variable('b', [output_size], tf.float32, initializer=tf.random_normal_initializer) + 0.1
                fc=tf.nn.bias_add(tf.matmul(input,w),b)
                fc=batch_norm_layer(fc,is_training,scope)
                # fc = tf.nn.relu(fc)
                return fc

    # convolution1
    conv1 = conv2d(tf.image.convert_image_dtype(x_img, tf.float32),
                   3, 1, 32, is_training, 'conv1')
    conv1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")  # [n,14,14,32]
    conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    conv1 = tf.nn.dropout(conv1, keep)

    # convolution2
    conv2 = conv2d(conv1,
                   3, 32, 64, is_training, 'conv2')
    conv2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")  # [n,7,7,64]
    conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    conv2 = tf.nn.dropout(conv2, keep)

    # full connect
    fc1=tf.reshape(conv2,[-1,7*7*64])

    fc1=fc_layer(fc1,7*7*64,512,is_training,'fc1')
    fc1=tf.nn.relu(fc1)
    fc1=tf.nn.dropout(fc1,keep)

    fc2=fc_layer(fc1,512,IMAGE_PIXELS*IMAGE_PIXELS,is_training,'output')
    # y=tf.nn.softmax(fc2)
    y=tf.reshape(fc2,[-1,IMAGE_PIXELS,IMAGE_PIXELS])

    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
    # loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_)))

    train_op=tf.train.AdamOptimizer(learn_rate).minimize(loss)

    # Calculate accuracy
    def compute_acc(xs, ys, IMAGE_PIXELS,flag=True):
        global y
        if flag:
            y1 = sess.run(y, {x: xs, y_: ys, keep: 1., is_training: True})
        else:
            y1 = sess.run(y, {x: xs, y_: ys, keep: 1., is_training: False})
        prediction = [1. if abs(x3 - 1) < abs(x3 - 0) else 0. for x1 in y1 for x2 in x1 for x3 in x2]
        prediction = np.reshape(prediction, [-1, IMAGE_PIXELS, IMAGE_PIXELS]).astype(np.uint8)
        accuracy = np.mean(np.equal(prediction, ys).astype(np.float32))
        return accuracy
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess=tf.InteractiveSession(graph=graph)

tf.global_variables_initializer().run()

for step in range(num_steps):
    # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs, _ = mnist.train.next_batch(batch_size)
    batch_ys=np.reshape(batch_xs,[-1,IMAGE_PIXELS,IMAGE_PIXELS])/255. # 将其像素值转成0/1 做标签
    train_op.run({x:batch_xs,y_:batch_ys,keep:droup_out,is_training:True})
    if step % disp_step==0:
        acc = compute_acc(batch_xs, batch_ys, IMAGE_PIXELS,True)
        print("step",step,'acc',acc,
              'loss',loss.eval({x:batch_xs,y_:batch_ys,keep:droup_out,is_training:True}))

# test acc
acc = compute_acc(mnist.test.images, np.reshape(mnist.test.images,[-1,IMAGE_PIXELS,IMAGE_PIXELS])/255., IMAGE_PIXELS,False)
print('test acc',acc)

sess.close()
