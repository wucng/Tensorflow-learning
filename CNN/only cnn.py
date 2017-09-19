#!/usr/bin/python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.python.framework import ops

# ops.reset_default_graph()
"""
only cnn ，not full connect
"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义一些参数
batch_size = 128
droup_out = 0.7
learn_rate = 0.001
num_steps = 100000
disp_step = 2000

img_size=28
n_classes = 10

with tf.Graph().as_default() as graph:
    # mnist图像大小是28x28 分成0~9 共10类
    x=tf.placeholder(tf.float32,[None,img_size*img_size])
    y_=tf.placeholder(tf.float32,[None,n_classes])
    keep=tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool, name='MODE')

    def batch_norm_layer(inputT, is_training=True, scope=None):
        # Note: is_training is tf.placeholder(tf.bool) type
        return tf.cond(is_training,
                       lambda: batch_norm(inputT, is_training=True,
                                          center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                       lambda: batch_norm(inputT, is_training=False,
                                          center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                                          scope=scope, reuse = True))

    def conv2d(input,kernel_size,input_size,output_size,is_training,name):
        with tf.name_scope(name) as scope:
            with tf.variable_scope(name):
                # scope.reuse_variables()
                w=tf.get_variable('w',[kernel_size,kernel_size,input_size,output_size],tf.float32,initializer=tf.random_uniform_initializer)*0.001
                b=tf.get_variable('b',[output_size],tf.float32,initializer=tf.random_normal_initializer)+0.1
                conv=tf.nn.conv2d(input,w,[1,1,1,1],padding="SAME")
                conv = tf.nn.bias_add(conv, b)
                conv=batch_norm_layer(conv,is_training,scope)
                conv=tf.nn.relu(conv)
        return conv


    x_img = tf.reshape(x, [-1, img_size, img_size, 1])

    # conv1
    conv1=conv2d(tf.image.convert_image_dtype(x_img,tf.float32),
                 3,1,16,is_training,'conv1')
    conv1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")  # [n,14,14,16]
    conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    conv1 = tf.nn.dropout(conv1, keep)

    # conv2
    conv2 = conv2d(conv1,
                   3, 16, 32, is_training,'conv2')
    conv2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")  # [n,7,7,32]
    conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    conv2 = tf.nn.dropout(conv2, keep)

    # conv3
    conv3 = conv2d(conv2,
                   3, 32, 64, is_training,'conv3')
    conv3 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")  # [n,3,3,64]
    conv3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    conv3 = tf.nn.dropout(conv3, keep)

    # conv4
    conv4 = conv2d(conv3,
                   3, 64, 10, is_training,'conv4')
    conv4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")  # [n,1,1,10]
    conv4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
    # conv4 = tf.nn.dropout(conv4, keep)

    # output
    y=tf.reshape(conv4,[-1,n_classes])

    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

    train_op=tf.train.AdamOptimizer(learn_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess=tf.InteractiveSession(graph=graph)

tf.global_variables_initializer().run()

for step in range(num_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    train_op.run({x:batch_xs,y_:batch_ys,keep:droup_out,is_training:True})
    if step % disp_step==0:
        print("step",step,'acc',accuracy.eval({x:batch_xs,y_:batch_ys,keep:droup_out,is_training:True}),
              'loss',loss.eval({x:batch_xs,y_:batch_ys,keep:droup_out,is_training:True}))

# test acc
# print('test acc',accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep:1.,is_training:False}))
print('test acc',accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep:1.,is_training:True}))
sess.close()
