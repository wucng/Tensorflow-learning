#!/usr/bin/python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
DNN
http://blog.csdn.net/wc781708249/article/details/78003037
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# mnist图像大小是28x28 分成0~9 共10类
x=tf.placeholder(tf.float32,[None,28*28*1])
y_=tf.placeholder(tf.float32,[None,10])

with tf.variable_scope('wb'):
    w=tf.get_variable('w',[28*28,128],initializer=tf.random_uniform_initializer)*0.001
    b=tf.Variable(tf.zeros([128])+0.1,dtype=tf.float32)
with tf.variable_scope('wb2'):
    w2 = tf.get_variable('w2', [128, 10], initializer=tf.random_uniform_initializer) * 0.001
    b2=tf.Variable(tf.zeros([10])+0.1,dtype=tf.float32)

y=tf.nn.relu(tf.add(tf.matmul(x,w),b))
y=tf.nn.softmax(tf.add(tf.matmul(y,w2),b2))

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

train_op=tf.train.AdamOptimizer(0.1).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess=tf.InteractiveSession(graph=tf.get_default_graph())

tf.global_variables_initializer().run()

for step in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(128)
    train_op.run({x:batch_xs,y_:batch_ys})
    if step % 1000==0:
        print("step",step,'acc',accuracy.eval({x:batch_xs,y_:batch_ys}),'loss',loss.eval({x:batch_xs,y_:batch_ys}))

# test acc
print('test acc',accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))

sess.close()
