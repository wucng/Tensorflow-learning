#!/usr/bin/python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
http://blog.csdn.net/wc781708249/article/details/77995831
nonlinear regression
"""

# 数据
train_x=np.linspace(-1,1,300).astype(np.float32)
train_x=np.reshape(train_x,[300,1])
train_y=3.*np.square(train_x)+5.
train_y=train_y+np.random.random([300,1])

x=tf.placeholder(tf.float32,[None,1],'x')
y_=tf.placeholder(tf.float32,[None,1],'y_')

with tf.variable_scope('wb'):
    w=tf.get_variable('w',(1,10),dtype=tf.float32,initializer=tf.random_uniform_initializer)
    b= tf.Variable(tf.zeros([1, 10]) + 0.1)
with tf.variable_scope('wb2') as scope:
    # scope.reuse_variables()
    w2=tf.get_variable('w2',(10,1),dtype=tf.float32,initializer=tf.random_uniform_initializer)
    b2= tf.Variable(tf.zeros([1, 1]) + 0.1)


y=tf.nn.tanh(tf.add(tf.matmul(x,w),b))
# y=tf.nn.relu(tf.add(tf.matmul(x,w),b))
y=tf.add(tf.matmul(y,w2),b2)

# loss function
loss=tf.reduce_mean(tf.reduce_sum(tf.square(y-y_),reduction_indices=[1]))

train_op=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess=tf.InteractiveSession(graph=tf.get_default_graph())

tf.global_variables_initializer().run()


fig = plt.figure()  # 设置图片框
ax = fig.add_subplot(1, 1, 1)  # 设置空白图片
ax.scatter(train_x, train_y)  # scatter以点的形式显示

for step in range(1000):
    sess.run(train_op,feed_dict={x:train_x,y_:train_y})

prdiction_value = sess.run(y, feed_dict={x: train_x})
lines = ax.plot(train_x, prdiction_value, 'r-', lw=5)
plt.show()
sess.close()
