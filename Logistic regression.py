#!/usr/bin/python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Logistic regression
http://blog.csdn.net/wc781708249/article/details/78002679
"""

# 数据
x1 = np.random.normal(-4, 2, 1000)[:,np.newaxis]  # 1000x1
x2 = np.random.normal(4,2 , 1000)[:,np.newaxis]
train_x = np.vstack((x1, x2)) # 2000x1
train_y = np.asarray([0.] * len(x1) + [1.] * len(x2))[:,np.newaxis] # 2000x1

plt.scatter(train_x, train_y)
# plt.show()
x=tf.placeholder(tf.float32,[None,1],'x')
y_=tf.placeholder(tf.float32,[None,1],'y_')

with tf.variable_scope('wb'):
    w=tf.get_variable('w',(1,1),dtype=tf.float32,initializer=tf.random_uniform_initializer)
    b= tf.Variable(tf.zeros([1, 1]) + 0.1)
with tf.variable_scope('wb2') as scope:
    # scope.reuse_variables()
    w2=tf.get_variable('w2',(10,1),dtype=tf.float32,initializer=tf.random_uniform_initializer)
    b2= tf.Variable(tf.zeros([1, 1]) + 0.1)


y=tf.nn.sigmoid(tf.add(tf.matmul(x,w),b))
# y=tf.nn.relu(tf.add(tf.matmul(x,w),b))
# y=tf.nn.sigmoid(tf.add(tf.matmul(y,w2),b2))


# loss function
# loss=tf.reduce_mean(tf.reduce_sum(tf.square(y-y_),reduction_indices=[1]))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

train_op=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess=tf.InteractiveSession(graph=tf.get_default_graph())

tf.global_variables_initializer().run()


for step in range(1000):
    sess.run(train_op,feed_dict={x:train_x,y_:train_y})

all_xs = np.linspace(-10, 10, 100)[:,np.newaxis]
prdiction_value = sess.run(y, feed_dict={x: all_xs})
lines = plt.plot(all_xs, prdiction_value, 'r-', lw=5)
plt.show()
sess.close()
