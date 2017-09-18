#!/usr/bin/python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
http://blog.csdn.net/wc781708249/article/details/77995292
liner regression
"""

# 数据
train_x=np.random.random([100,1]).astype(np.float32)
train_y=3.*train_x+5.
train_y=train_y+np.random.random([100,1])

x=tf.placeholder(tf.float32,[100,1],'x')
y_=tf.placeholder(tf.float32,[100,1],'y')

with tf.variable_scope('wb'):
    w=tf.get_variable('w',(1,1),dtype=tf.float32,initializer=tf.random_uniform_initializer)
    b=tf.Variable(0.0,dtype=tf.float32)


y=tf.add(tf.matmul(x,w),b)

# loss function
loss=tf.reduce_mean(tf.reduce_sum(tf.square(y-y_)))

train_op=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

sess=tf.InteractiveSession(graph=tf.get_default_graph())

tf.global_variables_initializer().run()

for step in range(1000):
    sess.run(train_op,feed_dict={x:train_x,y_:train_y})
    if step%100==0:
        print('w',w.eval(),'b',b.eval())
a=w.eval()
b=b.eval()
plt.figure()
plt.scatter(train_x,train_y,s=30,c='red',marker='o',alpha=0.5,label='C1')

plt.plot(train_x,train_x*a+b)
plt.show()

sess.close()
