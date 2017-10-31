#!/usr/bin/python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
# from tensorflow.contrib.layers.python.layers import batch_norm
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.python.framework import ops

# ops.reset_default_graph()
"""
nn+rnn mnist分类
http://blog.csdn.net/wc781708249/article/details/78009470
"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义一些参数
batch_size = 128
droup_out = 0.7
learn_rate = 0.001
num_steps = 100000
disp_step = 2000

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

with tf.Graph().as_default() as graph:
    # mnist图像大小是28x28 分成0~9 共10类
    x=tf.placeholder(tf.float32,[None,n_steps*n_input])
    y_=tf.placeholder(tf.float32,[None,n_classes])
    keep=tf.placeholder(tf.float32)

    # x_img=tf.reshape(x,[-1,n_steps,n_input,1])

    w1=tf.Variable(tf.random_normal([n_steps*n_input,n_steps*14]))
    b1=tf.Variable(tf.random_normal([n_steps*14]))

    x_img=tf.nn.relu(tf.nn.bias_add(tf.matmul(x,w1),b1))

    x_img=tf.reshape(x_img,[-1,n_steps,14])

    x_img=tf.unstack(x_img,n_steps,1) # 按时间序列，即第二维将[N,n_steps, n_input] 拆分成 n_steps个[N,14]序列 ，数据类型 list

    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # 加入多层rnn核
    lstm_cell = rnn.MultiRNNCell([lstm_cell] * 1, state_is_tuple=True)
    lstm_cell = rnn.MultiRNNCell([lstm_cell] * 1, state_is_tuple=True)
    lstm_cell = rnn.MultiRNNCell([lstm_cell] * 1, state_is_tuple=True)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x_img, dtype=tf.float32)

    with tf.variable_scope('output') as scope:
        w=tf.get_variable('w',[n_hidden,n_classes],tf.float32,initializer=tf.random_uniform_initializer)*0.001
        b=tf.Variable(tf.random_normal([n_classes])+0.001)
    y=tf.nn.softmax(tf.matmul(outputs[-1], w) + b)


    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

    train_op=tf.train.AdamOptimizer(learn_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess=tf.InteractiveSession(graph=graph)

tf.global_variables_initializer().run()

for step in range(num_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    train_op.run({x:batch_xs,y_:batch_ys,keep:droup_out})
    if step % disp_step==0:
        print("step",step,'acc',accuracy.eval({x:batch_xs,y_:batch_ys,keep:droup_out}),
              'loss',loss.eval({x:batch_xs,y_:batch_ys,keep:droup_out}))

# test acc
print('test acc',accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep:1.}))

sess.close()
