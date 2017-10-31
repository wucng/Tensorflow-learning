#!/usr/bin/python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.layers.python.layers import fully_connected,convolution2d
from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.python.framework import ops

# ops.reset_default_graph()
"""
cnn
参考：http://blog.csdn.net/wc781708249/article/details/78007593
"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义一些参数
batch_size = 128
droup_out = 0.7
learn_rate = 0.001
num_steps = 100000
disp_step = 2000

with tf.Graph().as_default() as graph:
    # mnist图像大小是28x28 分成0~9 共10类
    x=tf.placeholder(tf.float32,[None,28*28*1])
    y_=tf.placeholder(tf.float32,[None,10])
    keep=tf.placeholder(tf.float32)

    x_img=tf.reshape(x,[-1,28,28,1])

    # convolution1
    """
    conv1=tf.layers.conv2d(
        tf.image.convert_image_dtype(x_img,dtype=tf.float32),
        filters=32, # 输出通道由1->32
        kernel_size=(3,3), # 3x3卷积核
        activation=tf.nn.relu,
        padding='SAME',
        kernel_initializer=tf.random_uniform_initializer,
        bias_initializer=tf.random_normal_initializer
    )
    """
    conv1=convolution2d(
        tf.image.convert_image_dtype(x_img,dtype=tf.float32),
        num_outputs=32,
        kernel_size=(3,3),
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.layers.batch_normalization,
        weights_initializer=tf.random_uniform_initializer,
        biases_initializer=tf.random_normal_initializer,
        trainable=True
    )

    conv1=tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding="SAME") # [n,14,14,32]

    # convolution2
    """
    conv2=tf.layers.conv2d(
        conv1,
        filters=64, # 输出通道由1->32
        kernel_size=(3,3), # 3x3卷积核
        activation=tf.nn.relu,
        padding='SAME',
        kernel_initializer=tf.random_uniform_initializer,
        bias_initializer=tf.random_normal_initializer
    )
    """
    conv2=convolution2d(
        conv1,
        num_outputs=64,
        kernel_size=(3,3),
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.layers.batch_normalization,
        weights_initializer=tf.random_uniform_initializer,
        biases_initializer=tf.random_normal_initializer,
        trainable=True
    )

    conv2=tf.nn.max_pool(conv2,[1,2,2,1],[1,2,2,1],padding="SAME") # [n,7,7,64]

    # full connect
    fc1=tf.reshape(conv2,[-1,7*7*64])
    fc1=fully_connected(
        fc1,
        num_outputs=512,
        activation_fn=tf.nn.relu,
        normalizer_fn=tf.layers.batch_normalization,
        weights_initializer=tf.random_uniform_initializer,
        biases_initializer=tf.random_normal_initializer,
        weights_regularizer=tf.nn.l2_loss,
        biases_regularizer=tf.nn.l2_loss,
    ) # [N,512]
    fc1=tf.nn.dropout(fc1,keep)

    y=fully_connected(
        fc1,
        num_outputs=10,
        activation_fn=tf.nn.softmax,
        normalizer_fn=tf.layers.batch_normalization,
        weights_initializer=tf.random_uniform_initializer,
        biases_initializer=tf.random_normal_initializer,
        weights_regularizer=tf.nn.l2_loss,
        biases_regularizer=tf.nn.l2_loss,
    ) # [N,10]


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
        print("step",step,'acc',accuracy.eval({x:batch_xs,y_:batch_ys,keep:droup_out}),'loss',loss.eval({x:batch_xs,y_:batch_ys,keep:droup_out}))

# test acc
print('test acc',accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep:1.}))

sess.close()
