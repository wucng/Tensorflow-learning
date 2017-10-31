# -*- coding: utf-8 -*-

"""
mnist 卷积模型
改进： 舍弃了全连接层，只使用卷积池化
输入[N 28 28 1] --卷积池化--->[N 1 1 10] --reshape-->[N 10](这个就是最后的输出)
"""


from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_1(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def max_pool_2x2_1(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64
h_pool2 = tf.nn.dropout(h_pool2, keep_prob)

## conv3 layer ##
W_conv3 = weight_variable([5,5, 64, 64]) # patch 5x5, in size 64, out size 64
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d_1(h_pool2, W_conv3) + b_conv3) # output size 7x7x64
h_pool3 = max_pool_2x2_1(h_conv3)                          # output size 3x3x64
h_pool3 = tf.nn.dropout(h_pool3, keep_prob)

## conv4 layer ##
W_conv4 = weight_variable([5,5, 64, 10]) # patch 5x5, in size 64, out size 10
b_conv4 = bias_variable([10])
h_conv4 = tf.nn.relu(conv2d_1(h_pool3, W_conv4) + b_conv4) # output size 3x3x10
h_pool4= max_pool_2x2_1(h_conv4)                         # output size 1x1x10
# h_pool4 = tf.nn.dropout(h_pool4, keep_prob)

prediction = tf.nn.softmax(tf.reshape(h_pool4, [-1, 1*1*10])) # [N 10] 这就是全连接层的工作


"""  舍弃掉全连接层
## fc1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
"""

# the error between prediction and real data
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                               reduction_indices=[1]))       # loss

cross_entropy=tf.reduce_mean(tf.losses.softmax_cross_entropy(ys,logits=prediction))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord) # 启动线程

for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))

coord.request_stop() # 等待结束
coord.join(threads) # 关闭线程
sess.close() # 关闭会话
