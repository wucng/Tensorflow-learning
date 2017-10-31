#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""说明
改写成卷积模型
数据：mnist
模型建立 Model
数据的输入 Inputs
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import argparse
import sys

class Inputs(object):
    def __init__(self,file_path,batch_size,one_hot=True):
        self.file_path=file_path
        self.batch_size=batch_size
        self.mnist=input_data.read_data_sets(self.file_path, one_hot=one_hot)
    def inputs(self):
        batch_xs, batch_ys = self.mnist.train.next_batch(self.batch_size)
        return batch_xs, batch_ys
    def test_inputs(self):
        return self.mnist.test.images,self.mnist.test.labels

class Conv_model(object):
    def __init__(self,X,Y,weights,biases,learning_rate,keep):
        # super(Conv_model, self).__init__(X,Y,w,b,learning_rate)  # 返回父类的对象
        # 或者 model.Model.__init__(self,X,Y,w,b,learning_rate)
        self.X = X
        self.Y = Y
        self.weights=weights
        self.biases=biases
        self.learning_rate = learning_rate
        self.keep=keep

    def conv2d(self,x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)  # strides中间两个为1 表示x,y方向都不间隔取样
        return tf.nn.relu(x)

    def maxpool2d(self,x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')  # strides中间两个为2 表示x,y方向都间隔1个取样

    def inference(self,name='conv',activation='softmax'): # 重写inference函数
        with tf.name_scope(name):
            conv1 = self.conv2d(self.X, self.weights['wc1'], self.biases['bc1'])
            conv1 = self.maxpool2d(conv1, k=2)
            conv1 = tf.nn.lrn(conv1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)
            conv1 = tf.nn.dropout(conv1, self.keep)

            conv2=self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
            conv2 = self.maxpool2d(conv2, k=2)
            conv2 = tf.nn.lrn(conv2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)
            conv2 = tf.nn.dropout(conv2, self.keep)

            fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1, self.keep)

            y = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])

        if activation=='softmax':
            y=tf.nn.softmax(y)
        return y

    def loss(self,pred_value,MSE_error=False,one_hot=True):
        if MSE_error:return tf.reduce_mean(tf.reduce_sum(
            tf.square(pred_value-self.Y),reduction_indices=[1]))
        else:
            if one_hot:
                return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.Y, logits=pred_value))
            else:
                return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(self.Y, tf.int32), logits=pred_value))

    def evaluate(self,pred_value,one_hot=True):
        if one_hot:
            correct_prediction = tf.equal(tf.argmax(pred_value, 1), tf.argmax(self.Y, 1))
            # correct_prediction = tf.nn.in_top_k(pred_value, Y, 1)
        else:
            correct_prediction = tf.equal(tf.argmax(pred_value, 1), tf.cast(self.Y, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self,cross_entropy):
        global_step = tf.Variable(0, trainable=False)
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy, global_step=global_step)


IMAGE_PIXELS = 28  # 图像大小 mnist 28x28x1  (后续参考自己图像大小进行修改)
channels = 1
num_class = 10

def train():
    # Store layers weight & bias
    weights = {
    # 5x5 conv, 3 input, 32 outputs 彩色图像3个输入(3个频道)，灰度图像1个输入
    'wc1': tf.get_variable('wc1', [3, 3, channels, 32], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),  # 5X5的卷积模板

    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.get_variable('wc2', [3, 3, 32, 64], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer, regularizer=tf.nn.l2_loss),

    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([(IMAGE_PIXELS // 4) * (IMAGE_PIXELS // 4) * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_class]))
    }

    biases = {
    'bc1': tf.get_variable('bc1',[32],dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
      'bc2': tf.get_variable('bc2',[64],dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer,regularizer=tf.nn.l2_loss),
      'bd1': tf.Variable(tf.random_normal([1024])),
      'out': tf.Variable(tf.random_normal([num_class]))
    }


    # Input layer
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 28*28*1],'x')
        y_ = tf.placeholder(tf.float32, [None,10],'y_')
    keep=tf.placeholder(tf.float32)
    with tf.name_scope('input_reshape'):
         image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])

    input_model=Inputs(FLAGS.data_dir,FLAGS.batch_size,one_hot=FLAGS.one_hot)
    model=Conv_model(image_shaped_input,y_,weights,biases,FLAGS.learning_rate,keep)
    y=model.inference(activation=None)
    cross_entropy=model.loss(y,MSE_error=False,one_hot=FLAGS.one_hot)
    train_op=model.train(cross_entropy)
    accuracy=model.evaluate(y,one_hot=FLAGS.one_hot)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)
        for step in range(FLAGS.num_steps):
            batch_xs, batch_ys = input_model.inputs()
            train_op.run({x: batch_xs, y_: batch_ys,keep:0.8})

            if step % FLAGS.disp_step == 0:
                acc=accuracy.eval({x: batch_xs, y_: batch_ys,keep:1.})
                print("step", step, 'acc', acc,
                      'loss', cross_entropy.eval({x: batch_xs, y_: batch_ys,keep:1.}))
        # test acc
        test_x, test_y = input_model.test_inputs()
        acc = accuracy.eval({x: test_x, y_: test_y,keep:1.})
        print('test acc', acc)

def main(_):
    # if tf.gfile.Exists(FLAGS.log_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.log_dir)
    # if not tf.gfile.Exists(FLAGS.log_dir):
    #     tf.gfile.MakeDirs(FLAGS.log_dir)
    train()

if __name__=="__main__":
    # 设置必要参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=1000,
                        help = 'Number of steps to run trainer.')
    parser.add_argument('--disp_step', type=int, default=100,
                        help='Number of steps to display.')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of mini training samples.')
    parser.add_argument('--one_hot', type=bool, default=True,
                        help='One-Hot Encoding.')
    parser.add_argument('--data_dir', type=str, default='./MNIST_data/',
            help = 'Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='./log_dir',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
