#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""调用slim 提供的网络来运行自己的数据
这里调用alexnet 网络
使用的数据集 mnist
参考：
1、https://github.com/tensorflow/models/blob/master/research/slim/nets
2、http://blog.csdn.net/wc781708249/article/details/78414930
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
# slim = tf.contrib.slim
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys
import alexnet  # 导入alexnet


class Conv_model(object):
    # def __init__(self, X, Y, weights, biases, learning_rate, keep):
    def __init__(self, Y, learning_rate):
        # super(Conv_model, self).__init__(X,Y,w,b,learning_rate)  # 返回父类的对象
        # 或者 model.Model.__init__(self,X,Y,w,b,learning_rate)
        # self.X = X
        self.Y = Y
        # self.weights = weights
        # self.biases = biases
        self.learning_rate = learning_rate
        # self.keep = keep

    '''
    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)  # strides中间两个为1 表示x,y方向都不间隔取样
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')  # strides中间两个为2 表示x,y方向都间隔1个取样

    def inference(self, name='conv', activation='softmax'):  # 重写inference函数
        with tf.name_scope(name):
            conv1 = self.conv2d(self.X, self.weights['wc1'], self.biases['bc1'])
            conv1 = self.maxpool2d(conv1, k=2)  # shape [N,1,1,32]
            conv1 = tf.nn.lrn(conv1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75)
            conv1 = tf.nn.dropout(conv1, self.keep)

            fc1 = tf.reshape(conv1, [-1, self.weights['wd1'].get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1, self.keep)

            y = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])

        if activation == 'softmax':
            y = tf.nn.softmax(y)
        return y
    '''

    def loss(self, pred_value, MSE_error=False, one_hot=True):
        if MSE_error:
            return tf.reduce_mean(tf.reduce_sum(
                tf.square(pred_value - self.Y), reduction_indices=[1]))
        else:
            if one_hot:
                return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.Y, logits=pred_value))
            else:
                return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(self.Y, tf.int32), logits=pred_value))

    def evaluate(self, pred_value, one_hot=True):
        if one_hot:
            correct_prediction = tf.equal(tf.argmax(pred_value, 1), tf.argmax(self.Y, 1))
            # correct_prediction = tf.nn.in_top_k(pred_value, Y, 1)
        else:
            correct_prediction = tf.equal(tf.argmax(pred_value, 1), tf.cast(self.Y, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, cross_entropy):
        global_step = tf.Variable(0, trainable=False)
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy,
                                                                              global_step=global_step)

class Inputs(object):
    def __init__(self,file_path,batch_size,one_hot=True):
        self.file_path=file_path
        self.batch_size=batch_size
        self.mnist=input_data.read_data_sets(self.file_path, one_hot=one_hot)
    def inputs(self):
        batch_xs, batch_ys = self.mnist.train.next_batch(self.batch_size)
        return batch_xs, batch_ys
    def test_inputs(self):
        return self.mnist.test.images[:200],self.mnist.test.labels[:200]

FLAGS=None

def train():

    input_model = Inputs(FLAGS.data_dir, FLAGS.batch_size, one_hot=FLAGS.one_hot)

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 28*28*1],'x')
        y_ = tf.placeholder(tf.float32, [None,10],'y_')
        keep=tf.placeholder(tf.float32)
        is_training= tf.placeholder(tf.bool, name='MODE')

    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1]) # shape [n,28,28,1]
    # alexnet要求的数据shape是 224x224
    image_shaped_input=tf.image.resize_image_with_crop_or_pad(image_shaped_input,224,224) # shape[n,224,224,1]

    # with slim.arg_scope(cifarnet_arg_scope()):
    # with slim.arg_scope(inception_resnet_v2_arg_scope()):
    #     y, _ = cifarnet(images=image_shaped_input,num_classes=10,is_training=is_training,dropout_keep_prob=keep)

    # 上面的修改成
    with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
        y, _ = alexnet.alexnet_v2(inputs=image_shaped_input,num_classes=10,is_training=is_training,dropout_keep_prob=keep)

    model=Conv_model(y_,FLAGS.learning_rate)
    cross_entropy = model.loss(y, MSE_error=False, one_hot=FLAGS.one_hot)
    train_op = model.train(cross_entropy)
    accuracy = model.evaluate(y, one_hot=FLAGS.one_hot)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init)
        for step in range(FLAGS.num_steps):
            batch_xs, batch_ys = input_model.inputs()
            train_op.run({x: batch_xs, y_: batch_ys,keep:0.7,is_training:True})

            if step % FLAGS.disp_step == 0:
                acc=accuracy.eval({x: batch_xs, y_: batch_ys,keep:1.,is_training:False})
                print("step", step, 'acc', acc,
                      'loss', cross_entropy.eval({x: batch_xs, y_: batch_ys,keep:1.,is_training:False}))
        # test acc
        test_x, test_y = input_model.test_inputs()
        acc = accuracy.eval({x: test_x, y_: test_y,keep:1.,is_training:False})
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
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of mini training samples.')
    parser.add_argument('--one_hot', type=bool, default=True,
                        help='One-Hot Encoding.')
    parser.add_argument('--data_dir', type=str, default='./MNIST_data',
            help = 'Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='./log_dir',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)