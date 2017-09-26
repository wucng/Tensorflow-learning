#!/usr/bin/python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

# X=mnist.train.images
# total_samples = len(X)
image_dim = 784 # 28*28 pixels
z_dim = 14*14 # Noise data points

def linear(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b

def generator(input, h_dim):
    """
    生成网络，从随机噪声中生成与真实图像相类似的图像（伪造图像）
    :param input: 随机噪声值（随机值） 必须是2-D 
    :param h_dim: 隐藏层数
    :return: 与真实图像形状一样的伪造图像 shape 2-D
    """
    with tf.variable_scope('G'):
        input=tf.reshape(input,[-1,14,14,1]) # [n,14,14,1]
        w=tf.get_variable('w',[3,3,16,1],dtype=tf.float32,initializer=tf.random_uniform_initializer) # 注意 w 输入1 输出16，反卷积所以要反过来
        b = tf.get_variable('b', [16], dtype=tf.float32, initializer=tf.random_uniform_initializer)
        output_shape=[int(input.get_shape()[0]),28,28,16] # [n,28,28,16]
        h=tf.nn.conv2d_transpose(input, w, output_shape, [1,2,2,1], padding='SAME', name=None)
        h=tf.nn.relu(tf.nn.bias_add(h,b)) # [n,28,28,16]
        h=tf.reshape(h,[-1,28*28*16])
    h0 = tf.nn.softplus(linear(h, h_dim, 'g0'))
    h1 = linear(h0, image_dim, 'g1')
    return h1

def discriminator(input, h_dim, minibatch_layer=True):
    """
    分类器网络，尽可能的区分出真实图像与伪造图像，如果是真实图像返回True 虚假图像返回FALSE，2类
    :param input: 真实图像或生产网络生产的伪造图像 2-D
    :param h_dim: 隐藏层
    :param minibatch_layer: 
    :return: True or False 2类 2-D
    """
    input=tf.reshape(input,[-1,14,14,1])
    with tf.variable_scope('D'):
        w=tf.get_variable('w',[3,3,1,16],dtype=tf.float32,initializer=tf.random_uniform_initializer)
        b = tf.get_variable('b', [16], dtype=tf.float32, initializer=tf.random_uniform_initializer)
        h=tf.nn.conv2d(input,w,[1,2,2,1],padding="SAME")
        h=tf.nn.relu(tf.nn.bias_add(h,b)) # [n,28,28,16]
    input=tf.reshape(h,[-1,28*28*16])

    h0 = tf.nn.relu(linear(input, h_dim * 2, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.nn.relu(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3')) # True or False 2类 可以加激活函数 sigmoid
    return h3

def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - \
        tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)

def optimizer(loss, var_list):
    learning_rate = 0.00001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer

def log(x):
    '''
    Sometimes discriminiator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimisation.
    '''
    return tf.log(tf.maximum(x, 1e-5))

class GAN(object):
    def __init__(self, params):
        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(params.batch_size, z_dim))
            self.G = generator(self.z, params.hidden_size)

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.z).
        #
        # Here we create two copies of the discriminator network
        # that share parameters, as you cannot use the same network with
        # different inputs in TensorFlow.
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, image_dim))
        with tf.variable_scope('D'):
            self.D1 = discriminator(
                self.x,
                params.hidden_size,
                params.minibatch
            )
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(
                self.G,
                params.hidden_size,
                params.minibatch
            )

        # Define the loss for discriminator and generator networks
        # (see the original paper for details), and create optimizers for both
        self.loss_d = tf.reduce_mean(-log(self.D1) - log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-log(self.D2))

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)

def train(model, params):
    # anim_frames = []

    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        for step in range(params.num_steps + 1):
            # update discriminator
            # x = data.sample(params.batch_size)
            x, _ = mnist.train.next_batch(params.batch_size)
            # x=X
            # z = gen.sample(params.batch_size)
            # Generate noise to feed to the generator
            z = np.random.uniform(-1., 1., size=[params.batch_size, z_dim])
            loss_d, _, = session.run([model.loss_d, model.opt_d], {
                model.x: np.reshape(x, (params.batch_size, image_dim)),
                model.z: np.reshape(z, (params.batch_size, z_dim))
            })

            # update generator
            # z = gen.sample(params.batch_size)
            z = np.random.uniform(-1., 1., size=[params.batch_size, z_dim])
            loss_g, _ = session.run([model.loss_g, model.opt_g], {
                model.z: np.reshape(z, (params.batch_size, z_dim))
            })

            if step % params.log_every == 0:
                print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))

        plt.figure('GAN by mnist')
        images=session.run(model.G,{model.z: np.reshape(z, (params.batch_size, z_dim))})[:4].reshape([-1,28,28])
        # print(images[0])
        plt.subplot(2,2,1)
        plt.imshow(images[0])
        plt.subplot(2, 2, 2)
        plt.imshow(images[1])
        plt.subplot(2, 2, 3)
        plt.imshow(images[2])
        plt.subplot(2, 2, 4)
        plt.imshow(images[3])
        plt.show()

        #     if params.anim_path and (step % params.anim_every == 0):
        #         anim_frames.append(
        #             samples(model, session, data, gen.range, params.batch_size)
        #         )
        #
        # if params.anim_path:
        #     save_animation(anim_frames, params.anim_path, gen.range)
        # else:
        #     samps = samples(model, session, data, gen.range, params.batch_size)
        #     plot_distributions(samps, gen.range)

def main(args):
    model = GAN(args)
    train(model, args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=1000,
                        help='the number of training steps to take')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='MLP hidden size')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--minibatch', action='store_true',
                        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=100,
                        help='print loss after this many steps')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
