# -*- coding: utf-8 -*-
""" DCGAN Example
Use a deep convolutional generative adversarial network (DCGAN) to generate
digit images from a noise distribution.
References:
    - Unsupervised representation learning with deep convolutional generative
    adversarial networks. A Radford, L Metz, S Chintala. arXiv:1511.06434.
Links:
    - [DCGAN Paper](https://arxiv.org/abs/1511.06434).

https://github.com/tflearn/tflearn/blob/master/examples/images/dcgan.py
"""

from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tflearn

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data()
X = np.reshape(X, newshape=[-1, 28, 28, 1])

z_dim = 200 # Noise data points
total_samples = len(X)


# Generator
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tflearn.fully_connected(x, n_units=7 * 7 * 128)
        x = tflearn.batch_normalization(x)
        x = tf.nn.tanh(x)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        x = tflearn.upsample_2d(x, 2) # [n,7*2,7*2,128]
        x = tflearn.conv_2d(x, 64, 5, activation='tanh') # [n,14,14,64]
        x = tflearn.upsample_2d(x, 2) # [n,14*2,14*2,64]
        x = tflearn.conv_2d(x, 1, 5, activation='sigmoid') # [n,28,28,1]
        return x


# Discriminator
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tflearn.conv_2d(x, 64, 5, activation='tanh') # [n,28,28,64]
        x = tflearn.avg_pool_2d(x, 2) # [n,14,14,64]
        x = tflearn.conv_2d(x, 128, 5, activation='tanh') # [n,14,14,128]
        x = tflearn.avg_pool_2d(x, 2) # [n,7,7,128]
        x = tflearn.fully_connected(x, 1024, activation='tanh') # [n,1024]
        x = tflearn.fully_connected(x, 2) # [n,2] 2类 0 False ;1 True
        x = tf.nn.softmax(x)
        return x


# Input Data
gen_input = tflearn.input_data(shape=[None, z_dim], name='input_gen_noise')
input_disc_noise = tflearn.input_data(shape=[None, z_dim], name='input_disc_noise')
input_disc_real = tflearn.input_data(shape=[None, 28, 28, 1], name='input_disc_real')

# Build Discriminator
disc_fake = discriminator(generator(input_disc_noise)) # [n,200]->[n,28,28,1]->[n,2]
disc_real = discriminator(input_disc_real, reuse=True) # [n,28,28,1]->[n,2]
disc_net = tf.concat([disc_fake, disc_real], axis=0) # [2n,2]~~[n,2]
# Build Stacked Generator/Discriminator
gen_net = generator(gen_input, reuse=True) # [n,200]->[n,28,28,1]
stacked_gan_net = discriminator(gen_net, reuse=True) # [n,28,28,1]->[n,2]

# Build Training Ops for both Generator and Discriminator.
# Each network optimization should only update its own variable, thus we need
# to retrieve each network variables (with get_layer_variables_by_scope).
disc_vars = tflearn.get_layer_variables_by_scope('Discriminator') # 获取该节点下的所有可训练的变量
# We need 2 target placeholders, for both the real and fake image target.
disc_target = tflearn.multi_target_data(['target_disc_fake', 'target_disc_real'],
                                        shape=[None, 2])
disc_model = tflearn.regression(disc_net, optimizer='adam',
                                placeholder=disc_target,
                                loss='categorical_crossentropy',
                                trainable_vars=disc_vars,
                                batch_size=64, name='target_disc',
                                op_name='DISC')

gen_vars = tflearn.get_layer_variables_by_scope('Generator')
gan_model = tflearn.regression(stacked_gan_net, optimizer='adam',
                               loss='categorical_crossentropy',
                               trainable_vars=gen_vars,
                               batch_size=64, name='target_gen',
                               op_name='GEN')

# Define GAN model, that output the generated images.
gan = tflearn.DNN(gan_model)

# Training
# Prepare input data to feed to the discriminator
disc_noise = np.random.uniform(-1., 1., size=[total_samples, z_dim])
# Prepare target data to feed to the discriminator (0: fake image, 1: real image)
y_disc_fake = np.zeros(shape=[total_samples])
y_disc_real = np.ones(shape=[total_samples])
y_disc_fake = tflearn.data_utils.to_categorical(y_disc_fake, 2)
y_disc_real = tflearn.data_utils.to_categorical(y_disc_real, 2)

# Prepare input data to feed to the stacked generator/discriminator
gen_noise = np.random.uniform(-1., 1., size=[total_samples, z_dim])
# Prepare target data to feed to the discriminator
# Generator tries to fool the discriminator, thus target is 1 (e.g. real images)
y_gen = np.ones(shape=[total_samples])
y_gen = tflearn.data_utils.to_categorical(y_gen, 2)

# Start training, feed both noise and real images.
gan.fit(X_inputs={'input_gen_noise': gen_noise,
                  'input_disc_noise': disc_noise,
                  'input_disc_real': X},
        Y_targets={'target_gen': y_gen, # 对应生成网络，让生成的图像与真实图像尽量一直，目标是1
                   'target_disc_fake': y_disc_fake, # 对应分类网络，尽量将伪造的图像区分出 目标为0
                   'target_disc_real': y_disc_real}, # 对应分类网络，尽量将真实图像区分出 目标为1
        n_epoch=10)

# Create another model from the generator graph to generate some samples
# for testing (re-using same session to re-use the weights learnt).
gen = tflearn.DNN(gen_net, session=gan.session)

f, a = plt.subplots(4, 10, figsize=(10, 4))
for i in range(10):
    # Noise input.
    z = np.random.uniform(-1., 1., size=[4, z_dim])
    g = np.array(gen.predict({'input_gen_noise': z}))
    for j in range(4):
        # Generate image from noise. Extend to 3 channels for matplot figure.
        img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                         newshape=(28, 28, 3))
        a[j][i].imshow(img)

f.show()
plt.draw()
plt.waitforbuttonpress()
