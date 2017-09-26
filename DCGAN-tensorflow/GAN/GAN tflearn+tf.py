from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tflearn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

X=mnist.train.images

image_dim = 784 # 28*28 pixels
z_dim = 200 # Noise data points
total_samples = len(X)
batch_size=128

# Generator
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tflearn.fully_connected(x, 256, activation='relu')
        x = tflearn.fully_connected(x, image_dim, activation='sigmoid')
        return x


# Discriminator
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tflearn.fully_connected(x, 256, activation='relu')
        x = tflearn.fully_connected(x, 1, activation='sigmoid')
        return x

def optimizer(loss, var_list):
    learning_rate = 0.000001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer
# Build Networks
# gen_input = tflearn.input_data(shape=[None, z_dim], name='input_noise')
# disc_input = tflearn.input_data(shape=[None, 784], name='disc_input')
gen_input=tf.placeholder(tf.float32,[None,z_dim], name='input_noise')
disc_input=tf.placeholder(tf.float32,[None,784], name='disc_input')

# with tf.variable_scope('G'):
#     self.z = tf.placeholder(tf.float32, shape=(params.batch_size, z_dim))
#     self.G = generator(self.z, params.hidden_size)
#
#     # The discriminator tries to tell the difference between samples from
#     # the true data distribution (self.x) and the generated samples
#     # (self.z).
#     #
#     # Here we create two copies of the discriminator network
#     # that share parameters, as you cannot use the same network with
#     # different inputs in TensorFlow.
# self.x = tf.placeholder(tf.float32, shape=(params.batch_size, image_dim))
# with tf.variable_scope('D'):
#     self.D1 = discriminator(
#         self.x,
#         params.hidden_size,
#         params.minibatch
#     )
# with tf.variable_scope('D', reuse=True):
#     self.D2 = discriminator(
#         self.G,
#         params.hidden_size,
#         params.minibatch
#     )
with tf.variable_scope('G'):
    gen_sample = generator(gen_input)
with tf.variable_scope('D'):
    disc_real = discriminator(disc_input)
with tf.variable_scope('D', reuse=True):
    disc_fake = discriminator(gen_sample, reuse=True)

# Define Loss
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
gen_loss = -tf.reduce_mean(tf.log(disc_fake))

vars = tf.trainable_variables()
d_params = [v for v in vars if v.name.startswith('D/')]
g_params = [v for v in vars if v.name.startswith('G/')]

opt_d = optimizer(disc_loss, d_params)
opt_g = optimizer(gen_loss, g_params)

with tf.Session() as session:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()

    for epoch in range(50):
        for step in range(total_samples//batch_size+1):
            # update discriminator
            x, _ = mnist.train.next_batch(batch_size)
            z = np.random.uniform(-1., 1., size=[batch_size, z_dim])
            loss_d, _, = session.run([disc_loss, opt_d], {
                disc_input: np.reshape(x, (batch_size, image_dim)),
                gen_input: np.reshape(z, (batch_size, z_dim))
            })

            # update generator
            z = np.random.uniform(-1., 1., size=[batch_size, z_dim])
            loss_g, _ = session.run([gen_loss, opt_g], {
                gen_input: np.reshape(z, (batch_size, z_dim))
            })

            if step % 200 == 0:
                print('{}: {}: {:.4f}\t{:.4f}'.format(epoch,step, loss_d, loss_g))

    # Generate images from noise, using the generator network.
    f, a = plt.subplots(2, 10, figsize=(10, 4))
    for i in range(10):
        for j in range(2):
            # Noise input.
            z = np.random.uniform(-1., 1., size=[1, z_dim])
            # Generate image from noise. Extend to 3 channels for matplot figure.
            temp = [[ii, ii, ii] for ii in session.run(gen_sample,{gen_input:z})]
            # temp=session.run(gen_sample,{gen_input:z})
            a[j][i].imshow(np.reshape(temp, (28, 28,3)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()