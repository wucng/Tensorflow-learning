from scipy import ndimage
import tensorflow as tf
from spatial_transformer import transformer
import numpy as np
import matplotlib.pyplot as plt

# %% Create a batch of three images (1600 x 1200)
# %% Image retrieved from:
# %% https://raw.githubusercontent.com/skaae/transformer_network/master/cat.jpg
im = ndimage.imread('cat.jpg')
im = im / 255.
im = im.reshape(1, 1200, 1600, 3)
im = im.astype('float32')

# %% Let the output size of the transformer be half the image size.
out_size = (600, 800)

# %% Simulate batch
batch = np.append(im, im, axis=0)
batch = np.append(batch, im, axis=0)
num_batch = 3

x = tf.placeholder(tf.float32, [None, 1200, 1600, 3])
x = tf.cast(batch, 'float32')

# %% Create localisation network and convolutional layer
with tf.variable_scope('spatial_transformer_0'):

    # %% Create a fully-connected layer with 6 output nodes
    n_fc = 6
    W_fc1 = tf.Variable(tf.zeros([1200 * 1600 * 3, n_fc]), name='W_fc1')

    # %% Zoom into the image
    initial = np.array([[0.5, 0, 0], [0, 0.5, 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()

    b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
    h_fc1 = tf.matmul(tf.zeros([num_batch, 1200 * 1600 * 3]), W_fc1) + b_fc1
    h_trans = transformer(x, h_fc1, out_size)

# %% Run session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y = sess.run(h_trans, feed_dict={x: batch})

# plt.imshow(y[0])
plt.figure()
plt.subplot(1,2,1)
plt.imshow(im[0])

plt.subplot(1,2,2)
plt.imshow(y[0])
plt.show()