#!/usr/bin/python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.python.framework import graph_util

slim = tf.contrib.slim

"""
https://github.com/kwotsin/transfer_learning_tutorial
"""

logdir='./output/'

images=tf.placeholder(tf.float32,[None,28,28,1],name='input_x')
labels=tf.placeholder(tf.int64,[None,],name='input_y')

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

logits,end_points=inception_resnet_v2(images, num_classes=10, is_training=True)

one_hot_labels=slim.one_hot_encoding(labels, 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
# total_loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

# Now we can define the optimizer that takes on the learning rate
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)

# Create the train_op.
# train_op = slim.learning.create_train_op(total_loss, optimizer)
train_op=optimizer.minimize(loss)

# State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
predictions = tf.argmax(end_points['Predictions'], 1)
# probabilities = end_points['Predictions']
# accuracy, _ = tf.contrib.metrics.streaming_accuracy(predictions, labels)
correct_prediction = tf.equal(predictions, labels)
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
saver=tf.train.Saver()
# 验证之前是否已经保存了检查点文件
ckpt = tf.train.get_checkpoint_state(logdir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(128)
    batch_xs=np.reshape(batch_xs,[-1,28,28,1])
    train_op.run({images:batch_xs,labels:batch_ys})
    if step%100==0:
        print('step',step,'loss',loss.eval({images:batch_xs,labels:batch_ys}),
              'acc',accuracy.eval({images:batch_xs,labels:batch_ys}))

# test acc
print('test acc',accuracy.eval({images:np.reshape(mnist.test.images[:1000],[-1,28,28,1]),labels:mnist.test.labels[:1000]}))

saver.save(sess,logdir+'model.ckpt')

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,['input_x','input_y',"InceptionResnetV2/fc",'InceptionResnetV2/Logits/softmax'])
with tf.gfile.FastGFile('mnist.pb', mode='wb') as f:
    f.write(constant_graph.SerializeToString())
