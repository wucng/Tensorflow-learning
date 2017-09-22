#!/usr/bin/python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

slim = tf.contrib.slim

# input [28,28,1]
def inception_resnet_v2(inputs, num_classes=10, is_training=True,
                        dropout_keep_prob=0.7,
                        reuse=None,
                        scope='InceptionResnetV2'):
    """Creates the Inception Resnet V2 model.
      Args:
        inputs: a 4-D tensor of size [batch_size, height, width, 3].
        num_classes: number of predicted classes.
        is_training: whether is training or not.
        dropout_keep_prob: float, the fraction to keep before final layer.
        reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.
        scope: Optional variable_scope.
      Returns:
        logits: the logits outputs of the model.
        end_points: the set of end_points from the inception model.
      """
    end_points = {}

    with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # 28 x 28 x 32
                net = slim.conv2d(inputs, 32, 3, stride=1, padding='SAME',
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net

                # 14 x 14 x 32
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME',
                                      scope='MaxPool_1a_3x3')
                end_points['MaxPool_1a_3x3'] = net

                # 14 x 14 x 64
                net = slim.conv2d(inputs, 64, 3, stride=1, padding='SAME',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net

                # 7 x 7 x 64
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME',
                                      scope='MaxPool_2a_3x3')
                end_points['MaxPool_2a_3x3'] = net

                end_points['PrePool'] = net
                '''
                net = slim.flatten(net)
                net = slim.fully_connected(net, 1024, activation_fn=None,
                                           scope='fc') # 找不到变量fc
                '''
                # <-----------------------------
                weights = tf.Variable(tf.truncated_normal([7 * 7 * 64 * 4, 1024], stddev=0.001))
                biases = tf.Variable(tf.zeros([1024]))
                # 如果直接使用slim变量名不在图中，就不能通过变量名来保存变量了
                net = tf.nn.relu(tf.matmul(tf.reshape(net, [-1, 7 * 7 * 64 * 4]), weights) + biases, name='fc')
                # ------------------------------>
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='Dropout')
                end_points['PreLogitsFlatten'] = net
                with tf.variable_scope('Logits'):
                    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                                  scope='Logits')
                    end_points['Logits'] = logits
                    # end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
                    softmax = tf.nn.softmax(logits, name='softmax')
                    end_points['Predictions'] = softmax

    return logits, end_points

def inception_resnet_v2_arg_scope(weight_decay=0.00004,
                                  batch_norm_decay=0.9997,
                                  batch_norm_epsilon=0.001):
  """Yields the scope with the default parameters for inception_resnet_v2.
  Args:
    weight_decay: the weight decay for weights variables.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.
  Returns:
    a arg_scope with the parameters needed for inception_resnet_v2.
  """
  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as scope:
      return scope
