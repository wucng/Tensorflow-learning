#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tflearn
from tflearn.datasets import cifar10
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batch_size", help="The batch size",type=int,default=128)
parser.add_argument("-do", "--droup_out", help="The droup out",type=float,default=0.7)
parser.add_argument("-lr", "--learn_rate", help="The learn rate",type=float,default=1e-3)
parser.add_argument("-ns", "--num_steps", help="The num steps",type=int,default=10000)
parser.add_argument("-ds", "--disply_step", help="The disp step",type=int,default=100)
parser.add_argument("-ipi", "--img_piexl", help="The image piexl",type=int,default=32)
parser.add_argument("-ch", "--channels", help="The image channels",type=int,default=3)
parser.add_argument("-nc", "--n_classes", help="The image n classes",type=int,default=10)
parser.add_argument("-tr", "--train", help="The train/test mode",type=int,default=1)# 1 train 0 test
parser.add_argument("-log", "--logdir", help="The model logdir",type=str,default="./output/")
parser.add_argument("-md", "--model_name", help="The model name",type=str,default="model.ckpt")
args = parser.parse_args()
print("args:",args)


batch_size=args.batch_size
droup_out = args.droup_out
learn_rate = args.learn_rate
# INITIAL_LEARNING_RATE=args.learn_rate
num_steps = args.num_steps
disp_step = args.disply_step
img_piexl=args.img_piexl
channels=args.channels
n_classes=args.n_classes
logdir=args.logdir

# with tf.Graph().as_default() as graph:
x=tf.placeholder(tf.float32,[None,img_piexl,img_piexl,channels])
# y_=tf.placeholder(tf.float32,[None,n_classes])
y_ = tf.placeholder(tf.int64, [None, ])
keep=tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool, name='MODE')

def batch_norm_layer(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: batch_norm(inputT, is_training=True,
                                      center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                   lambda: batch_norm(inputT, is_training=False,
                                      center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                                      scope=scope, reuse=True))

def conv2d(input, kernel_size, input_size, output_size, is_training, name):
    with tf.name_scope(name) as scope:
        with tf.variable_scope(name):
            # scope.reuse_variables()
            w = tf.get_variable('w', [kernel_size, kernel_size, input_size, output_size], tf.float32,
                                initializer=tf.random_uniform_initializer) * np.sqrt(2.0/input_size)
            b = tf.get_variable('b', [output_size], tf.float32, initializer=tf.random_normal_initializer) + 0.1
            conv = tf.nn.conv2d(input, w, [1, 1, 1, 1], padding="SAME")
            conv = tf.nn.bias_add(conv, b)
            conv = batch_norm_layer(conv, is_training, scope)
            conv = tf.nn.relu(conv)
    return conv

# convolution1
conv1=conv2d(tf.image.convert_image_dtype(x,tf.float32),
                # x,
                 3,3,32,is_training,'conv1')
conv1=tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding="SAME") # [n,16,16,16]
conv1 = tf.nn.dropout(conv1, keep)
conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                  name='norm1')

# convolution2
conv2 = conv2d(conv1,
                   3, 32, 64, is_training,'conv2')
conv2=tf.nn.max_pool(conv2,[1,2,2,1],[1,2,2,1],padding="SAME") # [n,8,8,32]
conv2 = tf.nn.dropout(conv2, keep)
conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                  name='norm2')

# conv3
conv3 = conv2d(conv2,
               3, 64, 128, is_training,'conv3')
conv3 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")  # [n,4,4,64]
conv3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
conv3 = tf.nn.dropout(conv3, keep)


# conv4
conv4 = conv2d(conv3,
               3, 128, 256, is_training,'conv4')
conv4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")  # [n,2,2,128]
conv4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
conv4 = tf.nn.dropout(conv4, keep)

# conv5
conv5 = conv2d(conv4,
               3, 256, 10, is_training,'conv5')
conv5 = tf.nn.max_pool(conv5, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")  # [n,1,1,10]
conv5 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
# conv5 = tf.nn.dropout(conv5, keep)

# output
y=tf.reshape(conv5,[-1,n_classes])

loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,logits=y))

train_op=tf.train.AdamOptimizer(learn_rate).minimize(loss)

# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
correct_prediction = tf.equal(tf.argmax(y, 1), y_)

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

sess=tf.InteractiveSession()
# saver = tf.train.Saver()
tf.global_variables_initializer().run()
# 验证之前是否已经保存了检查点文件
ckpt = tf.train.get_checkpoint_state(logdir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

# input datas
(X, Y), (testX, testY) = cifar10.load_data()


start_index=0
second_index=0
def next_batch(x,y,batch_size):
    global start_index  # 必须定义成全局变量
    global second_index  # 必须定义成全局变量

    second_index=start_index+batch_size
    if second_index>len(x):
        second_index=len(x)
    img=x[start_index:second_index]
    label=y[start_index:second_index]
    start_index=second_index
    if start_index>=len(x):
        start_index = 0

    # normal
    img-=np.mean(img,axis=0)
    img/=(np.std(img,axis=0)+0.0001)

    # 将每次得到batch_size个数据按行打乱
    index = [i for i in range(len(img))]  # len(data1)得到的行数
    np.random.shuffle(index)  # 将索引打乱
    img = img[index]
    label=label[index]

    return img,label


for step in range(num_steps):
    if step %4000:
        index = [i for i in range(len(X))]  # len(data1)得到的行数
        np.random.shuffle(index)  # 将索引打乱
        X = X[index]
        Y = Y[index]
    batch_xs,batch_ys=next_batch(X,Y,batch_size)
    train_op.run({x: batch_xs, y_: batch_ys, keep: droup_out, is_training: True})
    if step % disp_step == 0:
        print("step", step, 'acc', accuracy.eval({x: batch_xs, y_: batch_ys, keep: droup_out, is_training: True}),
              'loss',loss.eval({x: batch_xs, y_: batch_ys, keep: droup_out, is_training: True}))
saver.save(sess, logdir + args.model_name)

# acc=accuracy.eval({x: testX[:2000], y_: testY[:2000], keep: 1.,is_training:False})
acc=accuracy.eval({x: testX[:2000], y_: testY[:2000], keep: 1.,is_training:True})
print('test acc',acc)

