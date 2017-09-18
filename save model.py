#!/usr/bin/python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse

tf.reset_default_graph()

"""
save model
http://blog.csdn.net/wc781708249/article/details/78013690
"""


parser = argparse.ArgumentParser()
parser.add_argument("-md", "--model_name", help="The model name",type=str,default="model.ckpt")
args = parser.parse_args()
print("args:",args)

logdir='./output/'

x=tf.Variable([[3,4],[4,5]],dtype=tf.float32,name='x')

sess=tf.InteractiveSession(graph=tf.get_default_graph())
saver = tf.train.Saver()

initial_step=0
# 验证之前是否已经保存了检查点文件
ckpt = tf.train.get_checkpoint_state(logdir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    initial_step=int(ckpt.model_checkpoint_path.rsplit('-',1)[1])
else:
    tf.global_variables_initializer().run()

print(x.eval())
assign_op = tf.assign(x, x + 1)
sess.run(assign_op)

saver.save(sess, logdir + args.model_name, global_step=initial_step+10)
print('-------------')
sess.close()
