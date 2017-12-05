#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.
This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.
The script should take about a minute to run.
参考：https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_and_convert_flowers.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

# The URL where the Flowers data can be downloaded.
_DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
# 只需将该下载路径换成其他图像文件路径，即可对其他图像进行下载和数据转换（转成tfr） ，要求图像的通道为3，格式为jpg
# 也可以通过修改ImageReader实现对其他数据的转换,如：使用PIL，scipy，skimage,opencv等替换

# The number of images in the validation set.
_NUM_VALIDATION = 350

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.
  返回所有图片对应的路径以及文件名(对应每一类)
  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.
  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  flower_root = os.path.join(dataset_dir, 'flower_photos')
  directories = []
  class_names = []
  for filename in os.listdir(flower_root):
    path = os.path.join(flower_root, filename)
    if os.path.isdir(path): # 如果是文件夹
      directories.append(path)
      class_names.append(filename) # 文件名对应类名 用于做标签

  photo_filenames = [] # 存储了所有图片的路径
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename) #图片完整路径
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names) #


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'flowers_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename) # 返回tfr文件路径


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.  图像转成tfr数据
  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images. 存放所有图片的路径 list
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS))) # 将总的文件分成_NUM_SHARDS份，每一份是num_per_shard个图片

  with tf.Graph().as_default(): # 新建图表
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames)) # 每num_per_shard个图片生成一个tfr文件
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id)) # 输出进度条
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()  # 数据类型也是bytes ，如果不是 使用tobytes转换
            height, width = image_reader.read_image_dims(sess, image_data)

            # 这里可以使用PIL，scipy，skimage，opencv等对图像操作，image_data需要用tobytes转成bytes格式
            # 如：
            # from PIL import Image
            # image_data=Image(filenames[i]).convert('RGB')
            # height, width=image_data.size
            # image_data=image_data.tobytes()



            class_name = os.path.basename(os.path.dirname(filenames[i])) # 类名（文件夹名）
            class_id = class_names_to_ids[class_name] # 换成对应的类的id 如 0,1等

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.
  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1] # 得到 'flower_photos.tgz'
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath) # 删除下载的文件

  tmp_dir = os.path.join(dataset_dir, 'flower_photos') #
  tf.gfile.DeleteRecursively(tmp_dir) # 删除解压的文件夹


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir):
  """Runs the download and conversion operation.
  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir): # 路径不存在
    tf.gfile.MakeDirs(dataset_dir) # 新建

  if _dataset_exists(dataset_dir): # 查看tfr数据是否已经存在
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names)))) # 将类别名与id对应

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames) # 将文件路径名随机打乱，实现数据打乱
  training_filenames = photo_filenames[_NUM_VALIDATION:] # 取其余做训练
  validation_filenames = photo_filenames[:_NUM_VALIDATION] # 350个做验证

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, class_names_to_ids,
                   dataset_dir) # 转成tfr数据
  _convert_dataset('validation', validation_filenames, class_names_to_ids,
                   dataset_dir) # 转成tfr数据

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names)) # 让id与文件名（类别）对应起来
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the Flowers dataset!')
