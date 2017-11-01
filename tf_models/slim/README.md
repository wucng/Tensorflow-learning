参考：
1、https://github.com/tensorflow/models/tree/master/research/slim
2、https://mp.weixin.qq.com/s/gUDJuS1HKkxUZaCHHUj7aw


----------
TF-Slim 是一个新的轻量的 TensorFlow 高级 API（tensorflow.contrib.slim），用于定义、训练、评估复杂模型。参考材料【1】包含几个广泛使用的 CNN 图像分类模型代码，可以利用提供的脚本从头训练一个模型，或者使用预训练的网络权值微调。TF-Slim 附带下载标准图像数据集、转换为 TensorFlow 原生 TFRecord 格式、使用 TF-Slim 数据读取队列等功能，可以很方便地在公开数据集上训练任意模型，包括 GoogLeNet【2】、Inception-v2【3】、Inception-v3【4】、Inception-v4【5】以及前两篇介绍的 MobileNet（用于移动和嵌入式视觉应用的 MobileNets）和 ShuffleNet（ShuffleNet——面向移动设备的极为高效的卷积神经网络）。


----------

```python
import tensorflow.contrib.slim as slim
# 然后就能使用slim API
```

# Preparing the datasets
![这里写图片描述](http://img.blog.csdn.net/20171101094912227?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# Downloading and converting to TFRecord format

```
$ DATA_DIR=/tmp/data/flowers
$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"
```
# Creating a TF-Slim Dataset Descriptor.

```python
import tensorflow as tf
from datasets import flowers

slim = tf.contrib.slim

# Selects the 'validation' dataset.
dataset = flowers.get_split('validation', DATA_DIR)

# Creates a TF-Slim DataProvider which reads the dataset in the background
# during both training and testing.
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])
```
# An automated script for processing ImageNet data.

```
# location of where to place the ImageNet data
DATA_DIR=$HOME/imagenet-data

# build the preprocessing script.
bazel build slim/download_and_preprocess_imagenet

# run it
bazel-bin/slim/download_and_preprocess_imagenet "${DATA_DIR}"
```

# Pre-trained Models
![这里写图片描述](http://img.blog.csdn.net/20171101095000480?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

```
$ CHECKPOINT_DIR=/tmp/checkpoints
$ mkdir ${CHECKPOINT_DIR}
$ wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
$ tar -xvf inception_v3_2016_08_28.tar.gz
$ mv inception_v3.ckpt ${CHECKPOINT_DIR}
$ rm inception_v3_2016_08_28.tar.gz
```



# Training a model from scratch

```
DATASET_DIR=/tmp/imagenet
TRAIN_DIR=/tmp/train_logs
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v3
```
# TensorBoard

```
tensorboard --logdir=${TRAIN_DIR}
```

# Fine-tuning a model from an existing checkpoint

```
$ DATASET_DIR=/tmp/flowers
$ TRAIN_DIR=/tmp/flowers-models/inception_v3
$ CHECKPOINT_PATH=/tmp/my_checkpoints/inception_v3.ckpt
$ python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=inception_v3 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits
```

# Evaluating performance of a model

```
CHECKPOINT_FILE = ${CHECKPOINT_DIR}/inception_v3.ckpt  # Example
$ python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=inception_v3
```

# Exporting the Inference Graph

```
$ python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=/tmp/inception_v3_inf_graph.pb

$ python export_inference_graph.py \
  --alsologtostderr \
  --model_name=mobilenet_v1 \
  --image_size=224 \
  --output_file=/tmp/mobilenet_v1_224.pb
```

# Freezing the exported Graph

```
bazel build tensorflow/python/tools:freeze_graph

bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=/tmp/inception_v3_inf_graph.pb \
  --input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
  --input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
  --output_node_names=InceptionV3/Predictions/Reshape_1
```
# Run label image in C++

```
bazel build tensorflow/examples/label_image:label_image

bazel-bin/tensorflow/examples/label_image/label_image \
  --image=${HOME}/Pictures/flowers.jpg \
  --input_layer=input \
  --output_layer=InceptionV3/Predictions/Reshape_1 \
  --graph=/tmp/frozen_inception_v3.pb \
  --labels=/tmp/imagenet_slim_labels.txt \
  --input_mean=0 \
  --input_std=255
```
# Troubleshooting
**The model runs out of CPU memory.**
See [Model Runs out of CPU memory](https://github.com/tensorflow/models/tree/master/research/inception#the-model-runs-out-of-cpu-memory).

**The model runs out of GPU memory.**

**The model training results in NaN's.**

**The ResNet and VGG Models have 1000 classes but the ImageNet dataset has 1001**

**I wish to train a model with a different image size.**

```
image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    preprocessing_name,
    height=MY_NEW_HEIGHT,
    width=MY_NEW_WIDTH,
    is_training=True)
```

**What hardware specification are these hyper-parameters targeted for?**


----------


> 【1】https://github.com/tensorflow/models/tree/master/research/slim
> 【6】https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py
