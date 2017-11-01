空间转换（图像姿态转换）

参考：
https://github.com/tensorflow/models/tree/master/research/transformer

![这里写图片描述](https://camo.githubusercontent.com/bb81d6267f2123d59979453526d958a58899bb4f/687474703a2f2f692e696d6775722e636f6d2f4578474456756c2e706e67)

```
transformer(U, theta, out_size)
```

Parameters

```
U : float 
    The output of a convolutional net should have the
    shape [num_batch, height, width, num_channels]. 
theta: float   
    The output of the
    localisation network should be [num_batch, 6].
out_size: tuple of two ints
    The size of the output of the network
```

Notes

To initialize the network to the identity transform init theta to :

```
identity = np.array([[1., 0., 0.],
                    [0., 1., 0.]]) 
identity = identity.flatten()
theta = tf.Variable(initial_value=identity)
```

作用：
对数据进行各种姿态变换，实现数据的预处理

