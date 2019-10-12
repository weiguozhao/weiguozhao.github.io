---
title: TensorFlow-Embedding笔记
tags:
  - TensorFlow
  - Embedding
mathjax: true
comments: false
copyright: true
date: 2019-10-12 19:42:42
categories: 深度学习
---

**embedding中牢记feature_batch中的value表示的都是embedding矩阵中的index**


### 1. tf 1.x中的Embedding实现

使用embedding_lookup函数来实现Emedding，如下：

```python
# embedding matrix 4x4
embedding = tf.constant(
    [
        [0.21, 0.41, 0.51, 0.11],
        [0.22, 0.42, 0.52, 0.12],
        [0.23, 0.43, 0.53, 0.13],
        [0.24, 0.44, 0.54, 0.14]
    ], dtype=tf.float32)

feature_batch = tf.constant([2, 3, 1, 0])
# 相当于把行号变化，原来index=2的变为index=0，index=3->1, index=1->2, index=3->0
get_embedding1 = tf.nn.embedding_lookup(embedding, feature_batch)
# 生成 4x4 的one-hot矩阵，1的位置有feature_batch中的值决定
feature_batch_one_hot = tf.one_hot(feature_batch, depth=4)
# embedding层其实是一个全连接神经网络层，那么其过程等价于：
# 矩阵乘法，描述了 embedding_lookup 的原理
get_embedding2 = tf.matmul(feature_batch_one_hot, embedding)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    embedding1, embedding2, batchOneHot = sess.run([get_embedding1, get_embedding2, feature_batch_one_hot])
    print("embedding1\n", embedding1)
    print("embedding2\n", embedding2)
    print("feature_batch_one_hot\n", batchOneHot)
```

代码运行结果：

```text
embedding1
 [[0.23 0.43 0.53 0.13]
 [0.24 0.44 0.54 0.14]
 [0.22 0.42 0.52 0.12]
 [0.21 0.41 0.51 0.11]]
embedding2
 [[0.23 0.43 0.53 0.13]
 [0.24 0.44 0.54 0.14]
 [0.22 0.42 0.52 0.12]
 [0.21 0.41 0.51 0.11]]
feature_batch_one_hot
 [[0. 0. 1. 0.]
 [0. 0. 0. 1.]
 [0. 1. 0. 0.]
 [1. 0. 0. 0.]]
```

<img src="/posts_res/2019-10-12-tensorflow-embedding笔记/1.jpg" />


### 2. tf 1.x中与Embedding类似操作 - 单维索引

```python
# 单维索引
embedding = tf.constant(
    [
        [0.21, 0.41, 0.51, 0.11],
        [0.22, 0.42, 0.52, 0.12],
        [0.23, 0.43, 0.53, 0.13],
        [0.24, 0.44, 0.54, 0.14]
    ], dtype=tf.float32)

index_a = tf.Variable([2, 3, 1, 0])
gather_a = tf.gather(embedding, index_a)
# axis=0: 按行取， axis=1：按列取
gather_a_axis1 = tf.gather(embedding, index_a, axis=1)

b = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
index_b = tf.Variable([2, 4, 6, 8])
gather_b = tf.gather(b, index_b)
# 一维gather直接使用 tf.gather
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gather_a, gather_b, gather_a_axis1 = sess.run([gather_a, gather_b, gather_a_axis1])
    print("gather_a\n", gather_a)
    print("gather_b\n", gather_b)
    print("gather_a_axis1\n", gather_a_axis1)
```

代码运行结果：

```text
gather_a
 [[0.23 0.43 0.53 0.13]
 [0.24 0.44 0.54 0.14]
 [0.22 0.42 0.52 0.12]
 [0.21 0.41 0.51 0.11]]
gather_b
 [3 5 7 9]
gather_a_axis1
 [[0.51 0.11 0.41 0.21]
 [0.52 0.12 0.42 0.22]
 [0.53 0.13 0.43 0.23]
 [0.54 0.14 0.44 0.24]]
```

### 3. tf 1.x中与Embedding类似操作 - 多维索引

```python
# 多维索引
a = tf.Variable([[1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15]])
index_a = tf.Variable([2])

b = tf.get_variable(name='b',
                    shape=[3, 3, 2],
                    initializer=tf.random_normal_initializer)
# [0, 1, 1]表示第一维选择index=0，第二维选择index=1，第三维选择index=1
index_b = tf.Variable([[0, 1, 1],
                       [2, 2, 0]])
# 注意多维gather要使用 tf.gather_nd
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("tf.gather_nd(a, index_a)\n", sess.run(tf.gather_nd(a, index_a)))
    print("b\n", sess.run(b))
    print("tf.gather_nd(b, index_b)\n", sess.run(tf.gather_nd(b, index_b)))
```

代码运行结果：

```text
tf.gather_nd(a, index_a)
 [11 12 13 14 15]
b
 [[[-3.9612162e-01  2.6312229e-04]
  [-1.5676118e-01  5.1959008e-01]
  [ 2.1008211e-01  1.3654137e+00]]

 [[ 3.5585640e-03 -1.0942048e+00]
  [-1.1834130e+00 -8.3026028e-01]
  [-5.5713791e-01 -1.7461585e-01]]

 [[-7.2083569e-01 -4.4790068e-01]
  [ 4.5505306e-01 -1.4471538e-01]
  [ 1.2432091e+00 -8.3164376e-01]]]
tf.gather_nd(b, index_b)
 [0.5195901 1.2432091]
```

### 4. tf 1.x中与Embedding类似操作 - 稀疏表示的Embedding

```python
# sparse embedding 稀疏表示的embedding
a = tf.SparseTensor(indices=[[0, 0],
                             [1, 2],
                             [1, 3]],
                    values=[1, 2, 3],
                    dense_shape=[2, 4])
b = tf.sparse_tensor_to_dense(a)

embedding = tf.constant(
    [
        [0.21, 0.41, 0.51, 0.11],
        [0.22, 0.42, 0.52, 0.12],
        [0.23, 0.43, 0.53, 0.13],
        [0.24, 0.44, 0.54, 0.14]
    ], dtype=tf.float32)
# b
#  [[1 0 0 0]
#  [0 0 2 3]]
# 使用 embedding_lookup_sparse，而不是 embedding_lookup
# [1 0 0 0] 表示将index=1的那一行拿出来
# [0 0 2 3] 表示将index=2和index=3的那两行拿出来，对应位置做 combiner
embedding_sparse = tf.nn.embedding_lookup_sparse(embedding, sp_ids=a, sp_weights=None, combiner='mean')
c = tf.matmul(tf.cast(b, tf.float32), embedding)
e = tf.reduce_sum(b, axis=1, keepdims=True)
l = tf.div(c, tf.cast(e, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("a\n", sess.run(a))
    print("embedding_sparse\n", sess.run(embedding_sparse))
    print("b\n", sess.run(b))
    print("c\n", sess.run(c))
    print("e\n", sess.run(e))
    print("l\n", sess.run(l))
```

代码运行结果：

```text
a
 SparseTensorValue(indices=array([[0, 0],
       [1, 2],
       [1, 3]]), values=array([1, 2, 3], dtype=int32), dense_shape=array([2, 4]))
embedding_sparse
 [[0.22       0.42       0.52       0.12      ]
 [0.235      0.435      0.53499997 0.13499999]]
b
 [[1 0 0 0]
 [0 0 2 3]]
c
 [[0.21      0.41      0.51      0.11     ]
 [1.18      2.1799998 2.68      0.68     ]]
e
 [[1]
 [5]]
l
 [[0.21       0.41       0.51       0.11      ]
 [0.23599999 0.43599996 0.536      0.136     ]]
```

### 5. tf 2.0中Embedding实现

在tf2.0中，embedding同样可以通过embedding_lookup来实现，不过不同的是，我们不需要通过sess.run来获取结果了，可以直接运行结果，并转换为numpy。

```python
embedding = tf.constant(
    [
        [0.21, 0.41, 0.51, 0.11],
        [0.22, 0.42, 0.52, 0.12],
        [0.23, 0.43, 0.53, 0.13],
        [0.24, 0.44, 0.54, 0.14]
    ], dtype=tf.float32)

feature_batch = tf.constant([2, 3, 1, 0])
get_embedding1 = tf.nn.embedding_lookup(embedding, feature_batch)

feature_batch_one_hot = tf.one_hot(feature_batch, depth=4)
get_embedding2 = tf.matmul(feature_batch_one_hot, embedding)

print(get_embedding1.numpy().tolist())
print(get_embedding2.numpy().tolist())
```

代码运行结果：

```
embedding1
 [[0.23000000417232513, 0.4300000071525574, 0.5299999713897705, 0.12999999523162842], 
 [0.23999999463558197, 0.4399999976158142, 0.5400000214576721, 0.14000000059604645], 
 [0.2199999988079071, 0.41999998688697815, 0.5199999809265137, 0.11999999731779099], 
 [0.20999999344348907, 0.4099999964237213, 0.5099999904632568, 0.10999999940395355]]
embedding2
 [[0.23000000417232513, 0.4300000071525574, 0.5299999713897705, 0.12999999523162842], 
 [0.23999999463558197, 0.4399999976158142, 0.5400000214576721, 0.14000000059604645], 
 [0.2199999988079071, 0.41999998688697815, 0.5199999809265137, 0.11999999731779099], 
 [0.20999999344348907, 0.4099999964237213, 0.5099999904632568, 0.10999999940395355]]
```

神经网络中使用embedding层，推荐使用Keras：

```python
from tensorflow.keras import layers

num_classes=10

input_x = tf.keras.Input(shape=(None,),)
embedding_x = layers.Embedding(num_classes, 10)(input_x)
hidden1 = layers.Dense(50,activation='relu')(embedding_x)
output = layers.Dense(2,activation='softmax')(hidden1)

x_train = [2,3,4,5,8,1,6,7,2,3,4,5,8,1,6,7,2,3,4,5,8,1,6,7,2,3,4,5,8,1,6,7,2,3,4,5,8,1,6,7,2,3,4,5,8,1,6,7,2,3,4,5,8,1,6,7,2,3,4,5,8,1,6,7]
y_train = [0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0,1]

model2 = tf.keras.Model(inputs = input_x,outputs = output)
model2.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

history = model2.fit(x_train, y_train, batch_size=4, epochs=1000, verbose=0)
```

----------------

> [tensorflow中的Embedding操作详解](https://zhuanlan.zhihu.com/p/85802954)