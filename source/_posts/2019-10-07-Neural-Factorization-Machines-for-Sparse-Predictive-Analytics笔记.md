---
title: Neural Factorization Machines for Sparse Predictive Analytics笔记
tags:
  - 模型算法
mathjax: true
comments: false
copyright: true
date: 2019-10-07 20:23:11
categories: 推荐系统
---

NFM充分结合了FM提取的二阶线性特征与神经网络提取的高阶非线性特征。总得来说，FM可以被看作一个没有隐含层的NFM，故NFM肯定比FM更具表现力。

FM预测公式：

$$
\hat{y}(\mathbf{x})_{FM} = w_0 + \sum_{i=1}^{n}w_i x_i + \frac{1}{2} \sum_{f=1}^{k} ((\sum_{i=1}^{n}v_{i,f} x_i)^2-\sum_{i=1}^{n}v_{i,f}^2 x_i^2)
$$

NFM预测公式：

$$
\hat{y}(\mathbf{x})_{NFM} = w_0 + \sum_{i=1}^{n}w_i x_i + f(\mathbf{x})
$$

其中第$1$项与第$2$项是与FM相似的线性回归部分，第3项是NFM的核心部分，它由一个如下图所示的网络结构组成：

<img src="/posts_res/2019-10-07-Neural-Factorization-Machines-for-Sparse-Predictive-Analytics笔记/1.png" />


- **Embedding Layer**
  - 该层是一个全连接层，将稀疏的向量给压缩表示。假设我们有 $v\_i \in R^k$ 为第 $i$ 个特征的embedding向量，那么在经过该层之后，我们得到的输出为 $ \lbrace x\_1 v\_1, \cdots, x\_n v\_n \rbrace$, 注意，该层本质上是一个全连接层，不是简单的`embedding lookup`。
- **Bi-Interaction Layer**
  - 上层得到的输出是一个特征向量的`embedding`的集合，本层本质上是做一个`pooling`的操作，让这个`embedding`向量集合变为一个向量，公式如下：

  $$
  f_{BI} (\mathcal{V}_x) = \sum_{i=1}^n \sum_{j=i+1}^n x_i v_i \bigodot x_j v_j
  $$

  - 其中 $\bigodot$ 代表两个向量对应的元素相乘。显然，该层的输出向量为 $k$ 维，本层采用的`pooling`方式与传统的`max pool`和`average pool`一样都是线性复杂度的，上式可以变换为:

  $$ 
  f_{BI}(\mathcal{V}_x) = \frac{1}{2} \lbrack (\sum_{i=1}^n x_i v_i)^2 - \sum_{i=1}^n (x_i v_i)^2 \rbrack 
  $$

  - 上式中用 $v\_2$ 来表示 $v \bigodot v$ ，其实本层本质上就是一个`FM`算法。
- **Hidden Layer**
  - 普通的全连接层。
- **Prediction Layer**
  - 将`Hidden Layer`的输出经过全连接层，得到最终的Score。

[NFM_tensorflow]()实现代码：

```python
# this is code
# Set graph level random seed
tf.set_random_seed(self.random_seed)
# Input data.
self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
self.train_phase = tf.placeholder(tf.bool)

# Variables.
self.weights = self._initialize_weights()

# Model.
# _________ sum_square part _____________
# get the summed up embeddings of features.
nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)
self.summed_features_emb = tf.reduce_sum(nonzero_embeddings, 1)  # None * K
# get the element-multiplication
self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

# _________ square_sum part _____________
self.squared_features_emb = tf.square(nonzero_embeddings)
self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

# ________ FM __________
self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
if self.batch_norm:
    self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_fm')
self.FM = tf.nn.dropout(self.FM, self.dropout_keep[-1])  # dropout at the bilinear interactin layer

# ________ Deep Layers __________
for i in range(0, len(self.layers)):
    self.FM = tf.add(tf.matmul(self.FM, self.weights['layer_%d' % i]),
                     self.weights['bias_%d' % i])  # None * layer[i] * 1
    if self.batch_norm:
        self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase,
                                        scope_bn='bn_%d' % i)  # None * layer[i] * 1
    self.FM = self.activation_function(self.FM)
    self.FM = tf.nn.dropout(self.FM, self.dropout_keep[i])  # dropout at each Deep layer
self.FM = tf.matmul(self.FM, self.weights['prediction'])  # None * 1

# _________out _________
Bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1
self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features),
                                  1)  # None * 1
Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
self.out = tf.add_n([Bilinear, self.Feature_bias, Bias])  # None * 1

# Compute the loss.
if self.loss_type == 'square_loss':
    if self.lambda_bilinear > 0:
        self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(
            self.lambda_bilinear)(self.weights['feature_embeddings'])  # regulizer
    else:
        self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
elif self.loss_type == 'log_loss':
    self.out = tf.sigmoid(self.out)
    if self.lambda_bilinear > 0:
        self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weight=1.0, epsilon=1e-07,
                                               scope=None) + tf.contrib.layers.l2_regularizer(
            self.lambda_bilinear)(self.weights['feature_embeddings'])  # regulizer
    else:
        self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weight=1.0, epsilon=1e-07,
                                               scope=None)

# Optimizer.
if self.optimizer_type == 'AdamOptimizer':
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                            epsilon=1e-8).minimize(self.loss)
elif self.optimizer_type == 'AdagradOptimizer':
    self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                               initial_accumulator_value=1e-8).minimize(self.loss)
elif self.optimizer_type == 'GradientDescentOptimizer':
    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
elif self.optimizer_type == 'MomentumOptimizer':
    self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
        self.loss)
```


> 1. [Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/abs/1708.05027)
> 2. [论文笔记《Neural Factorization Machines for Sparse Predictive Analytics》](https://blog.csdn.net/u014475479/article/details/81630959)
> 3. [hexiangnan/neural_factorization_machine](https://github.com/hexiangnan/neural_factorization_machine)
