---
title: Factorization Machines笔记
tags:
  - 模型算法
mathjax: true
comments: false
copyright: true
date: 2019-10-05 20:23:11
categories: 推荐系统
---

FM预测公式：

$$
\hat{y}(x) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n <v_i, v_j> \cdot x_i x_y
$$

其中，
$$
<v_i, v_j> = \sum_{f=1}^{k} v_{i,f} \cdot v_{j,f}
$$

二阶交叉部分可以通过数学转化，降低计算复杂度：

$$
\begin{aligned}
pair\_interactions &= \sum_{i=1}^n \sum_{j=i+1}^n <v_i, v_j> x_i x_j \\
&= \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n <v_i, v_j> x_i x_j - \frac{1}{2} \sum_{i=1}^n <v_i, v_i> x_i x_i \\
&= \frac{1}{2} \lgroup \sum_{i=1}^n \sum_{j=1}^n \sum_{f=1}^k v_{i, f} v_{j,f} x_i x_j - \sum_{i=1}^n \sum_{f=1}^k v_{i,f} v_{i,f} x_i x_i \rgroup \\
&= \frac{1}{2} \sum_{f=1}^k \lgroup \lgroup \sum_{i=1}^n v_{i,f} x_i \rgroup \lgroup \sum_{j=1}^n v_{j,f} x_j \rgroup - \sum_{i=1}^n v_{i,f}^2 x_i^2 \rgroup \\
&= \frac{1}{2} \sum_{f=1}^k \lgroup \lgroup \sum_{i=1}^n v_{i,f} x_i \rgroup^2 - \sum_{i=1}^n v_{i,f}^2 x_i^2 \rgroup
\end{aligned}
$$

最终FM的预测公式为：

$$
\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^{n}w_i x_i + \frac{1}{2} \sum_{f=1}^{k} ((\sum_{i=1}^{n}v_{i,f} x_i)^2-\sum_{i=1}^{n}v_{i,f}^2 x_i^2)
$$

FM的训练复杂度，利用SGD（Stochastic Gradient Descent）训练模型。模型各个参数的梯度如下:

$$
\frac{\partial}{\partial\theta} y (\mathbf{x}) = 
\begin{cases}
1, & if \quad \theta \quad is \quad w_0 \\
x_i, & if \quad \theta \quad is \quad w_i \\
x_i \sum_{j=1}^n v_{j, f} x_j - v_{i, f} x_i^2, & if \quad \theta \quad v_{i, f}
\end{cases}
$$

其中, $v\_{j,f}$ 是隐向量 $v\_j$ 的第 $f$ 个元素。
由于 $\sum\_{j=1}^n v\_{j,f} x\_j$ 只与 $f$ 有关，而与 $i$ 无关，在每次迭代过程中，只需计算一次所有 $f$ 的 $\sum\_{j=1}^n v\_{j,f} x\_j$, 
就能够方便地得到所有 $v\_{i,f}$ 的梯度。显然，计算所有 $f$ 的 $\sum\_{j=1}^n v\_{j,f} x\_j$ 的复杂度是 $O(kn)$；
已知 $\sum\_{j=1}^n v\_{j,f} x\_j$ 时，计算每个参数梯度的复杂度是 $O(1)$；得到梯度后，更新每个参数的复杂度是 $O(1)$；
模型参数一共有 $nk + n + 1$ 个。因此，FM参数训练的复杂度也是 $O(kn)$。综上可知，FM可以在线性时间训练和预测，是一种非常高效的模型。

MSE为：
$$
L = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda_w ||W||^2 + \lambda_v ||V||^2
$$

[FM_TensorFlow]()为：

其中`p`为特征维度，`k`为$v$的维度，label是`one-hot`形式的

```python
"""
y'(x) = w0 + sum( wi * xi ) + 0.5 * sum( (vi xi)**2 - vi**2 * xi**2 )
"""
with tf.variable_scope('linear_layer'):
  # 单独的全局bias
  w0 = tf.get_variable(name='w0',
            shape=[self.num_classes],
            initializer=tf.zeros_initializer())
	# 线性乘积部分
	self.w = tf.get_variable(name='w',
               shape=[self.p, num_classes],
               initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
  # [n, feature_num] * [feature_num, num_classes] -> [n, num_classes]
  # [n, num_classes] + [feature_num] -> [n, num_classes]
  self.linear_terms = tf.add(tf.matmul(self.X, self.w), w0)

  with tf.variable_scope('interaction_layer'):
    # 特征交叉部分
    self.v = tf.get_variable(name='v',
                 shape=[self.p, self.k],
                 initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
    self.interaction_terms = tf.multiply(0.5,
                      tf.reduce_mean(tf.subtract(tf.pow(tf.matmul(self.X, self.v), 2),
                                    tf.matmul(self.X, tf.pow(self.v, 2))),
                      1, keep_dims=True))
  with tf.name_scope("predict_layer"):
    self.y_out = tf.add(self.linear_terms, self.interaction_terms)
    if self.num_classes == 2:
      self.y_out_prob = tf.nn.sigmoid(self.y_out)
    elif self.num_classes > 2:
      self.y_out_prob = tf.nn.softmax(self.y_out)
```

论文及工程地址：

> 1. [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
> 2. [fm_tensorflow](https://github.com/babakx/fm_tensorflow/blob/master/fm_tensorflow.ipynb)
> 3. [LLSean/data-mining](https://github.com/LLSean/data-mining/tree/master/fm)

