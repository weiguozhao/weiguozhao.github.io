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
&= \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n <v_i, v_j> x_i x_j - \frac{1}{2} \sum_{i=1}^n <v_i, v_j> x_i x_j \\
&= \frac{1}{2} \lgroup \sum_{i=1}^n \sum_{j=1}^n \sum_{f=1}^k v_{i, f} v_{j,f} x_i x_j - \sum_{i=1}^n \sum_{f=1}^k v_{i,f} v_{j,f} x_i x_j \rgroup \\
&= \frac{1}{2} \sum_{f=1}^k \lgroup \lgroup \sum_{i=1}^n v_{i,f} x_i \rgroup \lgroup \sum_{j=1}^n v_{j,f} x_j \rgroup - \sum_{i=1}^n v_{i,f}^2 x_i^2 \rgroup \\
&= \frac{1}{2} \sum_{f=1}^k \lgroup \lgroup \sum_{i=1}^n v_{i,f} x_i \rgroup^2 - \sum_{i=1}^n v_{i,f}^2 x_i^2 \rgroup
\end{aligned}
$$

最终FM的预测公式为：

$$
\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^{n}w_i x_i + \frac{1}{2} \sum_{f=1}^{k} ((\sum_{i=1}^{n}v_{i,f} x_i)^2-\sum_{i=1}^{n}v_{i,f}^2 x_i^2)
$$

MSE为：
$$
L = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda_w ||W||^2 + \lambda_v ||V||^2
$$

[tensorflow实现代码](https://github.com/babakx/fm_tensorflow/blob/master/fm_tensorflow.ipynb)为：

其中`p`为特征维度，`k`为$v$的维度

```python
x = tf.placeholder('float', [None, p])
y = tf.placeholder('float', [None, 1])

# bias
w0 = tf.Variable(tf.zeros([1]))
# 一阶权重
w = tf.Variable(tf.zeros([p]))
# 二阶交叉权重
v = tf.Variable(tf.random_normal([k, p], mean=0, stddev=0.01))

# bias + 一阶线性乘积结果
linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(w, x), 1, keep_dims=True))  # n * 1
# 二阶交叉结果
pair_interactions = 0.5 * tf.reduce_sum(
    tf.subtract(
        tf.pow(tf.matmul(x, tf.transpose(v)), 2),
        tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(v, 2)))
    ), axis=1, keep_dims=True)

# FM预测结果
y_hat = tf.add(linear_terms, pair_interactions)

# 权重正则项系数
lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')
l2_norm = tf.reduce_sum(
    tf.add(
        tf.multiply(lambda_w, tf.pow(w, 2)),
        tf.multiply(lambda_v, tf.pow(v, 2))
    )
)

# label的误差
error = tf.reduce_mean(tf.square(y - y_hat))
loss = tf.add(error, l2_norm)

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
```

论文及工程地址：

> 1. [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
> 2. [fm_tensorflow](https://github.com/babakx/fm_tensorflow/blob/master/fm_tensorflow.ipynb)

> [backup](/posts_res/2019-10-05-Factorization-Machines/FM_model.py.pdf)
