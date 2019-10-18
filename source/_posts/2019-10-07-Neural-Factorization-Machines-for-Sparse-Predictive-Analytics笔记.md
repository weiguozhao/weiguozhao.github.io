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
```


> 1. [Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/abs/1708.05027)
> 2. [论文笔记《Neural Factorization Machines for Sparse Predictive Analytics》](https://blog.csdn.net/u014475479/article/details/81630959)

> [backup](/posts_res/2019-10-07-Neural-Factorization-Machines-for-Sparse-Predictive-Analytics笔记/NFM.py)
