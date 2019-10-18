---
title: Field-aware Factorization Machines for CTR Prediction笔记
tags:
  - 模型算法
mathjax: true
comments: false
copyright: true
date: 2019-10-06 20:23:11
categories: 推荐系统
---

FFM(Field Factorization Machine)是在FM的基础上引入了`场（Field）`的概念而形成的新模型。
在FM中计算特征 $x\_i$ 与其他特征的交叉影响时, 使用的都是同一个隐向量 $v\_i$ 。
而FFM将特征按照事先的规则分为多个`场(Field)`, 特征 $x\_i$ 属于某个特定的`场`$f$。
每个特征将被映射为多个隐向量 $v\_{i,1}, v\_{i,2}, \cdots, v\_{i,f}$ , 每个隐向量对应一个场。
当两个特征 $x\_i, x\_j$ 组合时, 用对方对应的场对应的隐向量做内积:

$$
w_{i,j} = v_{i, f_j} \cdot v_{j, f_i}
$$

FFM 由于引入了场, 使得每两组特征交叉的隐向量都是独立的, 可以取得更好的组合效果, FM 可以看做只有一个场的 FFM。
在FFM中，每一维特征 $x\_i$，针对其它特征的每一种`field` $f\_j$，都会学习一个隐向量 $v\_{i, f\_j}$。
这比FM增加了隐向量的数量。

FFM 预测公式：

$$
\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_{i, f_j}, \mathbf{v}_{j, f_i} \rangle x_i x_j
$$

其中，$f\_j$ 是第 $j$ 个特征所属的`field`。如果隐向量的长度为 $k$，那么FFM的二次参数有 $nfk$ 个，远多于FM模型的 $nk$ 个。
此外，由于隐向量与`field`相关，FFM二次项并不能够化简，其预测复杂度是 $O(kn^2)$。

Yu-Chin Juan实现了一个C++版的FFM模型，源码可从[Github下载](https://github.com/ycjuan/libffm)。这个版本的FFM省略了常数项和一次项，模型方程如下:

$$
\phi(\mathbf{w}, \mathbf{x}) = \sum_{j_1, j_2 \in \mathcal{C}_2} \langle \mathbf{w}_{j_1, f_2}, \mathbf{w}_{j_2, f_1} \rangle \cdot x_{j_1} x_{j_2}
$$

其中，$\mathcal{C}\_2$是非零特征的二元组合，$j\_1$ 是特征，属于`field` $f\_1$，$w\_{j\_1,f\_2}$ 是特征 $j\_1$ 对`field` $f\_2$ 的隐向量。
此FFM模型采用`logistic loss`作为损失函数，和`L2惩罚项`，因此只能用于**二元分类**问题。

$$
\min_{\mathbf{w}} \sum_{i=1}^L \log \big( 1 + \exp\{ -y_i \phi (\mathbf{w}, \mathbf{x}_i ) \} \big) + \frac{\lambda}{2} \| \mathbf{w} \|^2
$$

其中，$y\_i \in {−1,1}$ 是第 $i$ 个样本的label，$L$ 是训练样本数量，$\lambda$ 是惩罚项系数。

[FFM_TensorFlow](https://github.com/drcut/FFM-tensorflow)实现代码:

```python
# this is code
```

算法流程及工程实现的trick，详细阅读 `参考文献[1]`。


> 1. [深入FFM原理与实践](https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html)
> 2. [Field-aware Factorization Machine](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)

> [backup](/posts_res/2019-10-06-Field-aware-Factorization-Machines-for-CTR-Prediction笔记/FM_model.py)
