---
layout: post
title: 梯度下降
date: 2018-05-20 12:10 +0800
categories: 机器学习
tags:
- 优化算法
mathjax: true
copyright: true
---

目录

* 1 批量梯度下降BGD
* 2 随机梯度下降SGD
* 3 小批量梯度下降MBGD

------

在应用机器学习算法时，通常采用梯度下降法来对采用的算法进行训练。其实，常用的梯度下降法还具体包含有三种不同的形式，它们也各自有着不同的优缺点。

下面以线性回归算法来对三种梯度下降法进行比较。

一般线性回归函数的假设函数为：
\\[
h\_{\theta} = \sum\_{j=1}^n \theta\_j x\_j
\\]

对应的损失函数形式为：
\\[
J\_{train}(\theta) = \frac{1}{2m} \sum\_{i=1}^m (h\_{\theta} (x^{(i)}) - y^{(i)})^2
\\]

其中的\\( \frac{1}{2} \\)是为了计算方便，上标表示第 i 个样本，下标 j 表示第 j 维度。

下图为一个二维参数\\( \theta\_0 \\) 和 \\( \theta\_1 \\) 组对应损失函数的可视化图：

![theta01](/posts_res/2018-05-20-gradientdescent/0-1.png)


------

### 1. 批量梯度下降BGD

批量梯度下降法(Batch Gradient Descent，BGD)是梯度下降法最原始的形式，它的具体思路是在更新每一参数时都使用所有的样本来进行更新。
具体的操作流程如下：

* 1 随机初始化\\( \theta \\)；
* 2 更新 \\( \theta \\)使得损失函数减小，直到满足要求时停止；
\\[
\theta\_j = \theta\_j - \alpha \frac{\partial}{\partial \theta\_j} J(\theta)
\\]
这里\\( \alpha \\)表示学习率。

\\[
\begin{equation}
\begin{aligned}
\frac{\partial}{\partial \theta\_j} J(\theta)
& = \frac{\partial}{\partial \theta\_j} \frac{1}{2} (h\_{\theta}(x) - y )^2 \\\
& = 2 \cdot \frac{1}{2} ( h\_{\theta}(x) - y ) \cdot \frac{\partial}{\partial \theta\_j} ( h\_{\theta}(x) - y ) \\\
& = (h\_{\theta}(x) - y ) \cdot \frac{\partial}{\partial \theta\_j} \left( \sum\_{i=0}^n \theta\_i x\_i - y  \right) \\\
& = (h\_{\theta}(x) - y) x\_j
\end{aligned}
\end{equation}
\\]

则对所有数据点，上述损失函数的偏导(累加和)为：
\\[
\frac{\partial J(\theta)}{\partial \theta\_j} = - \frac{1}{m} \sum\_{i=1}^m (y^{(i)} - h\_{\theta}(x^{(i)})) x\_j^{(i)}
\\]

在最小化损失函数的过程中，需要不断反复的更新\\( \theta \\)使得误差函数减小，更新方式如下：
\\[
\theta\_j^{'} = \theta\_j + \frac{1}{m} \sum\_{i=1}^m ( y^{(i)} - h\_{\theta} (x^{(i)}) x\_j^{(i)}
\\]

BGD得到的是一个全局最优解，但是每迭代一步，都要用到训练集所有的数据，如果样本数目 m 很大，那么这种方法的迭代速度很慢！
所以，这就引入了另外一种方法，随机梯度下降。

* **优点：全局最优解；易于并行实现**
* **缺点：样本数目很多时，训练过程很慢**

从迭代的次数上来看，BGD迭代的次数相对较少。其迭代的收敛曲线示意图可以表示如下：

![bgd](/posts_res/2018-05-20-gradientdescent/1-1.png)


--------

### 2. 随机梯度下降SGD

由于批量梯度下降法在更新每一个参数时，都需要所有的训练样本，所以训练过程会随着样本数量的加大而变得异常的缓慢。
随机梯度下降法(Stochastic Gradient Descent，SGD)正是为了解决批量梯度下降法这一弊端而提出的。

将上面的损失函数写为如下形式：
\\[
\theta\_j^{'} = \theta\_j + ( y^{(i)} - h\_{\theta} (x^{(i)}) ) x\_j^{(i)}
\\]

利用每个样本的损失函数对 \\( \theta \\) 求偏导得到对应的梯度，来更新 \\( \theta \\)：

随机梯度下降是通过每个样本来迭代更新一次，如果样本量很大的情况（例如几十万），那么可能只用其中几万条或者几千条的样本，就已经将\\( \theta\\)迭代到最优解了，
对比上面的批量梯度下降，迭代一次需要用到十几万训练样本，一次迭代不可能最优，如果迭代10次的话就需要遍历训练样本10次。
但是，SGD伴随的一个问题是噪音较BGD要多，使得SGD并不是每次迭代都向着整体最优化方向。

* **优点：训练速度快**
* **缺点：准确度下降，不是全局最优；不易于并行实现**

从迭代的次数上来看，SGD迭代的次数较多，在解空间的搜索过程看起来很盲目。其迭代的收敛曲线示意图可以表示如下：

![sgd](/posts_res/2018-05-20-gradientdescent/2-1.png)


-----

### 3. 小批量梯度下降MBGD

有上述的两种梯度下降法可以看出，其各自均有优缺点，那么能不能在两种方法的性能之间取得一个折衷呢？即算法的训练过程比较快，而且也要保证最终参数训练的准确率，而这正是小批量梯度下降法(Mini-batch Gradient Descent，MBGD)的初衷。

MBGD在每次更新参数时使用 b 个样本( b一般为10 )，其具体的伪代码形式为：

![mbgd](/posts_res/2018-05-20-gradientdescent/3-1.png)


-----

### 4. 总结

1. BGD：每次迭代使用所有的样本；
2. SGD：每次迭代使用一个样本；
3. MBGD：每次迭代使用 b 个样本；


------

### 参考

>
1. [[Machine Learning] 梯度下降法的三种形式BGD、SGD以及MBGD](http://www.cnblogs.com/maybe2030/p/5089753.html)
2. [详解梯度下降法的三种形式BGD、SGD以及MBGD](https://zhuanlan.zhihu.com/p/25765735)

