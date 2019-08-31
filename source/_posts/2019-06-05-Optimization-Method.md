---
title: Optimization Method
tags:
- 优化算法
mathjax: true
copyright: true
date: 2019-06-05 18:05:24
categories: 机器学习
---

## 0 优化算法框架

1. 计算目标函数关于当前参数的梯度
  $$
  g_t = \nabla f(w_t)
  \tag{0-1}
  $$

2. 根据历史梯度计算一阶动量和二阶动量
  $$
  m_t = \phi (g_1, g_2, \cdots, g_t)
  \tag{0-2}
  $$
  $$
  V_t = \psi (g_1, g_2, \cdots, g_t)
  \tag{0-3}
  $$

3. 计算当前时刻的下降梯度
  $$
  \eta_t = \frac{\alpha}{\sqrt{V_t}} \cdot m_t
  \tag{0-4}
  $$

4. 根据下降梯度进行更新
  $$
  w_{t+1} = w_t - \eta_t
  \tag{0-5}
  $$

**核心区别是第3步执行的下降方向，在这个式子中，前半部分是实际的学习率（也即下降步长），后半部分是实际的下降方向。不同优化算法也就是不断地在这两部分上做文章。下文会将重要的地方标红显示**


## 1. Gradient Descent Variants

### 1.1 Batch Gradient Descent

1. 计算目标函数关于当前参数的梯度
  $$
  g_t = \nabla f(w_t)
  \tag{1.1-1}
  $$

2. 根据历史梯度计算一阶动量和二阶动量
  $$
  \color{red} {m_t = g_t}
  \tag{1.1-2}
  $$
  $$
  \color{red} {V_t = 1}
  \tag{1.1-3}
  $$

3. 计算当前时刻的下降梯度
  $$
  \eta_t = \frac{\alpha}{\sqrt{V_t}} \cdot m_t = \alpha \cdot g_t
  \tag{1.1-4}
  $$

4. 根据下降梯度进行更新
  $$
  \begin{align*}
  w_{t+1} &= w_t - \eta_t \\
  &= w_t - \alpha \cdot g_t
  \end{align*}
  \tag{1.1-5}
  $$
  


### 1.2 Stochastic Gradient Descent

1. 计算目标函数关于当前参数的梯度
  $$
  \color{red}  { g_t = \nabla f(w_t; x^{(i)}, y^{(i)}) } 
  \tag{1.2-1}
  $$

  其中$(x^{(i)}, y^{(i)})$表示第$i$个样本；

2. 根据历史梯度计算一阶动量和二阶动量
  $$
  \color{red} {m_t = g_t}
  \tag{1.2-2}
  $$
  $$
  \color{red} {V_t = 1}
  \tag{1.2-3}
  $$

3. 计算当前时刻的下降梯度
  $$
  \eta_t = \frac{\alpha}{\sqrt{V_t}} \cdot m_t = \alpha \cdot g_t
  \tag{1.2-4}
  $$

4. 根据下降梯度进行更新
  $$
  \begin{align*}
  w_{t+1} &= w_t - \eta_t \\ &= w_t - \alpha \cdot g_t
  \end{align*}
  \tag{1.2-5}
  $$


### 1.3 Mini-Bach Gradient Descent

1. 计算目标函数关于当前参数的梯度
  $$
  \color{red}  { g_t = \nabla f(w_t; x^{(i:i+n)}, y^{(i:i+n)}) } 
  \tag{1.3-1}
  $$

  其中$(x^{(i:i+n)}, y^{(i:i+n)})$表示第$i$个样本到第$i+n$个样本，$n$表示mini-batch的大小；

2. 根据历史梯度计算一阶动量和二阶动量
  $$
  \color{red} {m_t = g_t}
  \tag{1.3-2}
  $$
  $$
  \color{red} {V_t = 1}
  \tag{1.3-3}
  $$

3. 计算当前时刻的下降梯度
  $$
  \eta_t = \frac{\alpha}{\sqrt{V_t}} \cdot m_t = \alpha \cdot g_t
  \tag{1.3-4}
  $$

4. 根据下降梯度进行更新
  $$
  \begin{align*}
  w_{t+1} &= w_t - \eta_t \\ &= w_t - \alpha \cdot g_t
  \end{align*}
  \tag{1.3-5}
  $$


**上述算法存在的问题：**
- 很难调整出一个合适的learning_rate
- learning_rate的变化规则往往是预定义的，很难适应不同的数据
- 所有的特征共享相同的learning_rate
- 局部最有解的问题


---------------------------------------------------------------


## 2. Gradient Descent Optimization Algorithm

### 2.1 Gradient Descent with Momentum

1. 计算目标函数关于当前参数的梯度
  $$
  g_t = \nabla f(w_t)
  \tag{2.1-1}
  $$

2. 根据历史梯度计算一阶动量和二阶动量，计算当前时刻下降梯度(*将框架的第2步和第3步合并*)
  $$
  \color{red} {m_t = \gamma \cdot m_{t-1} + \alpha \cdot g_t }
  \tag{2.1-2&3}
  $$

  $$
  \color{red} { \eta_t = m_t }
  \tag{2.1-4}
  $$

3. 根据下降梯度进行更新
  $$
  \begin{align*}
  w_{t+1} &= w_t - \eta_t \\ &= w_t - ( \gamma \cdot m_{t-1} + \alpha \cdot g_t )
  \end{align*}
  \tag{2.1-5}
  $$

一阶动量是移动平均值，这里 $\gamma $ 的经验值为`0.9`。
以历史动量的和为主，适当加上一点当前的偏移，即考虑惯性的因素。


### 2.2 Nesterov Accelerated Gradient

1. 计算目标函数关于当前参数+下次变化的梯度
  $$
  \color{red} { g_t = \nabla f(w_t - \gamma m_{t-1}) }
  \tag{2.2-1}
  $$

2. 根据历史梯度计算一阶动量和二阶动量，计算当前时刻下降梯度(*将框架的第2步和第3步合并*)
  $$
  \color{red} {m_t = \gamma \cdot m_{t-1} + \alpha \cdot g_t }
  \tag{2.2-2&3}
  $$

  $$
  \color{red} { \eta_t = m_t }
  \tag{2.2-4}
  $$

3. 根据下降梯度进行更新
  $$
  \begin{align*}
  w_{t+1} &= w_t - \eta_t \\ &= w_t - ( \gamma \cdot m_{t-1} + \alpha \cdot g_t )
  \end{align*}
  \tag{2.2-5}
  $$

这里 $\gamma $ 的经验值为`0.9`，在对参数求导时，不再和之前方法一直，而是对下次参数求导，即多向前看一步可以用来指导当前怎么走。


### 2.3 AdaGrad

1. 计算目标函数关于当前参数的梯度
  $$
  g_t = \nabla f(w_t)
  \tag{2.3-1}
  $$

2. 根据历史梯度计算一阶动量和二阶动量
  $$
  m_t = g_t
  \tag{2.3-2}
  $$
  $$
  \color{red} {V_t = \sum_{\tau = 1}^t g_{\tau}^2 }
  \tag{2.3-3}
  $$

3. 计算当前时刻的下降梯度
  $$
  \begin{align*}
  \eta_t &= \frac{\alpha}{\sqrt{V_t}} \cdot m_t \\ &= \frac{\alpha}{\sqrt{ \sum_{\tau = 1}^t g_{\tau}^2 + \epsilon }} \cdot m_t
  \end{align*}
  \tag{2.3-4}
  $$

  这里$\epsilon$是为了避免分母为$0$，通常设置为$1e-8$。

4. 根据下降梯度进行更新
  $$
  \begin{align*}
  w_{t+1} &= w_t - \eta_t \\ &= w_t - \frac{\alpha}{\sqrt{ \sum_{\tau = 1}^t g_{\tau}^2 + \epsilon }} \cdot m_t
  \end{align*}
  \tag{2.3-5}
  $$

不同的参数具有不同的梯度，从而达到了不同参数具有不同学习率的目的，这样梯度更新快的参数，学习率(步长)会渐渐变小。
同样最终会面临学习率过小，模型无法继续学习的问题。


### 2.4 AdaDelta

首先定义动态平均值 $\color{red}{ E[ g^2 ]\_t = \gamma E[ g^2 ]\_{t-1} + (1 - \gamma) g_t^2 }$，该值仅取决于当前梯度值与上一时刻的动态平均值，其中$\gamma$通常设置成$0.9$。

1. 计算目标函数关于当前参数的梯度
  $$
  g_t = \nabla f(w_t)
  \tag{2.4-1}
  $$

2. 根据历史梯度计算一阶动量和二阶动量
  $$
  m_t = g_t
  \tag{2.4-2}
  $$
  $$
  \color{red} {V_t = E[ g^2 ]_t }
  \tag{2.4-3}
  $$

3. 计算当前时刻的下降梯度
  $$
  \begin{align*}
  \eta_t &= \frac{\alpha}{\sqrt{V_t}} \cdot m_t \\ &= \frac{\alpha}{\sqrt{ E[ g^2 ]_t + \epsilon }} \cdot m_t
  \end{align*}
  \tag{2.4-4.1}
  $$

  这里$\epsilon$是为了避免分母为$0$。将分母$ \sqrt{ E[ g^2 ]\_t + \epsilon} $ 记为 $\color{red}{ RMS[g]\_t} $，定义
  $$
  E[ \Delta g^2 ] _t = \gamma E[ \Delta g^2 ] _{t-1} + (1 - \gamma) \Delta g _t^2
  $$

  则：
  $$
  RMS[\Delta g] _t = \sqrt{ E[ \Delta g^2 ] _t + \epsilon }
  $$

  用$RMS[\Delta g]_{t-1}$代替学习率$\alpha$，则式$(2.4-4.1)$可以转化为:
  $$
  \begin{align*}
  \eta_t &= \frac{\alpha}{RMS[g] _t} \cdot g_t \\ &= \frac{ RMS[\Delta g] _{t-1} }{ RMS[g] _t } \cdot g _t
  \end{align*}
  \tag{2.4-4.2}
  $$

4. 根据下降梯度进行更新
  $$
  \begin{align*}
  w_{t+1} &= w_t - \eta_t \\ &= w_t - \frac{RMS[\Delta g] _{t-1}}{RMS[g] _t} \cdot g_t
  \end{align*}
  \tag{2.4-5}
  $$

$\color{red}{为什么}$用$RMS[\Delta g]_{t-1}$代替学习率$\alpha$??

### 2.5 RMSprop

RMSprop是AdaDelta算法的一个特例。

$$
E[g^2] _t = 0.9 E[g^2] _{t-1} + 0.1 g^2_t
$$

$$
w_{t+1} = w_t - \frac{\alpha}{ \sqrt{ E[g^2] _t + \epsilon } } g_t
$$

Hinton建议$\gamma$设置成$0.9$，学习率设置成$0.001$。


### 2.6 Adam

1. 计算目标函数关于当前参数的梯度
  $$
  g_t = \nabla f(w_t)
  \tag{2.6-1}
  $$

2. 根据历史梯度计算一阶动量和二阶动量
  $$
  \color{red}{ \hat{m_t} = \beta_1 \cdot \hat{m_{t-1}} + (1 - \beta_1) \cdot g_t } \\
  \color{red}{ m_t = \frac{\hat{m_t}}{1 - \beta_1^t} }
  \tag{2.6-2}
  $$
  $$
  \color{red}{ \hat{V_t} = \beta_2 \cdot \hat{V_{t-1}} + (1 - \beta_2) \cdot g_t^2 } \\
  \color{red}{ V_t = \frac{\hat{V_t}}{ 1 - \beta_2^t } }
  \tag{2.6-3}
  $$

  其中的$\beta_1$控制一阶动量，$\beta_2$控制二阶动量；

3. 计算当前时刻的下降梯度
  $$
  \begin{align*}
  \eta_t &= \frac{\alpha}{\sqrt{V_t}} \cdot m_t
  \end{align*}
  \tag{2.6-4}
  $$

  其中增加的$\epsilon$为了防止分母等于$0$；

4. 根据下降梯度进行更新
  $$
  \begin{align*}
  w_{t+1} &= w_t - \eta_t
  \end{align*}
  \tag{2.6-5}
  $$

  作者建议默认值$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 1e-8$。


### 2.7 AdaMax

Adamax是Adam的一种变体，此方法对学习率的上限提供了一个更简单的范围。

1. 计算目标函数关于当前参数的梯度
  $$
  g_t = \nabla f(w_t)
  \tag{2.7-1}
  $$

2. 根据历史梯度计算一阶动量和二阶动量
  $$
  \color{red}{ m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t }
  \tag{2.7-2}
  $$
  $$
  \color{red}{ 
  \begin{align*}
  V_t &= \beta_2^{\infty} V_{t-1} + (1 - \beta_2^{\infty}) \| g_t \| ^{\infty} \\
  &= max( \beta_2 \cdot V_{t-1}, \| g_t \| )
  \end{align*}
  }
  \tag{2.7-3}
  $$

  其中的$\beta_1$控制一阶动量，$\beta_2$控制二阶动量；

3. 计算当前时刻的下降梯度
  $$
  \color{red}{
  \begin{align*}
  \eta_t &= \frac{\alpha}{V_t} \cdot m_t \\
  &= \frac{\alpha}{ max( \beta_2 \cdot V_{t-1}, \| g_t \| ) } \cdot \{ \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \}
  \end{align*}
  }
  \tag{2.7-4}
  $$

4. 根据下降梯度进行更新
  $$
  \begin{align*}
  w_{t+1} &= w_t - \eta_t \\ 
  &= w_t - \frac{\alpha}{ max( \beta_2 \cdot V_{t-1}, \| g_t \| ) } \cdot \{ \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \}
  \end{align*}
  \tag{2.7-5}
  $$

  论文说合适的默认值为$\alpha = 0.002, \beta_1 = 0.9, \beta_2 = 0.999$。


### 2.8 Nadam

1. 计算目标函数关于当前参数的梯度
  $$
  \color{red}{ g_t = \nabla f(w_t - \frac{\alpha}{\sqrt{V_t}} \cdot m_{t-1}) }
  \tag{2.8-1}
  $$

2. 根据历史梯度计算一阶动量和二阶动量
  $$
  \color{red}{ \hat{m_t} = \gamma \cdot \hat{m_{t-1}} + (1 - \beta_1) \cdot g_t } \\
  \color{red}{ m_t = \frac{\hat{m_t}}{ 1 - \beta_1^t } }
  \tag{2.8-2}
  $$
  $$
  \color{red}{ \hat{V_t} = \beta_2 \cdot \hat{V_{t-1}} + (1 - \beta_2) \cdot g_t^2 } \\
  \color{red}{ V_t = \frac{\hat{V_t}}{ 1 - \beta_2^t } }
  \tag{2.8-3}
  $$

  其中的$\beta_1$控制一阶动量，$\beta_2$控制二阶动量；

3. 计算当前时刻的下降梯度
  $$
  \begin{align*}
  \eta_t &= \frac{\alpha}{\sqrt{V_t} + \epsilon } \cdot m_t
  \end{align*}
  \tag{2.8-4}
  $$

  其中增加的$\epsilon$为了防止分母等于$0$；

4. 根据下降梯度进行更新
  $$
  \begin{align*}
  w_{t+1} &= w_t - \eta_t
  \end{align*}
  \tag{2.8-5}
  $$


-----------------------

## 3. Other Optimization Algorithm

- 将不满足阈值的系数设置为0

### 3.1 Truncated Gradient

- 简单截断法的改进

### 3.2 FOBOS

- 微调标准梯度下降的结果

### 3.3 RDA

- 历史梯度加权平均
- 正则项对特征稀疏化
- 严格递增序列

### 3.4 FTRL

- 确保新的权重和历史权重不偏离太远
- L1正则稀疏性约束


------------

> - [一个框架看懂优化算法之异同 SGD/AdaGrad/Adam](https://zhuanlan.zhihu.com/p/32230623)
> - [SGD、Momentum、RMSprop、Adam区别与联系](https://zhuanlan.zhihu.com/p/32488889)
> - [An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf)
> - [在线最优化求解(Online Optimization)-冯扬]()
> - [比Momentum更快：揭开Nesterov Accelerated Gradient的真面目](https://zhuanlan.zhihu.com/p/22810533)
> - [深度学习最全优化方法总结比较（SGD，Adagrad，Adadelta，Adam，Adamax，Nadam）](https://zhuanlan.zhihu.com/p/22252270)
> - [Deep Learning 最优化方法之AdaGrad](https://zhuanlan.zhihu.com/p/29920135)
