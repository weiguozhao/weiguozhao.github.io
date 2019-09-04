---
title: Optimization Method
tags:
- 优化算法
mathjax: true
copyright: true
date: 2019-06-05 18:05:24
categories: 机器学习
---

*本文是从各个论文、博客、专栏等学习整理所得,如有任何错误疏漏等问题,欢迎评论或邮箱提出,大家一起学习进步！*

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

**核心区别是第3步执行的下降方向,在这个式子中,前半部分是实际的学习率（也即下降步长）,后半部分是实际的下降方向。不同优化算法也就是不断地在这两部分上做文章。下文会将重要的地方标红显示**


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

  其中$(x^{(i:i+n)}, y^{(i:i+n)})$表示第$i$个样本到第$i+n$个样本,$n$表示mini-batch的大小；

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
- learning_rate的变化规则往往是预定义的,很难适应不同的数据
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

2. 根据历史梯度计算一阶动量和二阶动量,计算当前时刻下降梯度(*将框架的第2步和第3步合并*)
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

一阶动量是移动平均值,这里 $\gamma $ 的经验值为`0.9`。
以历史动量的和为主,适当加上一点当前的偏移,即考虑惯性的因素。


### 2.2 Nesterov Accelerated Gradient

1. 计算目标函数关于当前参数+下次变化的梯度
  $$
  \color{red} { g_t = \nabla f(w_t - \gamma m_{t-1}) }
  \tag{2.2-1}
  $$

2. 根据历史梯度计算一阶动量和二阶动量,计算当前时刻下降梯度(*将框架的第2步和第3步合并*)
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

这里 $\gamma $ 的经验值为`0.9`,在对参数求导时,不再和之前方法一直,而是对下次参数求导,即多向前看一步可以用来指导当前怎么走。


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

  这里$\epsilon$是为了避免分母为$0$,通常设置为$1e-8$。

4. 根据下降梯度进行更新
  $$
  \begin{align*}
  w_{t+1} &= w_t - \eta_t \\ &= w_t - \frac{\alpha}{\sqrt{ \sum_{\tau = 1}^t g_{\tau}^2 + \epsilon }} \cdot m_t
  \end{align*}
  \tag{2.3-5}
  $$

不同的参数具有不同的梯度,从而达到了不同参数具有不同学习率的目的,这样梯度更新快的参数,学习率(步长)会渐渐变小。
同样最终会面临学习率过小,模型无法继续学习的问题。


### 2.4 AdaDelta

首先定义动态平均值 $\color{red}{ E[ g^2 ]\_t = \gamma E[ g^2 ]\_{t-1} + (1 - \gamma) g_t^2 }$,该值仅取决于当前梯度值与上一时刻的动态平均值,其中$\gamma$通常设置成$0.9$。

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

  这里$\epsilon$是为了避免分母为$0$。将分母$ \sqrt{ E[ g^2 ]\_t + \epsilon} $ 记为 $\color{red}{ RMS[g]\_t} $,定义
  $$
  E[ \Delta g^2 ] _t = \gamma E[ \Delta g^2 ] _{t-1} + (1 - \gamma) \Delta g _t^2
  $$

  则：
  $$
  RMS[\Delta g] _t = \sqrt{ E[ \Delta g^2 ] _t + \epsilon }
  $$

  用$RMS[\Delta g]_{t-1}$代替学习率$\alpha$,则式$(2.4-4.1)$可以转化为:
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

Hinton建议$\gamma$设置成$0.9$,学习率设置成$0.001$。


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

  其中的$\beta_1$控制一阶动量,$\beta_2$控制二阶动量；

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

  作者建议默认值$\beta_1 = 0.9$,$\beta_2 = 0.999$,$\epsilon = 1e-8$。


### 2.7 AdaMax

Adamax是Adam的一种变体,此方法对学习率的上限提供了一个更简单的范围。

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

  其中的$\beta_1$控制一阶动量,$\beta_2$控制二阶动量；

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

  其中的$\beta_1$控制一阶动量,$\beta_2$控制二阶动量；

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

## 3. Online Optimization Algorithm

上面描述的主要是批量训练的优化方法，批量训练有自身的局限性：面对高维高数据量的时候，批量处理的方式就显得笨重和不够高效。因此需要有在线处理的方法(Online)来解决相同的问题。在线学习算法的特点是：每来一个训练样本，就用该样本产生的loss和梯度对模型迭代一次，一个一个数据地进行训练，因此可以处理大数据量训练和在线训练。

### 3.1 Truncated Gradient

#### 3.1.1 L1正则法

$$
w_{t+1} = w_{t} - \eta_t \cdot g_t - \eta_t \cdot \lambda \cdot sgn(w_t)
$$

其中,
- $\lambda \in \mathbb{R} $是一个标量,且$\lambda \ge 0$,为L1正则化参数；
- $sgn(v)$为符号函数；
- $\eta_t$为学习率,通常将其设置为$1/\sqrt{t}$的函数；
- $g_t = \nabla f(w_t) $；

#### 3.1.2 简单截断法

以$k$为窗口,当$t/k$不为整数时采用标准SGD进行迭代；当$t/k$为整数时,采用如下权重更新方式；

$$
w_{t+1} = T_0 \left( w_t - \eta_t g_t, \theta \right)
$$

$$
T_0 (v_i, \theta) = 
\begin{cases}
0 & \quad \mid v_i \mid \le \theta \\
v_i & \quad otherwise
\end{cases}
$$

$\theta \in \mathbb{R}$是一个标量,且$\theta \ge 0$；

#### 3.1.3 截断梯度法(TG)

简单截断法太过激进,因此TG在此基础上进行了改进：

$$
w_{t+1} = T_1 \left( w_t - \eta_t g_t, \quad \eta_t \lambda_t, \quad \theta \right)
$$

$$
T_1 (v_i, \alpha, \theta) = 
\begin{cases}
max(0, \quad v_i - \alpha ) & \quad v_i \in [0, \theta] \\
max(0, \quad v_i + \alpha ) & \quad v_i \in [- \theta, 0 ] \\
v_i & \quad otherwise
\end{cases}
\tag{3.1.3-1}
$$

- 其中$\lambda_t \in \mathbb{R}$,且$\lambda_t \ge 0$；
- TG 同样是以$k$为窗口,每$k$步进行一次截断。
  - 当$t/k$不为整数时,$\lambda_t = 0$；
  - 当$t/k$为整数时,$\lambda_t = k \cdot \lambda$。
- 从公式$(3.1.3-1)$可以看出,$\lambda$和$\theta$决定了$w$的稀疏程度,这两个值越大,则稀疏性越强；尤其令$\lambda = \theta$时,只需要通过调节一个参数就能控制稀疏性。

#### 3.1.4 截断公式对比

![compare](/posts_res/2019-06-05-Optimization-Method/1.png)

其中左侧是简单截断法的截断公式，右侧是截断梯度的截断公式。

### 3.2 FOBOS前向后向切分

$$
w_{t+1}^{(i)} = 
\begin{cases}
0 & \quad \mid w_t^{(i)} - \eta_t \cdot g_t^{(i)} \mid \le \eta_{t+\frac{1}{2}} \lambda \\
\left( w_t^{(i)} - \eta_t \cdot g_t^{(i)} \right) - \eta_{t+\frac{1}{2}} \cdot \lambda sgn(w_t^{(i)} - \eta_t g_t^{(i)}) & \quad otherwise
\end{cases}
\tag{3.2-1}
$$

<br>

式$(3.2-1)$截断的含义是：当一条样本产生的梯度不足以令对应维度上的权重值发生足够大的变化($\eta_{t+\frac{1}{2}} \cdot \lambda$),则认为在本次更新过程中该维度不够重要，应当令其权重为`0`。

同时式$(3.2-1)$与TG的特征权重更新公式$(3.1.3-1)$对比，
发现如果令

$$
\begin{align*}
\theta &= \infty \\
k &= 1 \\
\lambda _t^{(TG)} &= \eta _{t+\frac{1}{2}} \cdot \lambda 
\end{align*}
$$

则L1-FOBOS与TG完全一致，因此可以认为L1-FOBOS是TG在特定条件下的特殊形式。

### 3.3 RDA正则对偶平均


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
> - [在线最优化求解(Online Optimization)-冯扬](/posts_res/2019-06-05-Optimization-Method/online_optimization_fengyang.pdf)
> - [比Momentum更快：揭开Nesterov Accelerated Gradient的真面目](https://zhuanlan.zhihu.com/p/22810533)
> - [深度学习最全优化方法总结比较（SGD,Adagrad,Adadelta,Adam,Adamax,Nadam）](https://zhuanlan.zhihu.com/p/22252270)
> - [Deep Learning 最优化方法之AdaGrad](https://zhuanlan.zhihu.com/p/29920135)
