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
w^{(t+1)} = w^{(t)} - \eta^{(t)} \cdot G^{(t)} - \eta^{(t)} \cdot \lambda \cdot sgn( w^{(t)} )
\tag{3.1-1}
$$

其中,
- $\lambda \in \mathbb{R} $是一个标量,且$\lambda \ge 0$,为L1正则化参数；
- $sgn(v)$为符号函数；
- $\eta^{(t)}$为学习率,通常将其设置为$1/\sqrt{t}$的函数；
- $G^{(t)} = \nabla\_w f(w^{(t)}, z^{(t)}) $代表了第$t$次迭代中损失函数的梯度；

#### 3.1.2 简单截断法

以$k$为窗口,当$t/k$不为整数时采用标准SGD进行迭代；当$t/k$为整数时,采用如下权重更新方式；

$$
w^{(t+1)} = T_0 \left( w^{(t)} - \eta^{(t)} G^{(t)}, \theta \right)
$$

$$
T_0 (v_i, \theta) = 
\begin{cases}
0 & \quad \mid v_i \mid \le \theta \\
v_i & \quad otherwise
\end{cases}
\tag{3.1-2}
$$

$\theta \in \mathbb{R}$是一个标量,且$\theta \ge 0$；

#### 3.1.3 截断梯度法(TG)

简单截断法太过激进,因此TG在此基础上进行了改进：

$$
w^{(t+1)} = T_1 \left( w^{(t)} - \eta^{(t)} G^{(t)}, \quad \eta^{(t)} \lambda^{(t)}, \quad \theta \right)
$$

$$
T_1 (v_i, \alpha, \theta) = 
\begin{cases}
max(0, \quad v_i - \alpha ) & \quad v_i \in [0, \theta] \\
max(0, \quad v_i + \alpha ) & \quad v_i \in [- \theta, 0 ] \\
v_i & \quad otherwise
\end{cases}
\tag{3.1-3}
$$

- 其中$\lambda^{(t)} \in \mathbb{R}$,且$\lambda^{(t)} \ge 0$；
- TG 同样是以$k$为窗口,每$k$步进行一次截断。
  - 当$t/k$不为整数时,$\lambda^{(t)} = 0$；
  - 当$t/k$为整数时,$\lambda^{(t)} = k \cdot \lambda$。
- 从公式$(3.1-3)$可以看出,超参数$\lambda$和$\theta$决定了$w$的稀疏程度,这两个值越大,则稀疏性越强；尤其令$\lambda = \theta$时,只需要通过调节一个参数就能控制稀疏性。

根据公式$(3.1-3)$，我们很容易写出 TG 的算法逻辑:

![TG](/posts_res/2019-06-05-Optimization-Method/5.png)


#### 3.1.4 截断公式对比

![compare](/posts_res/2019-06-05-Optimization-Method/1.png)

其中左侧是简单截断法的截断公式，右侧是截断梯度的截断公式。公式$(3.1-3)$进行改写，描述特征权重每个维度的更新方式:

$$
w^{(t+1)}_i = 
\begin{cases}
Trnc \{ ( w^{(t)}_i - \eta^{(t)} g^{(t)}_i ), \lambda^{(t)}_{TG}, \theta \} & if \quad mod(t,k) = 0 \\
w^{(t)}_i - \eta^{(t)} g^{(t)}_i & otherwise
\end{cases}
$$

$$
\lambda^{(t)}_{TG} = \eta^{(t)} \lambda k
\tag{3.1-4}
$$

$$
Trnc( w, \lambda^{(t)}_{TG}, \theta ) = 
\begin{cases}
0 & if \quad \mid w \mid \le \lambda^{(t)}_{TG} \\
w - \lambda^{(t)}_{TG} sgn(w) & if \quad \lambda^{(t)}_{TG} \le \mid w \mid \le 0 \\
w & otherwise
\end{cases}
$$

如果令$\lambda^{(t)}\_{TG} = \theta$, 截断公式$Trnc(w, \lambda^{(t)}\_{TG}, \theta)$变成:

$$
Trnc(w, \theta, \theta) = 
\begin{cases}
0 & if \quad \mid w \mid \le 0 \\
w & otherwise
\end{cases}
$$

**此时$TG$退化成简单截断法**。

如果令$\theta = \infty$截断公式$Trnc(w, \lambda^{(t)}_{TG}, \theta)$变成:

$$
Trnc(w, \lambda^{(t)}_{TG}, \infty) = 
\begin{cases}
0 & if \quad \mid w \mid \le \lambda^{(t)}_{TG} \\
w & otherwise
\end{cases}
$$

如果再令$k=1$,那么特征权重维度更新公式变成:

$$
\begin{align*}
w^{(t+1)}_i &= Trnc\{ (w^{(t)}_i - \eta^{(t)} g^{(t)}_i), \eta^{(t)} \lambda, \infty \} \\
&= w^{(t)}_i - \eta^{(t)} g^{(t)}_i - \eta^{(t)} \cdot \lambda \cdot sgn(w^{(t)}_i) 
\end{align*}
$$

**此时 $TG$ 退化成 L1正则化法**。


### 3.2 FOBOS前向后向切分

#### 3.2.1 FOBOS算法原理

在 FOBOS 中, 将权重的更新分为两个步骤:

$$
w^{(t+0.5)} = w^{(t)} - \eta^{(t)} \cdot G^{(t)} \\
w^{(t+1)} = argmin_w \{ \frac{1}{2} \| w - w^{(t+0.5)} \|^2 + \eta^{(t+0.5)} \cdot \psi (w) \}
\tag{3.2-1}
$$

- 前一个步骤实际上是一个标准的梯度下降步骤;
- 后一个步骤可以理解为对梯度下降的结果进行微调;
  - 前一部分保证微调发生在梯度下降结果的附近
  - 后一部分则用于处理正则化，产生稀疏性

如果将公式 $(3.2-1)$ 中的两个步骤合二为一, 即将 $w^{(t+0.5)}$ 的计算带入到 $w^{(t+1)}$ 中, 有:

$$
w^{(t+1)} = argmin _w \{ \frac{1}{2} \| w - w^{(t)} + \eta^{(t)} G^{(t)} \|^2 + \eta^{(t+0.5)} \psi (w) \}
$$

令$F(w) = \frac{1}{2} || w - w^{(t)} + \eta^{(t)} G^{(t)} ||^2 + \eta^{(t+0.5)} \psi (w)$, 
如果$w^{(t+1)}$存在一个最优解, 那么就可以推断$0$向量一定属于$F(w)$的次梯度集合:
$$
0 \in \partial F(w) = w - w^{(t)} + \eta^{(t)} G^{(t)} + \eta^{(t+0.5)} \partial \psi (w)
$$

由于$w^{(t+1)} = argmin_w F(w)$, 那么有:
$$
0 = \{ w - w^{(t)} - \eta^{(t)} G^{(t)} + \eta^{(t+0.5)} \partial \psi (w) \} \mid_{w = w^{(t+1)}}
$$

上式实际上给出了 FOBOS 中权重更新的另一种形式:
$$
w^{(t+1)} = w^{(t)} - \eta^{(t)} G^{(t)} - \eta^{(t+0.5)} \partial \psi (w^{(t+1)})
$$

我们这里可以看到, $w^{(t+1)}$不仅仅与迭代前的状态$w^{(t)}$有关，而且与迭代后的$\psi (w^{(t+1)})$有关。


#### 3.2.2 L1-FOBOS

在 L1 正则化下，有$\psi (w) = \lambda || w ||\_1$, 
为了简化描述, 用向量$v= \[ v\_1, v\_2, \cdots, v\_N \] \in \mathbb{R}^N$ 来表示$w^{(t+0.5)}$, 
用标量$\hat{\lambda} \in \mathbb{R}$ 来表示 $\eta^{(t+0.5)} \lambda$, 并将公式$(3.2-1)$等号右边按维度展开:
$$
w^{(t+1)} = argmin_w \sum_{i=1}^N ( \frac{1}{2} (w_i - v_i)^2 + \hat{\lambda} \mid w_i \mid )
\tag{3.2-2}
$$

可以看到,在求和公式$\sum_{i=1}^N \( \frac{1}{2} (w_i - v_i)^2 + \hat{\lambda} \mid w_i \mid \)$ 中的每一项都是大于等于 $0$ 的,
所以公式$(3.2-2)$可以拆解成对特征权重$w$每一维度单独求解:
$$
w^{(t+1)}_i = argmin_{w_i} ( \frac{1}{2} (w_i - v_i)^2 + \hat{\lambda} \mid w_i \mid )
$$

![derived](/posts_res/2019-06-05-Optimization-Method/6.png)

![derived](/posts_res/2019-06-05-Optimization-Method/7.png)

因此, 综合上面的分析得到在 FOBOS 在 L1 正则化条件下，特征权重的各个维度更新的方式为:
$$
\begin{align*}
w^{(t+1)}_i &= sgn(v_i) max (0, \mid v_i \mid - \hat{\lambda}) \\
&= sgn(w_i^{(t)} - \eta^{(t)} g^{(t)}_i ) max \{ 0, \mid w^{(t)}_i - \eta_t \cdot g_i^{(t)} \mid - \eta^{(t+0.5)} \lambda \}
\end{align*}
\tag{3.2-3}
$$

其中, $g_i^{(t)}$ 为梯度 $G^{(t)}$ 在维度 $i$ 上的取值。

根据公式$(3.2-3)$，我们很容易就可以设计出 L1-FOBOS 的算法逻辑:

![fobos](/posts_res/2019-06-05-Optimization-Method/8.png)

#### 3.2.3 L1-FOBOS与TG的关系

对于 L1-FOBOS 特征权重的各个维度更新公式$(3.2-3)$，也可以写作如下形式:

$$
w^{(t+1)}_i = 
\begin{cases}
0 & \quad \mid w^{(t)}_i - \eta^{(t)} \cdot g^{(t)}_i \mid \le \eta^{(t+0.5)} \lambda \\
\left( w^{(t)}_i - \eta^{(t)} \cdot g^{(t)}_i \right) - \eta^{(t+0.5)} \cdot \lambda \cdot sgn(w^{(t)}_i - \eta^{(t)} g^{(t)}_i) & \quad otherwise
\end{cases}
$$

上式截断的含义是：当一条样本产生的梯度不足以令对应维度上的权重值发生足够大的变化 $( \eta^{(t+0.5)} \cdot \lambda )$, 
则认为在本次更新过程中该维度不够重要，应当令其权重为$0$。

同时上式与TG的特征权重更新公式$(3.1-4)$对比, 发现如果令$ \theta = \infty, k = 1, \lambda^{(t)}_{TG} = \eta^{(t+0.5} \lambda $,则L1-FOBOS与TG完全一致，因此可以认为L1-FOBOS是TG在特定条件下的特殊形式。

### 3.3 RDA正则对偶平均

简单截断、TG、FOBOS 都是建立在 SGD 的基础之上的，属于梯度下降类型的方法，这类型方法的优点就是**精度比较高**，
并且TG、FOBOS也都能在稀疏性上得到提升。但是有些其它类型的算法，例如RDA，是从另一个方面来求解 Online Optimization,
并且更有效地提升了特征权重的稀疏性。

#### 3.3.1 RDA算法原理

在RDA中，特征权重的更新策略为:
$$
w^{(t+1)} = argmin_w \lbrace \frac{1}{t} \sum_{r=1}^t < G^{(r)}, w > + \psi (w) + \frac{\beta^{(t)}}{t} h(w) \rbrace
\tag{3.3-1}
$$

其中，$< G^{(r)}, w >$表示梯度 $G^{(r)}$对 $w$ 的积分平均值(积分中值); $\psi (w)$ 为正则项；$h(w)$为一个辅助的严格凸函数；
$ \lbrace \beta^{(t)} | t \ge 1 \rbrace $是一个非负且非自减序列。本质上，公式$(3.3-1)$中包含了3个部分：
- 线性函数 $ \frac{1}{t} \sum_{r=1}^t < G^{(t)}, w > $, 包含了之前所有梯度(或次梯度)的平均值(dual average);
- 正则项 $\psi (w)$;
- 额外正则项 $\frac{\beta^{(t)}}{t} h(w)$, 这是个严格凸函数;

#### 3.3.2 L1-RDA

令$\psi (w) = \lambda || w ||\_1$，并且由于$h(w)$是一个关于$w$的严格凸函数，不妨令 $h(w) = \frac{1}{2} || w ||\_2^2 $,
此外，将非负非自减序列 $ \lbrace \beta^{(t)} | t \ge 1 \rbrace $ 定义为 $ \beta^{(t)} = \gamma \sqrt{t} $，将L1正则化代入公式$(3.3-1)$有:
$$
w^{(t+1)} = argmin_w \lbrace \frac{1}{t} \sum_{r=1}^t < G^{(r)}, w > + \lambda \| w \|_1 + \frac{\gamma}{2 \sqrt{t}} \| w \| _2^2 \rbrace
\tag{3.3-2}
$$

针对特征权重的各个维度将其拆解成 N 个独立的标量最小化问题:
$$
minimize_{w_i \in \mathbb{R}} \lbrace \bar{g}_i^{(t)} w_i + \lambda \mid w_i \mid + \frac{\gamma}{2\sqrt{t}} w_i^2 \rbrace
\tag{3.3-3}
$$

这里 $ \lambda > 0, \frac{\gamma}{\sqrt{t}} > 0, \bar{g}\_i^{(t)} = \frac{1}{t} \sum\_{r=1}^t g\_i^{(r)} $, 
公式$(3.3-3)$就是一个无约束的非平滑最优化问题。其中第2项 $\lambda | w_i |$ 在 $w_i$ 处不可导。

假设 $w\_i^{\ast}$ 是其最优解，并且定义 $\xi \in \partial | w\_i^{\ast} $ 为 $ | w\_i | $ 在 $w\_i^{\ast}$ 的次导数，那么有：

![derived](/posts_res/2019-06-05-Optimization-Method/9.png)

之后可以得到L1-RDA特征权重的各个维度更新的方式为：
$$
w_i^{(t+1)} = 
\begin{cases}
0 & \quad if \quad \mid \bar{g}_i^{(t)} \mid < \lambda \\
-\frac{\sqrt{t}}{\gamma} ( \bar{g}_i^{(t)} - \lambda \cdot sgn( \bar{g}_i^{(t)} ) ) & \quad otherwise
\end{cases}
\tag{3.3-6}
$$

这里发现，当某个维度上累积梯度平均值的绝对值 $ | g_i^{(t)} | $ 小于阈值𝜆的时候，该维度权重将被置 $0$，特征权重的稀疏性由此产生。

根据公式 $(3.3-6)$，可以设计出 L1-RDA 的算法逻辑:

![rda](/posts_res/2019-06-05-Optimization-Method/10.png)


#### 3.3.3 L1-RDA与L1-FOBOS的比较

在 $3.2.2$ 中我们看到了 L1-FOBOS 实际上是 TG 的一种特殊形式, 在 L1-FOBOS 中，进行 **截断** 的判定条件是
$ | w\_i^{(t)} - \eta^{(t)} g\_i^{(t)} | \le \lambda^{(t)}\_{TG} = \eta^{(t+0.5)} \lambda $ 。
通常会定义 $\eta$ 为与 $1$ 正相关的函数 $(\eta = \Theta (\frac{1}{\sqrt{t}}))$, 
因此 L1-FOBOS 的**截断阈值**为 $\Theta (\frac{1}{\sqrt{t}})) \lambda $, 随着 $t$ 的增加, 这个阈值会逐渐降低。

相比较而言，从$(3.3-6)$可以看出, L1-RDA 的**截断阈值**为 $\lambda$, 是一个常数，并不随着 $t$ 而变化, 
因此可以认为 L1-RDA 比 L1-FOBOS 在截断判定上更加激进, 这种性质使得 L1-RDA 更容易产生稀疏性;
此外，RDA 中判定对象是梯度的累加平均值 $\bar{g}\_i^{(t)}$, 不同于 TG 或 L1-FOBOS 中针对单次梯度计算的结果进行判定,
避免了由于某些维度由于训练不足导致截断的问题。并且通过调节 $\lambda$ 一个参数，很容易在精度和稀疏性上进行权衡


### 3.4 FTRL

FTRL综合了L1-FOBOS基于梯度下降方法具有较高的精度、L1-RDA能在损失一定精度的情况下产生更好的稀疏性。

#### 3.4.1 L1-FOBOS和L1-RDA在形式上的统一

L1-FOBOS在形式上，令 $ \eta^{(t+0.5)} = \eta^{(t)} = \Theta ( \frac{1}{\sqrt{t}} ) $ 是一个随 $t$变化的非增正序列, 
每次迭代都可以表示为：
$$
w^{(t+0.5)} = w^{(t)} - \eta^{(t)} G^{(t)}
$$

$$
w^{(t+1)} = argmin_w \lbrace \frac{1}{2} | w - w^{(t + 0.5)} |_2^2 + \eta^{(t)} \lambda | w |_1 \rbrace
$$

把这两个公式合并到一起，有:
$$
w^{(t+1)} = argmin_w \lbrace \frac{1}{2} | w - w^{(t)} + \eta^{(t)} G^{(t)} |_2^2 + \eta^{(t)} \lambda | w |_1 \rbrace
$$

通过这个公式很难直接求出 $w^{(t+1)}$ 的解析解，但是我们可以按维度将其分解为 N 个独立的最优化步骤:
$$
\begin{align*}
最优化 &= minimize_{w_i \in \mathbb{R}} \lbrace \frac{1}{2} (w_i - w_i^{(t)} + \eta^{(t)} g_i^{(t)} )^2 + \eta^{(t)} \lambda \mid w_i \mid \rbrace \\
&= minimize_{w_i \in \mathbb{R}} \lbrace \frac{1}{2} (w_i - w_i^{(t)})^2 + \frac{1}{2} ( \eta^{(t)} g_i^{(t)} )^2 + w_i \eta^{(t)} g_i^{(t)} + w_i^{(t)} \eta^{(t)} g_i^{(t)} + \eta^{(t)} \lambda \mid w_i \mid \rbrace \\
&= minimize_{w_i \in \mathbb{R}} \lbrace w_i g_i^{(t)} + \lambda \mid w_i \mid + \frac{1}{2} \eta^{(t)} (w_i - w_i^{(t)})^2 + \lbrack \frac{ \eta^{(t)} }{2} (g_i^{(t)})^2 + w_i^{(t)} g_i^{(t)} \rbrack  \rbrace
\end{align*}
$$

由于 $\frac{ \eta^{(t)} }{2} (g\_i^{(t)})^2 + w\_i^{(t)} g\_i^{(t)}$ 与变量 $ w_i $ 无关，因此上式可以等价于:
$$
minimize_{w_i \in \mathbb{R}} \lbrace w_i g_i^{(t)} + \lambda \mid w_i \mid \frac{1}{2 \eta^{(t)} (w_i - w_i^{(t)})^2 }  \rbrace
$$

再将这 N 个独立最优化子步骤合并，那么 L1-FOBOS 可以写作:
$$
w^{(t+1)} = argmin_w \lbrace G^{(t)} \cdot w + \lambda | w |_1 + \frac{1}{2 \eta^{(t)}} | w - w^{(t)} |_2^2 \rbrace
$$

而对于 L1-RDA 的公式$(3.3.2-1)$，我们可以写作:
$$
w^{(t+1)} = argmin_w \lbrace G^{(1:t)} \cdot w + t \lambda |w|_1 +  \frac{1}{2 \eta^{(t)}} | w - 0 |_2^2 \rbrace
$$

这里 $G^{(1:t)} = \sum_{s=1}^t G^{(s)}$; 如果令 $ \sigma^{(s)} = \frac{1}{\eta^{(s)}} - \frac{1}{\eta^{(s-1)}}, \sigma^{(1:t)} = \frac{1}{\eta^{(t)}} $, 上面两个式子可以写做:
$$
w^{(t+1)} = argmin_w \lbrace G^{(t)} \cdot w + \lambda | w |_1 + \frac{1}{2} sigma^{(1:t)} | w- w^{(t)} | _2^2  \rbrace
\tag{3.4.1-1}
$$

$$
w^{(t+1)} = argmin_w \lbrace G^{(t)} \cdot w + t \lambda | w |_1 + \frac{1}{2} sigma^{(1:t)} | w- 0 | _2^2  \rbrace
\tag{3.4.1-2}
$$

比较$(3.4.1-1)$和$(3.4.1-2)$这两个公式，可以看出 L1-FOBOS 和 L1-RDA 的区别在于:
- (1) 前者对计算的是累加梯度以及 L1 正则项只考虑当前模的贡献，而后者采用了累加的处理方式;
- (2) 前者的第三项限制$ w $的变化不能离已迭代过的解太远，而后者则限制 $w$ 不能离 0 点太远;

#### 3.4.2 FTRL算法原理

FTRL 综合考虑了 FOBOS 和 RDA 对于正则项和$w$限制的区别，其特征权重的更新公式为:
$$
w^{(t+1)} = argmin_w \lbrace G^{(1:t)} \cdot w + \lambda_1 |w|_1 + \lambda_2 |w|_2^2 + \frac{1}{2} \sum_{s=1}^t \sigma^{(s)} | w - w^{(s)} |_2^2 \rbrace
\tag{3.4.2-1}
$$

注意，公式 $(3.4.2-1)$ 中出现了L2正则项 $ \frac{1}{2} || w ||_2^2 $，
L2正则项的引入仅仅相当于对最优化过程多了一个约束，使得结果求解结果更加“平滑”。

公式$(3.4.2-1)$看上去很复杂，更新特征权重貌似非常困难的样子。不妨将其进行改写，将最后一项展开，等价于求下面这样一个最优化问题:
$$
w^{(t+1)} = argmin_w \lbrace (G^{(1:t)} - \sum_{s=1}^t \sigma^{(s)} w^{(s)} ) \cdot w + \lambda_1 |w|_1 + \frac{1}{2} ( \lambda_2 + \sum_{s=1}^t \sigma^{(s)} ) \cdot | w |_2^2 + \frac{1}{2} \sum_{s=1}^t \sigma^{(s)} |w^{(s)}|_2^2 \rbrace
$$

由于 $\frac{1}{2} \sum\_{s=1}^t \sigma^{(s)} || w^{(s)} ||\_2^2$ 相对于 $w$ 来说是一个常数，并且令 $ z^{(t)} = G^{(1:t)} - \sum\_{s=1}^t \sigma^{(s)} w^{(s)} $, 上式等价于:
$$
w^{(t+1)} = argmin_w \lbrace z^{(t)} \cdot w + \lambda_1 |w|_1 + \frac{1}{2} (\lambda_2 + \sum_{s=1}^t \sigma^{(s)} )  |w|_2^2 \rbrace
$$

针对特征权重的各个维度将其拆解成N个独立的标量最小化问题:
$$
minimize_{w_i \in \mathbb{R}} \lbrace z_i^{(t)} w_i + \lambda_1 \mid w \mid_1 + \frac{1}{2} ( \lambda_2 + \sum_{s=1}^t \sigma^{(s)} ) w_i^2 \rbrace
$$

到这里，我们遇到了与式$(3.3.2-2)$类似的优化问题，用相同的分析方法可以得到:
$$
w_i^{t+1} = 
\begin{cases}
0 & \quad if \mid z_i^{(t)} \mid < \lambda \\
-( \lambda_2 + \sum_{s=1}^t \sigma^{(s)} )^{-1} ( z_i^{(t)} - \lambda_1 sng( z_i^{(t)} ) ) & \quad otherwise
\end{cases}
\tag{3.4.2-2}
$$

从式 $(3.4.2-2)$ 可以看出，引入 L2 正则化并没有对 FTRL 结果的稀疏性产生任何影响。


#### 3.4.3 Per-Coordinate Learning Rates

前面介绍了 FTRL 的基本推导，但是这里还有一个问题是一直没有被讨论到的:关于学习率 $\eta^{(t)}$ 的选择和计算。
事实上在 FTRL 中，每个维度上的学习率都是单独考虑的 (Per-Coordinate Learning Rates)。

在一个标准的OGD里面使用的是一个全局的学习率策略$\eta^{(t)} = \frac{1}{\sqrt{t}}$，这个策略保证了学习率是一个正的非增长序列，
对于每一个特征维度都是一样的。考虑特征维度的变化率: 如果特征1 比特征2 的变化更快，那么在维度1 上的学习率应该下降得更快。
我们很容易就可以想到可以用某个维度上梯度分量来反映这种变化率。在 FTRL 中，维度 $i$ 上的学习率是这样计算的:
$$
\eta^{(t)}_i = \frac{\alpha}{ \beta + \sqrt{ \sum_{s=1}^t (g_i^{(s)})^2 } }
\tag{3.4.3-1}
$$

由于 $\sigma^{(1:t)} = \frac{1}{\eta^{(t)}}$，所以公式$(3.4.2-2)$中 $\sum\_{s=1}^t \sigma^{(s)} = \frac{1}{\eta^{(t)}} = (\beta + \sqrt{ \sum\_{s=1}^t (g\_i^{(s)})^2 }) / \alpha$。
这里的 $\alpha$ 和 $\beta$ 是需要输入的参数, 公式 $(3.4.2-2)$ 中学习率写成累加的形式，是为了方便理解后面 FTRL 的迭代计算逻辑。


#### 3.4.4 FTRL算法逻辑

到现在为止，我们已经得到了 FTRL 的特征权重维度的更新方法$公式(3.4.2-2)$，每个特征维度的学习率计算方法公式$(3.4.3-1)$，
那么很容易写出 FTRL 的算法逻辑, 这里是根据$(3.4.2-2)$ 和$(3.4.3-1)$ 写的 L1&L2-FTRL 求解最优化的算法逻辑，如下：

![fengyang](/posts_res/2019-06-05-Optimization-Method/2.png)

而论文`Ad Click Prediction: a View from the Trenches`中 Algorithm 1 给出的是 L1&L2-FTRL 针对 Logistic Regression 的算法逻辑:

![Ad Click Prediction](/posts_res/2019-06-05-Optimization-Method/3.png)


### 3.5 Online总结

从类型上来看，简单截断法、TG、FOBOS 属于同一类，都是梯度下降类的算法,并且TG在特定条件可以转换成简单截断法和FOBOS;
RDA属于简单对偶平均的扩展应用;FTRL 可以视作 RDA 和 FOBOS 的结合，同时具备二者的优点。
目前来看， RDA 和 FTRL 是最好的稀疏模型 Online Training 的算法。

谈到高维高数据量的最优化求解，不可避免的要涉及到并行计算的问题, [冯扬(8119)的博客](http://blog.sina.com.cn/s/blog_6cb8e53d0101oetv.html)讨论了 batch 模式下的并行逻辑回归，其实只要修改损失函数，就可以用于其它问题的最优化求解。
另外，对于 Online 下，[Parallelized Stochastic Gradient Descent](http://martin.zinkevich.org/publications/nips2010.pdf)给出了一种很直观的方法:

![Parallelized Stochastic Gradient Descent](/posts_res/2019-06-05-Optimization-Method/4.png)

对于 Online 模式的并行化计算，一方面可以参考 ParallelSGD 的思路，另一方面也可以借鉴 batch 模式下对高维向量点乘以及梯度分量并行计算的思路。
总之，在理解算法原理的 基础上将计算步骤进行拆解，使得各节点能独自无关地完成计算最后汇总结果即可。


------------

> - [一个框架看懂优化算法之异同 SGD/AdaGrad/Adam](https://zhuanlan.zhihu.com/p/32230623)
> - [SGD、Momentum、RMSprop、Adam区别与联系](https://zhuanlan.zhihu.com/p/32488889)
> - [An overview of gradient descent optimization algorithms](https://arxiv.org/pdf/1609.04747.pdf)
> - [在线最优化求解(Online Optimization)-冯扬](/posts_res/2019-06-05-Optimization-Method/online_optimization_fengyang.pdf)
> - [比Momentum更快：揭开Nesterov Accelerated Gradient的真面目](https://zhuanlan.zhihu.com/p/22810533)
> - [深度学习最全优化方法总结比较（SGD,Adagrad,Adadelta,Adam,Adamax,Nadam）](https://zhuanlan.zhihu.com/p/22252270)
> - [Deep Learning 最优化方法之AdaGrad](https://zhuanlan.zhihu.com/p/29920135)
> - [Ad_Click_Prediction_a_View_from_the_Trenches](/posts_res/2019-06-05-Optimization-Method/Ad Click Prediction- a View from the Trenches.pdf)
> - [各大公司广泛使用的在线学习算法FTRL详解](https://www.cnblogs.com/EE-NovRain/p/3810737.html)

