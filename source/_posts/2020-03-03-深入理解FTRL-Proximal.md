---
title: 深入理解FTRL-Proximal
tags:
  - 优化算法
  - 在线学习
mathjax: true
comments: false
copyright: true
date: 2020-03-03 16:39:00
categories: 机器学习
---


### 1. 前言

写这片post的目的是因为最近在实践中需要对不同的优化器进行调整，
突然发现自己对各种优化算法的了解都停留在一个相对比较浅的层面上，
虽然之前自己从各个渠道汇总整理了一份优化算法的总结性post——[优化算法整理post](/2019/06/05/Optimization-Method/),
但是都是浮于表面的总结，并没有沉淀下来成为自己的东西，
因此这篇post主要记录一下自己在看 [Ad Click Prediction a View from the Trenches](https://arxiv.org/pdf/1609.04747.pdf) 这篇论文中得到和不理解的地方。


### 2. FTRL的背景

点击率预估问题(CTR)是推荐系统(计算广告)中非常重要的模块，预估一个用户对广告/Item的点击概率，从而提升广告/Item效果。
LR模型是CTR问题中最经典的模型，而LR模型训练也是无约束最优化问题中经典案例。因此问题最终归约为如何有效优化LR模型；
同时还需要考虑模型稀疏性，模型稀疏性可能会和训练数据不足、模型复杂度过高相关，也是控制过拟合的有效手段。
LR模型需要解决问题是：

样本为正例的概率表示为:
$$
\begin{equation*}
\begin{aligned}
p_t &= \sigma(w_t \cdot x_t) \\
\sigma(a) &= \frac{1}{1 + exp(-a)}
\end{aligned}
\end{equation*}
$$

对应的log损失函数为:
$$
l(w_t) = -y_t log p_t - (1 - y_t) log(1 - p_t)
$$

批量GD算法能够保证精度，同时可以加上正则项，L1或者L2正则预防过拟合问题。
在线GD算法需要能够利用实时产生的正负样本，一定程度上能够优化模型效果。
在线优化算法需要特殊关注模型**鲁棒性**和**稀疏性**，由于样本是一个一个到来，即使加了L1或者L2正则也无法保证取得稀疏的模型。
因此在线GD会着重处理模型稀疏性，FTRL就是综合了RDA和FOBOS优点，既有RDA的精度，又有FOBOS的稀疏性。


### 3. FTRL的主要内容

#### 3.1 FTRL更新策略

$$
w_{t+1}=argmin(g_{1:t}w+\frac12\sum_{s=1}^t\sigma_s||w-w_s||^2+\lambda_1||w||+\frac12 \lambda_2||w||^2)
$$

其中 $ g\_{1:t}=\sum\_{s=1}^t g\_t $ ,相当于新产生的权重验证所有样本的梯度并且和历史权重不偏离太远。最后通过L1正则进行稀疏性约束。这样既保证了权重更新的精度有保证了稀疏性。
另外参数 $\sigma\_s$ 是一个和学习率相关参数 $\sum\_{s=1}^t \sigma\_s=\frac{1}{\eta\_t}$ 而 $\eta\_t=\frac{1}{\sqrt{t}}$ 是一个非增序列


#### 3.2 公式推导

$$
\begin{equation*}
\begin{aligned}
F(w)
&=g_{1:t}w+\frac12\sum_{s=1}^t\sigma_s||w-w_s||^2+\lambda_1||w||+\frac12 \lambda_2||w||^2 \\
&=g_{1:t}w+\frac12\sum_{s=1}^t\sigma_s(w^Tw-2w^Tw_s+w_s^Tw_s)+\lambda_1||w||+\frac12 \lambda_2||w||^2 \\
&=(g_{1:t}-\sum_{s=1}^t\sigma_s w_s) w+\frac12(\sum_{s=1}^t\sigma_s+\lambda_2)w^Tw+\lambda_1||w||+const \\
&=z_t^T w+\frac12(\frac 1 \eta_t+\lambda_2)w^Tw+\lambda_1||w||+const
\end{aligned}
\end{equation*}
$$

其中 $z\_{t-1}=g\_{1:t-1}-\sum\_{s=1}^{t-1}\sigma\_s w\_s$，根据定义可以得到 $z\_{t-1}=z\_{t-1}+g\_t-(\frac{1}{\eta\_t}-\frac{1}{\eta\_{t-1}})w\_t$

对上式进行求导可以得到

$$
z_t+(\frac 1 \eta_t+\lambda_2)w+\lambda_1 \partial|W|=0
$$

从上式中可以容易得到$w$和$z$必须是异号，否则等式不能成立。  


> note: 考虑凸函数 $f(x)=|x|$。在原点的次微分是区间\[−1, 1\]。$x<0$时，次导数是单元素集合 $\{-1\}$，而$x>0$，则是单元素集合 $\lbrace 1\rbrace$。

$$
\partial |W| =
\begin{cases}
0,  & \text{if $-1< w < 1$ } \\
1, & \text{if $ w > 1$ } \\
-1 , & \text{if $ w < -1$}
\end{cases}
$$

因此根据L1正则的次导数和梯度公式，讨论如下（论文推导中没有带 $\lambda\_2$正则系数，Algorithm1则带了）。

> 1. 当 $|z\_{t,i}| < \lambda\_1$ 时 <br> 
$w\_i > 0$，则有 $w = \frac{-z\_i - \lambda\_1}{\frac{1}{\eta_t} + \lambda\_2} < 0$ 不成立 <br>
$w\_i < 0$，则有 $w = \frac{-z\_i + \lambda\_1}{\frac{1}{\eta_t} + \lambda\_2} > 0$ 不成立 <br>
$w_i=0$
2. 当$z\_t > \lambda\_1$时 <br>
由于两者必须异号，此时有$w_i < 0$ <br> 
$w = \frac{-z\_i + \lambda\_1}{\frac{1}{\eta_t} + \lambda\_2}$
3. 当$z\_t < − \lambda\_1$时，同理 <br>
$w = \frac{-z\_i + \lambda\_1}{\frac{1}{\eta_t} + \lambda\_2}$
4. 因此有 <br>
$$
w_i =
\begin{cases}
0,  & \text{if $|z_i| \le \lambda_1$ } \\
\frac{-(z_i-sgn(z_i)\lambda_1)}{\frac 1 \eta_t+\lambda_2}, & \text{if $others$ } 
\end{cases}
$$

#### 3.3 算法流程(论文中的Algorithm1)

<img src="/posts_res/2020-03-03-深入理解FTRL-Proximal/algorithm.png" width=60% height=60% />

其中 $n\_i$表示梯度的累加和，在论文3.1节中也说了可以对 $n\_i^{0.5}$ 中的 $power=0.5$ 进行不同的尝试。


### 4. 工程上的tricks

**下列的这些方法都可以在实际的生产环境中进行实验，包括word2vec的采样等操作，都是实际工作中值得学习，尝试的方法。**

#### 4.1 Per-Coordinate Learning Rates

即各个维度采用不同的学习率:
$$
\eta_{t_i}=\frac {\alpha}{\beta+\sqrt \sum_{s=1}^tg^2_{s_i}}
$$

对于不同的维度的特征的出现次数是不一样的，因此更新的步长(学习率)也应该是不一样的。
- 出现次数多的特征应该学习率变化的次数也要多一些，收缩的速度(次数)也要快一些，这个维度的学习率绝对值learning_rate也应该是相对较小的；
- 出现次数少的特征应该学习率变化的次数相对少一些，收缩的速度(次数)也要相对慢一些，这个维度的学习率绝对值learninng_rate也应该是相对较大的，保证快速的更新。

#### 4.2 Probabilistic Feature Inclusion

背景：在训练数据中存在着大量的稀疏特征，如何将一些用不到的稀疏特征过滤掉是这一小节讨论的问题。

方法：
1. L1正则的方式，将稀疏特征的维度系数更新为零；会导致不可接受的精度损失；
2. hash的方式，并没有什么收益；
3. 概率特征引入，可以通过之前的数据预先统计，然后应用在线上；
  - Poisson Inclusion: 假设特征是随机到达的，也就是说应该符合Poisson分布，在一个特征未被引入之前，以概率$p$进行添加，一旦添加后序和OGD的流程一致；
  - Bloom Filter Inclusion: 使用Bloom Filter对特征出现次数统计，超过一定次数$n$加入到训练中，但布隆过滤器并非精确的，而是false positive的，因为产生冲突时，可能会加入还没有满$n$次的特征来进行训练，实际中我们并不要求完全精确的筛选。**在实验中，Bloom Filter表现出更高的RAM节省及更少的AucLoss，更推荐。**

#### 4.3 Encoding Values with Fewer Bits

论文中指出几乎所有的有效系数都落在 $(-2, +2)$这个范围里，因此使用`float32`或者`float64`是完全没有必要的。
这里提出了新的编码方式`q2.13`。

`q2.13`中2位为整数部分，13位为小数部分，另外有1位符号位。

#### 4.4 Training Many Similar Models

在做特征筛选或者设置优化时，基于已有的训练数据，控制变量，生成多个不同的变体来进行对比实验是一个不错的方法，但这样做比较耗费资源。
之前有一个比较省资源的方法是基于一个先验模型，使用其他模型来预测残差(应该是类似boosting的思想)，但是这样不能进行特征移除和特征替换的效果。

这里Google提出了一个方法，由于每个模型变体的相似度是很高的，一部分特征是变体之间共享的，一部分特征是变体独有的。
如果采用一张hashtable来存储所有变体的模型参数，就可以摊销各个模型都进行独立存储key的内存消耗，同时也会降低看网络带宽、CPU、磁盘的消耗。

**A Single Value Structure**

这个是指不同模型变体之间共享一个特征参数，而不是每一个模型变体存储单独的一个，用一个位字段来标识这个特征被哪些变体使用。
更新的方式如下：
1. 每一个模型变体去进行prediction并计算loss；
2. 对于每一个特征给出一个迭代后的参数值；
3. 对所有的更新后的参数值进行平均并进行更新存储；
4. 后序这个更新的参数值会供所有的变体共享；

#### 4.5 Computing Learning Rates with Counts

学习率更新优化通过正负样本个数进行优化。

假设有 $P$ 个正样本，$N$ 个负样本，那么预测正样本的概率应该是 $p = P / (P + N)$，
假设使用 Logistic Regression模型的话，那么正样本的梯度就是 $p - 1$，负样本的梯度就是 $p$，因此梯度平方和就可以表示为：
$$
\begin{equation*}
\begin{aligned}
\sum g^2_{t,i} 
&= \sum_{positive\_events} (1 - p_t)^2 + \sum_{negative\_events} p_t^2 \\
&\approx P(1 - \frac{P}{N+P})^2 + N (\frac{P}{N+P})^2 \\
&= \frac{PN}{N+P}
\end{aligned}
\end{equation*}
$$

#### 4.6 Subsampling Training Data

对于CTR模型，负样本数量特别多，因此正样本更有价值，考虑采用负样本采样，采样方式如下：

1. 正样本全部保留
2. 负样本赋值一个分数 $r \in (0, 1]$

如果只按照上面的采样方式进行训练的话，会带来严重的偏差预测，因此需要对样本按照分数和label赋值一个重要性权重 $weight$；

$$
w_t = 
\begin{cases}
& 1 \quad \quad \text{event } t \text{ is in a clicked query} \\
& \frac{1}{r} \quad \quad \text{event } t \text{ is in a query with no clicks}
\end{cases}
$$

同时相当于梯度也会乘上该系数，不会改变最终的损失。

定义 $s\_t$ 表示event t被采样的概率($1$ or $r$)，同时 $s\_t = 1 / w\_t$，因此有
$$
El(w_t) = s_tw_tl(w_t)+(1-s_t)0 = s_t \frac{1}{s_t} l_t (w_t) = l(w_t)
$$

所以线性期望证明了 $weight$ 加权的目标函数等价于原始数据集的目标函数。


### 5. 模型效果评估

评测模型效果一般基于历史数据，综合很多指标来进行评估，比如AucLoss，LogLoss，SquaredError等。下面5.1 & 5.2是Google进行模型评估时的尝试：

#### 5.1 Progressive Validation

常规操作一般是从训练集里抽一部分出来做test，但Google认为都不如online test可以反馈在最近的query数据上的效果，可以使用100%的数据做测试和训练。
由于不同的国家，不同的时间点，不同的query下，指标的benchmark是不同的，而相对指标变化是稳定的，应该关注指标的相对值。同时也说明观察模型效果要对数据进行切片观察。

#### 5.2 Deep Understanding through Visualization

我们看到的模型整体提升是一个聚合结果，可能只发生在某一个slice上的效果提升了，其他slice并没有或者发生下降。
我们可以利用天生对图像的敏感进行不同维度切分下指标的可视化来表达模型的表现。
于是，Google开发了一个web可视化工具GridViz，可以可视化不同变体在不同slice上的指标变化。
通过不同的冷暖色，不同的透明程度来表达升高降低，以及对应的幅度信息。

#### 5.3 置信度评估

前面说了CTR预测是怎么发生的，但在实际场景中只预测click发生的概率是不够的，还要对这个预测进行一个置信度评估，
来达到explore/exploit E&E问题的一个tradeoff。一方面做出精准的预测，把比较好的广告符合用户兴趣的广告展示出来，
但是这类数据总归是少的；公司总是要赚钱的，而且应广告主的需求，也需要将一些展示次数比较少能获得较高收益的广告也展示出来。

传统标准的置信度评估方法不适合这个场景，原因在于：
1. 标准方法需要完全收敛的、没有正则批处理模型，数据要IID的。不符合；
2. 标准方法需要NxN的矩阵，在Google场景下N的量级是billion，无奈放不下；不符合；
3. 实际场景中，计算置信度的时间应该与做ctr判定在一个数量级上；

Google提出了一个`uncertainty score`来进行评估：

核心思想在于对每一个特征的频次进行计数，该频次决定这学习率，学习率越小则该特征更可信，与我们的直觉是相符的。

为了简单，考虑将 $\lambda\_1 = \lambda\_2 = 0$，所以FTRL-Proximal等价于在线梯度下降算法OGD，令 $n\_{t,i} = \beta + \sum\_{s=1}^t g\_{s,i}^2$，则得到：

$$
\begin{equation*}
\begin{aligned}
| x \cdot w_t - x \cdot w_{t+1} | 
&= \sum_{i:|x_i| > 0} \eta_{t,i} | g_{t,i} | \\
&\le \alpha \sum_{i:|x_i| > 0} \frac{x_{t,i}}{\sqrt{n_{t,i}}} \\
&= \alpha \boldsymbol{\eta} \cdot x \\
&\equiv u(x)
\end{aligned}
\end{equation*}
$$

因此定义 `uncertainty score` 为 $u(x) = \alpha \boldsymbol{\eta} \cdot x$ 的上界值，这个值可以通过点积快速计算。


### 6. 失败的尝试

1. 对特征使用哈希技巧   
  一些文献声称的feature hashing（用于节省内存）的方式在试验中无效。因此保存可解释（即non-hashed）的特征数值向量。
2. Dropout   
  没有增强泛化性能反而损害了精度。我们认为这是在深度学习领域(比如图像识别)中特征是稠密的，所以Dropout可以大放异彩，我们的数据本来就很稀疏，再Dropout就没了。
3. 特征bagging   
  不知道为什么反正就是结果并不好。
4. 特征标准化normalize   
  可能是由于学习速度和正规化的相互作用，特征归一化之后效果并不好


----------------------

> 1. [Ad Click Prediction a View from the Trenches](https://arxiv.org/pdf/1609.04747.pdf)
> 2. [【每周一文】Ad Click Prediction: a View from the Trenches(2013)](https://blog.csdn.net/fangqingan_java/article/details/51020653)
> 3. [CTR预测算法之FTRL-Proximal](https://zhuanlan.zhihu.com/p/25315553)

