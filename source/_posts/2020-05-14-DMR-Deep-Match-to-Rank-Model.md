---
title: '[DMR]Deep Match to Rank Model'
tags:
  - DIN
  - Youtube
  - CTR
mathjax: true
comments: false
copyright: true
date: 2020-05-14 20:13:44
categories: 推荐系统
---


### 1. 背景

推荐系统通常分为两个阶段，即召回和排序阶段。在召回阶段会对用户和物品进行匹配，得到较小的一部分候选集进入到排序阶段。
在召回阶段，协同过滤方法是最常用来计算用户和物品相关性的方法。
在排序阶段，排序模型会对候选集的每个物品进行打分，然后选取得分最高的N个物品推荐给用户。
而打分最为常用的方式是预测用户对物品的点击率。
因此，点击率预估也受到了学术界和工业界众多研究者的关注。而本文也重点关注点击率预估问题。


对于点击率预估问题，个性化是提升其效果的一个重要的方面。个性化的一个重要方面就是对用户兴趣的刻画，如之前提到过的DIN、DIEN、DSIN等模型。
但是这些模型忽略了建模用户和物品之间的相关性。用户和物品之间的相关性，可以直接衡量用户对目标商品的偏好强度。

表征user2item的相关性，主要有基于矩阵分解和基于深度学习的方法。
基于深度学习的方法，如比较熟悉的Youtube的DNN召回模型。


### 2. DMR模型

<img src="/posts_res/2020-05-14-DMR-Deep-Match-to-Rank-Model/1.png" />


#### 2.1 特征表示

输入可以分为四部分:
1. 用户特征(UserProfile): 用户ID、消费等级等，用 $x\_p$ 表示。
2. 用户行为特征(UserBehavior): 用户交互过的物品集合，每个物品包括其物品ID、品类ID，用 $x\_b = \[ e\_1, e\_2, \cdots, e\_T \]$ 表示；其中$e\_1$即用户交互过的第一个物品的向量，不同用户的行为长度通常不同，因此往往对所有物品对应的向量进行pooling操作。
3. 目标物品特征(TargetItem): 物品ID、品类ID，用 $x\_t$ 表示。
3. 上下文特征(Context): 时间、召回方式、对应的召回评分，用 $x\_c$ 表示。

大多数输入特征是离散特征，通过Embedding的方式转换成对应的嵌入向量。
其中用户行为和目标物品共用同一组Embedding来节省存储空间(后面会介绍有特别的一个Embedding矩阵)。
随后，将 $x = \[ x\_p, x\_b, x\_t, x\_c \]$ 输入到多层全连接网络中，
中间层使用 $pRelu$ 激活函数，最后一层使用 $sigmoid$ 激活函数，得到点击率预估值。

损失函数使用交叉熵损失：

$$
L_{target} = - \frac{1}{N} \sum_{(x,y) \in D} ylog(f(x)) + (1 - y) log(1 - f(x))
$$


#### 2.2 User-to-Item Netword

<img src="/posts_res/2020-05-14-DMR-Deep-Match-to-Rank-Model/2.png" />

`User-to-Item Netword`类似于YoutubeDNN的结构：即**基于用户历史行为得到用户向量表示，然后与目标物品对应的向量求内积，来表征相关性**。

目标物品的向量，即物品ID对应的Embedding，并且**单独**使用了另一组物品ID的Embedding。

*这里需要注意的是，除了User-to-Item Network中目标物品ID的Embedding外，其他输入部分的物品ID的Embedding都是同一组Embedding。后续实验也表明使用两组Embedding尽管增大了存储空间，但模型的表征能力更强，同时实验效果也好于仅仅把Embedding的长度扩大两倍。*

用户的向量表示`u`仍是其历史行为中物品向量的加权求和，但与之前不同的是，这里计算相关性并没有用到目标物品向量。
这里的加权个人认为更多取决于偏置向量，如时间因素等，那么有可能学到的事近期发生的行为具有更大的权重，时间较远的行为权重较小，
这和我们的直觉是一致的。

$$
\begin{align*}
a_t &= z^T tanh (W_p p_t + W_e e_t + b) \\
\alpha_t &= \frac{exp(a_t)}{\sum_{i=1}^T exp(a_i)} \\
u &= g(\sum_{t=1}^T (\alpha_t e_t)) = g( \sum_{t=1}^T (h_t) )
\end{align*}
$$

随后，用户和物品的相关性通过内积的方式得到：

$$
r = u^T \cdot v_{\cdot}'
$$

我们希望$r$越大，代表用户与目标物品的相关性更高，但是仅仅通过反向传播方式去训练并不容易实现这一目标，
同时目标物品的另一组Embedding(单独的那个Embedding矩阵)也学习的不充分。因此在上述的基础上增加了辅助网络(类似于DIEN)。

辅助网络的任务是基于用户的前 `T-1` 次行为，来预测用户第 `T` 次交互的物品。
基于用户的向量表示`u`，我们很容易得到用户前`T-1`次行为后的向量表示 $u\_{T-1}$，即拿前`T-1`次行为重复上述用户向量表示的计算过程即可。那么用户第`T`次交互的物品为`j`的概率为：

$$
p_j = \frac{exp(u^T_{T-1} v_j')}{\sum_{i=1}^K exp(u^T_{T-1} v_i')}
$$


可以看到，辅助网络是一个多分类网络，为了减少计算复杂性，进行了负采样操作，因此辅助网络的损失为：

$$
L_{NS} = - \frac{1}{N} \sum_{i=1}^N (log(\sigma(u^T_{T-1} v_o')) + \sum_{j=1}^k log(\sigma(-u^T_{T-1} v_j')))
$$

那么此时总的损失计算如下：

$$
L_{final} = L_{target} + \beta L_{NS}
$$


这里增加 `User-to-Item Network` 网络，个人感觉还带来了另一个好处，可以将该模型应用于召回阶段做召回，
然后保存预测值，这样就可以直接作为一维特征用于DMR模型中，减少了DMR模型计算的耗时。


#### 2.3 Item-to-Item Network

<img src="/posts_res/2020-05-14-DMR-Deep-Match-to-Rank-Model/3.png" />

`Item-to-Item Network`的思路基本跟DIN是相同的，即首先计算用户历史行为中每个物品跟目标物品之间的相关性得分，然后基于该相关性得分对历史行为物品向量进行加权平均。
相关性得分计算公式如下：

$$
\hat{a_t} = \hat{z}^T tanh (\hat{W}_c e_c + \hat{W}_p p_t + \hat{W}_e e_t + \hat{b})
$$

这里输入主要有三部分，分别是`历史行为中第t个物品对应的向量`$e\_t$、`第t个物品的偏置向量`$p\_t$，以及`目标物品向量`$e\_c$。


在计算相似性之后，有三部分输出到concat层：
1. 对历史行为中物品向量的加权求和：
$$
\begin{align*}
\hat{\alpha_t} = \frac{exp(\hat{a}_t) }{ \sum_{i=1}^T exp(\hat{a}_i)} \\
\hat{u} = \sum_{t=1}^T ( \hat{\alpha}_t \hat{e}_t )
\end{align*}
$$
2. 所有历史行为中物品与目标物品相关性评分的求和，注意这里是softmax之前的得分，从另一个角度刻画了用户与目标物品的相关性：
$$
\hat{r} = \sum_{t=1}^T \hat{a}_{t\cdot}
$$
3. 目标物品向量。


### 3. 实验结果

见论文。


---------------------------

> 1. [Deep Match to Rank Model for Personalized Click-Through Rate Prediction](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-LyuZ.5099.pdf)
2. [lvze92/DMR](https://github.com/lvze92/DMR)
