---
title: Deep Interest Network
tags:
  - 排序
  - Attention
mathjax: true
comments: false
copyright: true
date: 2020-01-21 21:13:51
categories: 推荐系统
---

### 1. 问题

在推荐系统领域中，通常的做法是将UserProfile、UserBehaviors、CandidateItem、ContextFeatures分别通过Embedding之后，从高维稀疏特征转化为低维稠密特征，
然后通过神经网络对这些特征进行学习，输出对CandidateItem的CTR。通常的推荐系统神经网络模型结构如下图所示：

<img src="/posts_res/2020-01-21-Deep-Interest-Network/02.png" title="Base Model" />

上述传统做法的一大缺点是将用户所有的行为记录UserBehaviors都平等地对待，对应的模型中就是 `average poolingn` 或者 `sum pooling`，将用户交互过的Item的embedding vector进行
简单的平均或者加和来表示这个用户历史行为的 vector，可能稍微加点trick的话，对不同时间的行为加上一个 `time decay` 系数，对兴趣进行时间衰减。

但是，通过我们的一系列行为中，有一部分是无效或者叫暂时兴趣，只是在当时的那个时刻存在之后就消失的兴趣，在上述传统 `average pooling` 中也将这部分兴趣点平等对待地考虑了进来，
这无疑对用户的兴趣点捕捉是存在问题的，因此这篇论文提出了将 `Attention`机制 应用于推荐系统领域中。主要解决的是：**让模型对不同时刻的交互行为学习不同的权重，而不是平等对待每一个历史交互行为**

<img src="/posts_res/2020-01-21-Deep-Interest-Network/01.png" />

如上图所示，用户历史的购买鞋子行为对推荐大衣的广告的影响很小。


### 2. Attention机制

Paper中提出了新的模型 `Deep Interest Network, DIN` 用来解决传统对用户行为一概而论的缺点。模型结构如下：

<img src="/posts_res/2020-01-21-Deep-Interest-Network/03.png" />

其中不同的地方主要集中在对用户历史行为的学习部分，因此这里不再赘述其他部分，单独只说对用户历史行为的学习部分(引入Attention的部分)。

1. Embedding层都是一样的，Item和其Categroy都是单独进行Embedding，然后进行Concat，这样每个 `ItemVector` 都是由其本身的ID和cate组合而成
2. CandidateItem，即待评估ctr的Item同其他Item的Embedding操作是一样的，共用相同的Embedding Matrix，生成 `CandidateVector`
3. 将 `ItemVector` 和 `CandidateVector` 一起传入到一个 **Activate Unit** 中，学习历史交互Item和CandidateItem之间的关联性weight(归一化到[0, 1])，即Item这个历史行为对推荐CandiateItem的影响权重有多少
4. 这里的 **Activate Unit** 其实就是 Attention机制，里面具体的操作在后续 **5.代码逻辑** 中再详细描述
5. 将学习到的Item 和 CandidateItem之间的weight同 `ItemVector`进行 element-wise 乘
6. 所有的历史交互Item进行 `sum pooling`，得到一个同 `ItemVector` 相同纬度的 `BehaviorsVector` 来表示用户历史兴趣对待推荐 CandidateItem 的分布
7. 将 `UserProfilesFeatures`、`BehaviorsVector`、`CandidateVector`、`ContextFeatures` 一起再输入到神经网路中学习一些特征交叉、特征组合，最终输出预测的ctr

论文中公式表述Attention的公式(3) 如下：

$$
v_U(A) = f(v_A, e_1, e_2, .., e_H) = \sum_{j=1}^H a(e_j, v_A) \cdot e_j = \sum_{j=1}^H w_j e_j
$$

其中: $\{ e\_1, e\_2, ..., e\_H \}$ 是长度为 `H` 的用户 `U` 历史行为embedding vector列表；$v_A$是`Item A`的embedding vector，$v_U(A)$表示在候选Item是 `A` 的时候，对用户历史学习到的用户表示向量。

### 3. Metrics

$$
AUC = \frac{\sum_{i=1}^n \sharp impression_i \times AUC_i}{\sum_{i=1}^n \sharp impression_i}
$$

其中 $n$ 是用户数量，$ \sharp impression\_i$ 和 $AUC\_i$ 是第 $i$ 个用户的impression和$AUC$

### 4. 激活函数

无论是PReLU还是ReLU的控制函数都是一个阶跃函数，其变化点在 $0$ 处，意味着面对不同的输入这个变化点是不变的。
实际上神经元的输出分布是不同的，面对不同的数据分布采用同样的策略可能是不合理的，因此提出的 **Dice** 中改进了这个控制函数，
让它根据数据的分布来调整，这里我们选择了统计神经元输出的均值和方差来描述数据的分布：

$$
f(s) = p(s) \cdot s + (1 - p(s)) \cdot \alpha s, \quad p(s) = \frac{1}{1 + exp( - \frac{s - E[s]}{ \sqrt( Var[s] + \epsilon ) } )}
$$

其中：训练阶段$E\[s\]$ 和 $Var\[s\]$ 是每个mini-batch的均值和方差；*测试阶段 $E\[s\]$ 和 $Var\[s\]$ is calculated by moving averages E[s] and Var[s] over data*. $\epsilon$是一个很小的常数，论文中设置为 $10^{-8}$。

Ps：测试阶段这一句没理解

### 5. 代码逻辑

先说格式：

- train set format: list((用户ID, 历史购买序列, 再次购买的商品, label))
- test set format: list((用户ID, 历史购买序列, (购买商品, 没购买商品)))

其中的各种ID都已经映射到了index，历史购买序列是一个index序列，label非0即1

Attention机制部分：

```python

def attention(queries, keys, keys_length):
    '''
    queries:     [B, H]         (batchSize, 32)                     待测商品和其类别的emb concat
    keys:        [B, T, H]      (batchSize, sequenceLength, 32)     购买序列的商品和其类别的emb concat
    keys_length: [B]            (batchSize, sequenceLength)         batch中每个样本的购买序列长度
    '''
    # 32
    queries_hidden_units = queries.get_shape().as_list()[-1]
    # 把query复制sequenceLength次
    queries = tf.tile(queries, [1, tf.shape(keys)[1]])
    # 转化为和keys相同的shape
    queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
    # Activation Unit 增加了  `queries - keys` 的部分
    din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # B*T*4H
    # 三层全链接，学习两个 商品 之间的 Activation Weight
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')  # B*T*1
    # outputs 的结果等价于 Q * K，(batchSize, 1, sequenceLength)
    outputs = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])  # B*1*T 

    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])
    # (batchSize, 1, sequenceLength)
    key_masks = tf.expand_dims(key_masks, 1)  # B*1*T 
    # # 在补足的地方附上一个很小的值而不是0, 为了softmax之后的值接近为0
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  
    # 将每个样本序列中空缺的商品都赋值为(-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # B * 1 * T 

    # Scale, 做 Q*K / sqrt(dim)
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation, 归一化到 [0, 1]
    # 这里的output是attention计算出来的权重，即论文公式(3)里的w
    outputs = tf.nn.softmax(outputs)  # B * 1 * T 

    # Weighted Sum, attention结束 softmax{ Q * K / sqrt(dim) } * V
    # 三维矩阵相乘，相乘发生在后两维，即 B * (( 1 * T ) * ( T * H ))
    outputs = tf.matmul(outputs, keys)  # B * 1 * H 
    # 返回attention之后的history sequence length的值,
    # 即论文中 Goods x Goods_Weight 并SUM Pooling后的结果
    return outputs
```


### Ref

> 1. [Paper: Deep Interest Network for Click-Through Rate Prediction](/posts_res/2020-01-21-Deep-Interest-Network/din.pdf)
> 2. [Code: zhougr1993/DeepInterestNetwork](https://github.com/zhougr1993/DeepInterestNetwork)
> 3. [推荐系统中的注意力机制——阿里深度兴趣网络（DIN）](https://www.jianshu.com/p/758c05864e7b)
> 4. [实战DIN](https://github.com/princewen/tensorflow_practice/tree/master/recommendation/Basic-DIN-Demo)

