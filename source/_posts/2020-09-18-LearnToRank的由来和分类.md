---
title: LearnToRank的由来和分类
tags:
  - 排序
  - 推荐系统
mathjax: true
comments: false
copyright: true
date: 2020-09-18 19:42:03
categories: 算法常识
---

> [学习排序 Learning to Rank：从 pointwise 和 pairwise 到 listwise，经典模型与优缺点](https://blog.csdn.net/lipengcn/article/details/80373744)

Ranking 是信息检索领域的基本问题，也是搜索引擎背后的重要组成模块。本文将对结合机器学习的 ranking 技术 -- Learning2Rank 做个系统整理，包括 pointwise、pairwise、listwise 三大类型，它们的经典模型，解决了什么问题，仍存在什么缺陷。本文主要参考刘铁岩老师的《Learning to Rank for Information Retrieval》和李航老师的《Learning to rank for information retrieval and natural language processing》。

### 1. 概述

### 1.1 Ranking

Ranking 模型可以粗略分为**基于相关度**和**基于重要性**进行排序的两大类。
早期基于相关度的模型，通常利用 query 和 doc 之间的词共现特性（如布尔模型）、VSM（如 TFIDF、LSI 等）、概率排序思想（BM25、LMIR 等）等方式。
基于重要性的模型，利用的是 doc 本身的重要性，如 PageRank、TrustRank 等。
这里我们关注基于相关度的 ranking。

- 相关度的标注
  - 最流行也相对好实现的一样方式时，人工标注 MOS，即相关度等级。
  - 其次是，人工标注 pairwise preference，即一个 doc 是否相对另一个 doc 与该 query 更相关。
  - 最 costly 的方式是，人工标注 docs 与 query 的整体相关度排序。

- 评估指标

即评估 query 与 docs 之间的真实排序与预测排序的差异。
大部分评估指标都是针对每组 query-docs 进行定义，然后再在所有组上进行平均。

常用的基于度量的 ranking 错误率如下

- MAP
- NDCG

[各种评价指标](/2018/09/29/machinelearningevaluate)有各种指标详细说明


可以发现，这些评估指标具备两大特性：

- 基于 query ，即不管一个 query 对应的 docs 排序有多糟糕，也不会严重影响整体的评价过程，因为每组 query-docs 对平均指标都是相同的贡献。
- 基于 position ，即显式的利用了排序列表中的位置信息，这个特性的副作用就是上述指标是离散不可微的。

一方面，这些指标离散不可微，从而没法应用到某些学习算法模型上；另一方面，这些评估指标较为权威，通常用来评估基于各类方式训练出来的 ranking 模型。因此，即使某些模型提出新颖的损失函数构造方式，也要受这些指标启发，符合上述两个特性才可以。这些细节在后面会慢慢体会到。

#### 1.2 Learning to Rank

Learning2Rank 即将 ML 技术应用到 ranking 问题，训练 ranking 模型。通常这里应用的是判别式监督 ML 算法。经典L2R框架如下：

![wise](/posts_res/2020-09-18-LearnToRank的由来和分类/2.png)

- 特征向量 x 反映的是某 query 及其对应的某 doc 之间的相关性，通常前面提到的传统 ranking 相关度模型都可以用来作为一个维度使用。
- L2R 中使用的监督机器学习方法主要是判别式类。


根据上图基本元素（输入空间、假设空间、输出空间、损失函数）方面的差异，L2R可以分为三大类，pointwise 类，pairwise 类，listwise 类。
总结如下:

![wise](/posts_res/2020-09-18-LearnToRank的由来和分类/1.png)


### 2 Pointwise Approach

#### 2.1 特点

Pointwise 类方法，其 L2R 框架具有以下特征：

- 输入空间中样本是单个 doc（和对应 query）构成的特征向量；
- 输出空间中样本是单个 doc（和对应 query）的相关度；
- 假设空间中样本是打分函数；
- 损失函数评估单个 doc 的预测得分和真实得分之间差异。

这里讨论下，关于人工标注标签怎么转换到 pointwise 类方法的输出空间：

- 如果标注直接是相关度 $s\_j$，则 $doc\_{x\_j}$ 的真实标签定义为 $y\_j = s\_j$
- 如果标注是 pairwise preference $s\_{u,v}$，则 $doc\_{x\_j}$ 的真实标签可以利用该 doc 击败了其他 docs 的频次
- 如果标注是整体排序 $\pi$，则 $doc\_{x\_j}$ 的真实标签可以利用映射函数，如将 doc 的排序位置序号当作真实标签

根据使用的 ML 方法不同，pointwise 类可以进一步分成三类：基于回归的算法、基于分类的算法，基于有序回归的算法。

#### 2.2 基于回归的算法

此时，输出空间包含的是实值相关度得分。

采用 ML 中传统的回归方法即可。

#### 2.3 基于分类的算法
此时，输出空间包含的是无序类别。

对于二分类，SVM、LR 等均可；对于多分类，提升树等均可。

#### 2.4 基于有序回归的算法

此时，输出空间包含的是有序类别。

通常是找到一个打分函数，然后用一系列阈值对得分进行分割，得到有序类别。采用 PRanking、基于 margin 的方法都可以。

#### 2.5 缺陷

回顾概述中提到的评估指标应该基于 query 和 position，

- ranking 追求的是排序结果，并不要求精确打分，只要有相对打分即可。
- pointwise 类方法并没有考虑同一个 query 对应的 docs 间的内部依赖性。一方面，导致输入空间内的样本不是 IID 的，违反了 ML 的基本假设，另一方面，没有充分利用这种样本间的结构性。其次，当不同 query 对应不同数量的 docs 时，整体 loss 将会被对应 docs 数量大的 query 组所支配，前面说过应该每组 query 都是等价的。
- 损失函数也没有 model 到预测排序中的位置信息。因此，损失函数可能无意的过多强调那些不重要的 docs，即那些排序在后面对用户体验影响小的 doc。

#### 2.6 改进

Pointwise 类算法也可以再改进，比如在 loss 中引入基于 query 的正则化因子的 RankCosine 方法。


### 3. Pairwise Approach

#### 3.1 特点

Pairwise 类方法，其 L2R 框架具有以下特征：

- 输入空间中样本是（同一 query 对应的）两个 doc（和对应 query）构成的两个特征向量；
- 输出空间中样本是 pairwise preference；
- 假设空间中样本是二变量函数；
- 损失函数评估 doc pair 的预测 preference 和真实 preference 之间差异。

这里讨论下，关于人工标注标签怎么转换到 pairwise 类方法的输出空间：

- 如果标注直接是相关度 $s\_j$，则 $doc\_{pair(x\_u,x\_v)}$ 的真实标签定义为 $y\_{u,v} = 2 * I\_{s\_u > s\_v} - 1$
- 如果标注是 pairwise preference $s\_{u,v}$，则 $doc\_{pair(x\_u,x\_v)}$ 的真实标签定义为 $y\_{u,v} = s\_{u,v}$
- 如果标注是整体排序 $\pi$，则 $doc\_{pair(x\_u,x\_v)}$ 的真实标签定义为 $y\_{u,v} = 2 * I\_{π\_u, π\_v} - 1$

#### 3.2 基于二分类的算法

pairwise 类方法基本就是使用二分类算法即可。

经典的算法有 基于 NN 的 SortNet，基于 NN 的 RankNet，基于 fidelity loss 的 FRank，基于 AdaBoost 的 RankBoost，基于 SVM 的 RankingSVM，基于提升树的 GBRank。

#### 3.3 缺陷

虽然 pairwise 类相较 pointwise 类 model 到一些 doc pair 间的相对顺序信息，但还是存在不少问题，回顾概述中提到的评估指标应该基于 query 和 position，

- 如果人工标注给定的是第一种和第三种，即已包含多有序类别，那么转化成 pairwise preference 时必定会损失掉一些更细粒度的相关度标注信息。
- doc pair 的数量将是 doc 数量的二次，从而 pointwise 类方法就存在的 query 间 doc 数量的不平衡性将在 pairwise 类方法中进一步放大。
- pairwise 类方法相对 pointwise 类方法对噪声标注更敏感，即一个错误标注会引起多个 doc pair 标注错误。
- pairwise 类方法仅考虑了 doc pair 的相对位置，损失函数还是没有 model 到预测排序中的位置信息。
- pairwise 类方法也没有考虑同一个 query 对应的 doc pair 间的内部依赖性，即输入空间内的样本并不是 IID 的，违反了 ML 的基本假设，并且也没有充分利用这种样本间的结构性。

#### 3.4 改进

pairwise 类方法也有一些尝试，去一定程度解决上述缺陷，比如：

- Multiple hyperplane ranker，主要针对前述第一个缺陷
- magnitude-preserving ranking，主要针对前述第一个缺陷
- IRSVM，主要针对前述第二个缺陷
- 采用 Sigmoid 进行改进的 pairwise 方法，主要针对前述第三个缺陷
- P-norm push，主要针对前述第四个缺陷
- Ordered weighted average ranking，主要针对前述第四个缺陷
- LambdaRank，主要针对前述第四个缺陷
- Sparse ranker，主要针对前述第四个缺陷


### 4. Listwise Approach

#### 4.1 特点

Listwise 类方法，其 L2R 框架具有以下特征：

- 输入空间中样本是（同一 query 对应的）所有 doc（与对应的 query）构成的多个特征向量（列表）；
- 输出空间中样本是这些 doc（和对应 query）的相关度排序列表或者排列；
- 假设空间中样本是多变量函数，对于 docs 得到其排列，实践中，通常是一个打分函数，根据打分函数对所有 docs 的打分进行排序得到 docs 相关度的排列；
- 损失函数分成两类，一类是直接和评价指标相关的，还有一类不是直接相关的。具体后面介绍。

这里讨论下，关于人工标注标签怎么转换到 listwise 类方法的输出空间：

- 如果标注直接是相关度 s_j，则 doc set 的真实标签可以利用相关度 s_j 进行比较构造出排列
- 如果标注是 pairwise preference s_{u,v}，则 doc set 的真实标签也可以利用所有 s_{u,v} 进行比较构造出排列
- 如果标注是整体排序 π，则 doc set 则可以直接得到真实标签

根据损失函数构造方式的不同，listwise 类可以分成两类直接基于评价指标的算法，间接基于评价指标的算法。

#### 4.2 直接基于评价指标的算法

直接取优化 ranking 的评价指标，也算是 listwise 中最直观的方法。但这并不简单，因为前面说过评价指标都是离散不可微的，具体处理方式有这么几种：

- 优化基于评价指标的 ranking error 的连续可微的近似，这种方法就可以直接应用已有的优化方法，如SoftRank，ApproximateRank，SmoothRank
- 优化基于评价指标的 ranking error 的连续可微的上界，如 SVM-MAP，SVM-NDCG，PermuRank
- 使用可以优化非平滑目标函数的优化技术，如 AdaRank，RankGP

上述方法的优化目标都是直接和 ranking 的评价指标有关。现在来考虑一个概念，informativeness。通常认为一个更有信息量的指标，可以产生更有效的排序模型。而多层评价指标（NDCG）相较二元评价（AP）指标通常更富信息量。因此，有时虽然使用信息量更少的指标来评估模型，但仍然可以使用更富信息量的指标来作为 loss 进行模型训练。

#### 4.3 非直接基于评价指标的算法

这里，不再使用和评价指标相关的 loss 来优化模型，而是设计能衡量模型输出与真实排列之间差异的 loss，如此获得的模型在评价指标上也能获得不错的性能。
经典的如 ，ListNet，ListMLE，StructRank，BoltzRank。

#### 4.4 缺陷

listwise 类相较 pointwise、pairwise 对 ranking 的 model 更自然，解决了 ranking 应该基于 query 和 position 问题。

listwise 类存在的主要缺陷是：一些 ranking 算法需要基于排列来计算 loss，从而使得训练复杂度较高，如 ListNet和 BoltzRank。此外，位置信息并没有在 loss 中得到充分利用，可以考虑在 ListNet 和 ListMLE 的 loss 中引入位置折扣因子。

### 5. 总结
实际上，前面介绍完，可以看出来，这三大类方法主要区别在于**<font color='#CE0000'>损失函数</font>**。不同的损失函数指引了不同的模型学习过程和输入输出空间。



