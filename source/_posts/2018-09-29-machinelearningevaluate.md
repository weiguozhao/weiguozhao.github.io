---
layout: post
title: 机器学习评价指标
date: 2018-09-29 12:10 +0800
categories: 基础知识
tags:
- 评估方法
mathjax: true
copyright: false
comments: false
---


- 混淆矩阵 Confusion Matrix

<table>
<tr>
 <th></th>
 <th>实际负例</th>
 <th>实际正例</th>
</tr>

<tr>
 <td>预测负例</td>
 <td>TN  (将负例预测为负例个数)</td>
 <td>FN  (将正例预测为负例个数)</td>
</tr>

<tr>
 <td>实际正例</td>
 <td>FP  (将负例预测为正例个数)</td>
 <td>TP  (将正例预测为正例个数)</td>
</tr>
</table>


- 召回率 Recall

$$ Recall = \frac{TP}{TP + FN} $$


- 精确率 Precision

$$ Precision = \frac{TP}{TP + FP} $$


- 准确率 Accuracy

$$ Accuracy = \frac{TP + FN}{TP + TN + FP + FN} $$


- 调和均值 F-score

$$ F-score = \frac{2 \times Precision \times Recall}{Precision + Recall} $$


- True Positive Rate (TPR)
    - 代表将正例分对的概率

$$ TPR = \frac{TP}{TP + FN} $$

- False Positive Rate (FPR)
    - 代表将负例错分为正例的概率

$$ FPR = \frac{FP}{FP + TN} $$


### 受试者曲线 ROC

- 在ROC空间中，每个点的横坐标是 **FPR**，纵坐标是 **TPR**，这也就描绘了分类器在 TP(真正的正例) 和 FP(错误的正例) 间的 trade-off。
- ROC的主要分析工具是一个画在ROC空间的曲线(ROC curve)，对于二分类问题，实例的值往往是连续值，我们通过设定一个阈值，将实例分类到正例或负例（比如大于阈值划分为正类）。
- 因此可以变化阈值，根据不同的阈值进行分类，根据分类结果计算得到ROC空间中相应的点，连接这些点就形成ROC curve。
- ROC curve经过(0,0)、(1,1)，实际上(0,0)和(1,1)连线形成的ROC curve实际上代表的是一个随机分类器；一般情况下，这个曲线都应该处于(0,0)和(1,1)连线的上方。


### 受试者曲线下面积 AUC

- 用ROC curve来表示分类器的performance很直观好用。可是，人们总是希望能有一个数值来标志分类器的好坏。
- 于是 AUC (Area Under roc Curve)就出现了。AUC的值就是处于ROC curve下方的那部分面积的大小；通常，AUC的值介于0.5到1.0之间，较大的AUC代表了较好的Performance。


### AP(Average Precision)

假使当我们使用google搜索某个关键词，返回了10个结果。当然最好的情况是这10个结果都是我们想要的相关信息。
但是假如只有部分是相关的，比如5个，那么这5个结果如果被显示的比较靠前也是一个相对不错的结果。
但是如果这个5个相关信息从第6个返回结果才开始出现，那么这种情况便是比较差的。
这便是AP所反映的指标，与recall的概念有些类似，不过是“顺序敏感的recall”。

比如对于用户 $u$, 我们给他推荐一些物品，那么 $u$ 的平均准确率定义为：

$$
AP_u = \frac{1}{\| I^{te}_u \|} \sum _{i \in I^{te}_u} \frac{\sum _{j \in I^{te}_u} \delta ( p _{uj} \lt p _{ui} ) + 1}{p _{ui}} 
$$

在这里 $p\_{ui}$ 表示推荐列表中物品 $i$ 的排序位置。$p\_{uj} \lt p\_{ui}$ 表示在对用户 $u$ 的排序列表中物品 $j$ 的排序位置在物品 $i$ 的前面。

```python
def AP(ranked_list, ground_truth):
    """
    Compute the average precision (AP) of a list of ranked items
    """
    hits = 0
    sum_precs = 0
    for n in range(len(ranked_list)):
        if ranked_list[n] in ground_truth:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0
```


### MAP(Mean Average Precision)

即为所有用户 $u$ 的`AP`再取均值(mean)而已。

$$
MAP = \frac{\sum_{u \in U^{te}} AP_{u}}{ U^{te} }
$$


### CG(Cummulative Gain)

直接翻译的话叫做“累计增益”。 在推荐系统中，CG即将每个推荐结果相关性(relevance)的分值累加后作为整个推荐列表(list)的得分。

$$
CG_k = \sum_{i=1}^k rel_i
$$

这里， $rel\_i$ 表示处于位置 $i$ 的推荐结果的相关性，$k$ 表示所要考察的推荐列表的大小。


### DCG(Discounted Cummulative Gain)

CG的一个缺点是没有考虑每个推荐结果处于不同位置对整个推荐效果的影响，例如我们总是希望相关性高的结果应排在前面。显然，如果相关性低的结果排在靠前的位置会严重影响用户体验，
所以在CG的基础上引入位置影响因素，即DCG(Discounted Cummulative Gain), “Discounted”有打折，折扣的意思，这里指的是对于排名靠后推荐结果的推荐效果进行“打折处理”:

$$
\begin{equation*}
DCG_k = \sum_{i=1}^k \frac{2^{rel_i}-1}{\log_2 \left(i+1\right)}
\end{equation*}
$$

- 分子部分 $2^{rel\_i}−1$, $rel\_i$越大，即推荐结果 $i$ 的相关性越大，推荐效果越好， DCG越大。
- 分母部分 $log\_2 (i+1)$, $i$ 表示推荐结果的位置，$i$ 越大，则推荐结果在推荐列表中排名越靠后，推荐效果越差，DCG越小。


### NDCG(Normalized Discounted Cummulative Gain)

DCG仍然有其局限之处，即不同的推荐列表之间，很难进行横向的评估。而我们评估一个推荐系统，不可能仅使用一个用户的推荐列表及相应结果进行评估，
而是对整个测试集中的用户及其推荐列表结果进行评估。 那么不同用户的推荐列表的评估分数就需要进行归一化，也即`NDCG(Normalized Discounted Cummulative Gain)`。

在介绍`NDCG`之前，还需要了解一个概念：`IDCG`, 即`Ideal DCG`， 指推荐系统为某一用户返回的最好推荐结果列表，
即假设返回结果按照相关性排序，最相关的结果放在最前面，此序列的`DCG`为`IDCG`。因此`DCG`的值介于 $(0, IDCG]$，故`NDCG`的值介于$(0,1]$。

对于用户 $u$ 的 $NDCG@k$ 定义为：

$$
\begin{equation*}
NDCG_u@k = \frac{DCG_u@k}{IDCG_u}
\end{equation*}
$$

这里的 $k$ 表示推荐列表的大小。那么，则有：

$$
\begin{equation*}
NDCG@k = \frac{\sum_{u\in \mathcal{U}^{te}}NDCG_u@k}{|\mathcal{U}^{te}|}
\end{equation*}
$$

在具体操作中， 可以事先确定推荐目标和推荐结果的相关性分级。
- 例如可以使用 $0，1$ 分别表示相关或不相关，比如此处我们用 $ref\_i = \delta( i \in I\_u^{te} )$, 在这里如果 $x$ 为`true`, 则 $\delta (x) = 1$，否则 $\delta(x)=0$。
- 或是使用 $0 \sim 5$ 分别表示严重不相关到非常相关, 也即相当于确定了 $rel$ 值的范围。之后对于每一个推荐目标的返回结果给定 $rel$ 值，然后使用`DCG`的计算公式计计算出返回结果的`DCG`值。使用根据排序后的 `rel` 值序列计算`IDCG`值，即可计算`NDCG`.


---------------------------

> [推荐系统常见评测标准之MAP与NDCG](https://blog.csdn.net/simple_the_best/article/details/52296608)
