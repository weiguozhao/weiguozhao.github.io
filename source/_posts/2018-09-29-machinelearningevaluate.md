---
layout: post
title: 各种评价指标
date: 2018-09-29 12:10 +0800
categories: 基础知识
tags:
    - 评估方法
    - ROC & AUC & GAUC
    - AP & MAP
    - CG & DCG & NDCG
mathjax: true
copyright: false
comments: false
---


### 1. 混淆矩阵及其衍生

#### 1.1 Confusion Matrix

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

#### 1.2 召回率 Recall

$$ Recall = \frac{TP}{TP + FN} $$

#### 1.3 精确率 Precision

$$ Precision = \frac{TP}{TP + FP} $$

#### 1.4 准确率 Accuracy

$$ Accuracy = \frac{TP + FN}{TP + TN + FP + FN} $$

#### 1.5 调和均值 F-score

$$ F-score = \frac{2 \times Precision \times Recall}{Precision + Recall} $$

#### 1.6 True Positive Rate (TPR)：代表将正例分对的概率

$$ TPR = \frac{TP}{TP + FN} $$

#### 1.7 False Positive Rate (FPR)：代表将负例错分为正例的概率

$$ FPR = \frac{FP}{FP + TN} $$


### 2. ROC & AUC & GAUC

#### 2.1 受试者曲线 ROC

在ROC空间中，每个点的横坐标是 **FPR(假正数-负样本被预测为正样本的个数/概率)**，纵坐标是 **TPR(真正数-正样本被预测为正样本的个数/概率)**，
这也就描绘了分类器在 TP(真正的正例) 和 FP(错误的正例) 间的 trade-off。
ROC的主要分析工具是一个画在ROC空间的曲线(ROC curve)，对于二分类问题，实例的值往往是连续值，我们通过设定一个阈值，将实例分类到正例或负例（比如大于阈值划分为正类）。
因此可以变化阈值，根据不同的阈值进行分类，根据分类结果计算得到ROC空间中相应的点，连接这些点就形成ROC curve。
ROC curve经过(0,0)、(1,1)，实际上(0,0)和(1,1)连线形成的ROC curve实际上代表的是一个随机分类器；一般情况下，这个曲线都应该处于(0,0)和(1,1)连线的上方。


#### 2.2 受试者曲线下面积 AUC

用ROC curve来表示分类器的performance很直观好用。可是，人们总是希望能有一个数值来标志分类器的好坏。于是 AUC (Area Under roc Curve)就出现了。AUC的值就是处于ROC curve下方的那部分面积的大小。

**计算方法一**

- example1: 

    <table>
        <tr>
            <th>index</th>
            <th>label</th>
            <th>predictScore</th>
        </tr>
        <tr>
            <td>0</td>
            <td>0</td>
            <td>0.1</td>
        </tr>
        <tr>
            <td>1</td>
            <td>0</td>
            <td>0.4</td>
        </tr>
        <tr>
            <td>2</td>
            <td>1</td>
            <td>0.35</td>
        </tr>
        <tr>
            <td>3</td>
            <td>1</td>
            <td>0.8</td>
        </tr>
    </table>

    负样本和正样本两两组合后的index集合是：(0, 2)、(0, 3)、(1, 2)、(1, 3)

    - (0, 2): 中正样本概率大于负样本概率(0.35 > 0.1), 计 1
    - (0, 3): 中正样本概率大于负样本概率(0.8 > 0.1), 计 1
    - (1, 2): 中正样本概率小于负样本概率(0.35 < 0.4), 计 0
    - (1, 3): 中正样本概率大于负样本概率(0.8 > 0.4), 计 1

    所以这个batch的AUC为：$AUC = \frac{3}{4} = 0.75$

- example2: 当存在正负样本得到的 `predictScore` 相同时，计 $0.5$

    <table>
        <tr>
            <th>index</th>
            <th>label</th>
            <th>predictScore</th>
        </tr>
        <tr>
            <td>0</td>
            <td>0</td>
            <td>0.1</td>
        </tr>
        <tr>
            <td>1</td>
            <td>0</td>
            <td>0.4</td>
        </tr>
        <tr>
            <td>2</td>
            <td>1</td>
            <td>0.4</td>
        </tr>
        <tr>
            <td>3</td>
            <td>1</td>
            <td>0.8</td>
        </tr>
    </table>

    负样本和正样本两两组合后的index集合是：(0, 2)、(0, 3)、(1, 2)、(1, 3)

    - (0, 2): 中正样本概率大于负样本概率(0.4 > 0.1), 计 1
    - (0, 3): 中正样本概率大于负样本概率(0.8 > 0.1), 计 1
    - (1, 2): 中正样本概率小于负样本概率(0.4 = 0.4), 计 0.5
    - (1, 3): 中正样本概率大于负样本概率(0.8 > 0.4), 计 1

    所以这个batch的AUC为：$AUC = \frac{3.5}{4} = 0.875$

```python
def calculate_auc(y_true, y_pred):
    import numpy as np
    
    pos_index = np.where(y_true == 1)
    neg_index = np.where(y_true == 0)

    pos_cnt = sum(y_true)
    neg_cnt = len(y_true) - pos_cnt

    count = 0
    for pindex in pos_index[0]:
        for nindex in neg_index[0]:
            if y_pred[pindex] > y_pred[nindex]:
                count += 1
            elif y_pred[pindex] == y_pred[nindex]:
                count += 0.5
            else:
                count += 0

    return 1.0 * count / (pos_cnt * neg_cnt)
```

**计算方法二**

- $$AUC = \frac{\sum_{ins_i \in positiveclass} rank_{ins_i} - 0.5 \times M \times (M + 1)}{M \times N} $$
- 其中 $rank\_{ins\_i}$ 代表第 $i$ 条样本的序号。(概率得分从小到大，排在第$rank$个位置); 
- $M$表示正样本的个数; $N$表示负样本的个数; 
- $\sum\_{ins\_i \in positiveclass}$ 把所有正样本的累加

- 使用example1中的样本计算，按照概率排序后得到：

    <table>
        <tr>
            <th>index</th>
            <th>label</th>
            <th>predictScore</th>
            <th>rank</th>
        </tr>
        <tr>
            <td>0</td>
            <td>0</td>
            <td>0.1</td>
            <td>1</td>
        </tr>
        <tr>
            <td>1</td>
            <td>1</td>
            <td>0.35</td>
            <td>2</td>
        </tr>
        <tr>
            <td>2</td>
            <td>0</td>
            <td>0.4</td>
            <td>3</td>
        </tr>
        <tr>
            <td>3</td>
            <td>1</td>
            <td>0.8</td>
            <td>4</td>
        </tr>
    </table>

    按照上面的公式，将正样本的序号加起来，即index=1, index=3的rank值加起来，之后减去一个常数项 $0.5 \times M \times (M + 1)$;
    得到的公式结果：$\frac{(2 + 4) - 0.5 \times 2 \times (2 + 1)}{2 \times 2} = \frac{6 - 3}{4} = 0.75$

- 当存在相同的 `predictScore`, 并且其中存在着正负样本

    <table>
        <tr>
            <th>index</th>
            <th>label</th>
            <th>predictScore</th>
            <th>rank</th>
        </tr>
        <tr>
            <td>0</td>
            <td>0</td>
            <td>0.3</td>
            <td>1</td>
        </tr>
        <tr>
            <td>1</td>
            <td>1</td>
            <td>0.5</td>
            <td>2</td>
        </tr>
        <tr>
            <td>2</td>
            <td>1</td>
            <td>0.5</td>
            <td>3</td>
        </tr>
        <tr>
            <td>3</td>
            <td>0</td>
            <td>0.5</td>
            <td>4</td>
        </tr>
        <tr>
            <td>4</td>
            <td>0</td>
            <td>0.5</td>
            <td>5</td>
        </tr>
        <tr>
            <td>5</td>
            <td>1</td>
            <td>0.7</td>
            <td>6</td>
        </tr>
        <tr>
            <td>6</td>
            <td>1</td>
            <td>0.8</td>
            <td>7</td>
        </tr>
    </table>

    **这里需要注意的是：相等概率得分的样本，无论正负，谁在前，谁在后无所谓。**

    由于只考虑正样本的rank值：  
    对于正样本index=1，其rank值为 (5+4+3+2)/4  
    对于正样本index=2，其rank值为 (5+4+3+2)/4  
    对于正样本index=5，其rank值为 6  
    对于正样本index=6，其rank值为 7  

    最终得到：$\frac{(5+4+3+2)/4 + (5+4+3+2)/4 + 6 + 7 - 0.5 \times 4 \times (4 + 1)}{4 \times 3} = \frac{10}{12}$

```python
def calculate_auc(y_true, y_pred):
    pos_cnt = sum(y_true)
    neg_cnt = len(y_true) - pos_cnt
    sum_rank = 0

    ranked_data = sorted(zip(y_true, y_pred), key=lambda x: x[1])

    score2rank = {}
    for index, (label, score) in enumerate(ranked_data):
        if score not in score2rank:
            score2rank[score] = [index + 1]
        else:
            score2rank[score].append(index + 1)

    for label, score in ranked_data:
        if label == 1:
            sum_rank += sum(score2rank[score]) / 1.0 * len(score2rank[score])

    numerator = sum_rank - 0.5 * pos_cnt * (pos_cnt + 1)
    denominator = pos_cnt * neg_cnt
    return numerator / denominator
```

#### 2.3 推荐系统常用 GAUC (group AUC)

即按照用户进行聚合后单独计算每个用户的AUC，然后将所有用户的AUC进行按照样本数量进行加权平均

```python
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import numpy as np

def cal_group_auc(labels, preds, user_id_list):
    """Calculate group auc"""
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    # 标记一个用户是不是全是正样本或者全是负样本
    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    for user_id in group_flag:
        # 全是正样本或负样本的用户不统计
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]),
                                np.asarray(group_score[user_id]))
            # 用户的auc按照样本数量进行加权
            total_auc += auc * len(group_truth[user_id])
            # 总的统计auc的样本数量
            impression_total += len(group_truth[user_id])

    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    return group_auc
```


### 3. AP & MAP

#### 3.1 AP(Average Precision)

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

#### 3.2 MAP(Mean Average Precision)

即为所有用户 $u$ 的`AP`再取均值(mean)而已。

$$
MAP = \frac{\sum_{u \in U^{te}} AP_{u}}{ U^{te} }
$$


### 4. CG & DCG & NDCG

#### 4.1 CG(Cummulative Gain)

直接翻译的话叫做“累计增益”。 在推荐系统中，CG即将每个推荐结果相关性(relevance)的分值累加后作为整个推荐列表(list)的得分。

$$
CG_k = \sum_{i=1}^k rel_i
$$

这里， $rel\_i$ 表示处于位置 $i$ 的推荐结果的相关性，$k$ 表示所要考察的推荐列表的大小。


#### 4.2 DCG(Discounted Cummulative Gain)

CG的一个缺点是没有考虑每个推荐结果处于不同位置对整个推荐效果的影响，例如我们总是希望相关性高的结果应排在前面。显然，如果相关性低的结果排在靠前的位置会严重影响用户体验，
所以在CG的基础上引入位置影响因素，即DCG(Discounted Cummulative Gain), “Discounted”有打折，折扣的意思，这里指的是对于排名靠后推荐结果的推荐效果进行“打折处理”:

$$
\begin{equation*}
DCG_k = \sum_{i=1}^k \frac{2^{rel_i}-1}{\log_2 \left(i+1\right)}
\end{equation*}
$$

- 分子部分 $2^{rel\_i}−1$, $rel\_i$越大，即推荐结果 $i$ 的相关性越大，推荐效果越好， DCG越大。
- 分母部分 $log\_2 (i+1)$, $i$ 表示推荐结果的位置，$i$ 越大，则推荐结果在推荐列表中排名越靠后，推荐效果越差，DCG越小。


#### 4.3 NDCG(Normalized Discounted Cummulative Gain)

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

### Ref

> 1. [推荐系统常见评测标准之MAP与NDCG](https://blog.csdn.net/simple_the_best/article/details/52296608)
> 2. [AUC的计算方法](https://blog.csdn.net/qq_22238533/article/details/78666436)
> 3. [sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
> 4. [图解AUC和GAUC](https://blog.csdn.net/zhaohang_1/article/details/92794489)
> 5. [qiaoguan/deep-ctr-prediction/DeepCross/metric.py](https://github.com/qiaoguan/deep-ctr-prediction/blob/master/DeepCross/metric.py#L12)

