---
title: gauc&timeauc
tags:
  - GAUC
  - TimeAUC
mathjax: true
comments: false
copyright: true
date: 2020-11-27 16:48:31
categories: 基础知识
---


### 1. AUC

AUC是评估模型对pair数据，将正样本的预测分数大于负样本的预测分数的能力；

计算方式，scala写的：

```scala
// 预测值 + 标签
case class LabeledPred(predict: Double, label: Int)

def auc(points: Seq[LabeledPred]) = {
    val posNum = points.count(_.label > 0)
    val negNum = points.length - posNum

    if (posNum == 0 || negNum == 0) {
      println("Error: Lables of all samples are the same.")
      0.0
    } else {
      val sorted = points.sortBy(_.predict)

      var negSum = 0
      // pos greater than neg
      var posGTNeg = 0
      for (p <- sorted) {
        if (p.label > 0) {
          posGTNeg = posGTNeg + negSum
        } else {
          negSum = negSum + 1
        }
      }

      posGTNeg.toDouble/(posNum * negNum).toDouble
    }
}
```

上面表达的是按预测分升序排列(负样本在前，正样本在后)后遍历，

每遇到一个正样本，将在它之前的负样本数量加上(表示模型预测对的pair数量)，

每遇到一个负样本，则加1，

最后用预测对的pair对数量 **处以** 总的区分正负样本的pair对的数量，即得auc。


### 2. GAUC

在auc的基础上，按照用户、刷次等进行分组，剔除整组都是正样本或整组都是负样本的数据，

按照展现的样本数量比例为权重，计算得到模型的结果。

```scala
// 预测值 + 标签 + 用户组
  case class LabeledPredDev(device: String, labeledPred: LabeledPred)

  /**
    * @param points
    * @return
    */
  def gAuc(points: Seq[LabeledPredDev]) = {
    val userMap = points.groupBy(_.device)
        .filterNot(_._2.forall(_.labeledPred.label > 0)) // 去掉全部记录为正例的用户
        .filterNot(_._2.forall(_.labeledPred.label < 1)) // 去掉全部记录为负例的用户

    val userCntMap = userMap.mapValues(_.size)
    val userAucMap = userMap.mapValues(tmp => auc(tmp.map(_.labeledPred)))

    val cntAucPairs = for (key <- userCntMap.keys.toSeq)
      yield (userCntMap.getOrElse(key, 0), userAucMap.getOrElse(key, 0.0))

    val sumImprs = cntAucPairs.map(_._1).sum
    val weightedAuc = cntAucPairs.foldLeft(0.0){
      (sum, pair) => {
        pair._1 * pair._2 + sum
      }
    }
    weightedAuc.toDouble/sumImprs.toDouble
  }
```


### 3. 逆序数

其实上面auc的计算，体现的也是逆序数的逻辑。

正序数、逆序数可作为模型效果的衡量指标。当样本的之间存在序关系时，由样本间两两组成的，若模型预测结果的序关系与之间的序关系相同，称为正序；若模型预测结果的序关系与之间的序关系相反，称为逆序。当正序数量越多、逆序数量越少时，表明模型对序关系的刻画越准确，模型效果越好。正逆序即为正序数量与逆序数量的比值。

逆序数的计算，可以参考LeetCode上的这道题：[计算右侧小于当前元素的个数](https://leetcode-cn.com/problems/count-of-smaller-numbers-after-self/)


```python
class Solution:

    def countSmallerTL(self, nums: List[int]) -> List[int]:
        """
        暴力，O(N^2)
        """
        length = len(nums)
        count = [0 for _ in range(length)]
        for i in range(length):
            if i == length - 1:
                break
            for n in nums[i+1:]:
                if n < nums[i]:
                    count[i] += 1
        return count

    def countSmaller(self, nums: List[int]) -> List[int]:
        """
        归并排序，O(NlogN)
        """
        res = [0] * len(nums)
        length = len(nums)

        index_nums = list(zip(range(length), nums))

        def merge_sort(arr):
            if len(arr) <= 1:
                return arr
            mid = len(arr) // 2
            left = merge_sort(arr[:mid])
            right = merge_sort(arr[mid:])
            return merge(left, right)

        def merge(left, right):
            temp = []
            i, j = 0, 0
            while i < len(left) or j < len(right):
                if j == len(right) or i < len(left) and left[i][1] <= right[j][1]:
                    temp.append(left[i])
                    res[left[i][0]] += j
                    i += 1
                else:
                    temp.append(right[j])
                    j+= 1
            return temp
        
        merge_sort(index_nums)
        return res
```


### 4. TimeAUC

对于点击，我们可以直接使用auc、gauc来进行计算，衡量模型的排序能力。

对于时长目标的时候，我们同样可以参考逆序数和auc的计算逻辑，构造一个timeauc的指标衡量模型对有点击行为的时长预测的能力。

计算有点击的时长的时候，需要将时长等于0的样本过滤掉，

- timeAuc:
  - 计算逆序数累加和(模型预测错误的数量)
  - 1.0 - 逆序数累加和 / pair总数

其实在真正计算逆序率的时候，是根据肯德尔相关系数来计算的。

```python
import numpy as np
from scipy.stats._stats import _kendall_dis

def inverse_ratio(x, y):
  """
  x = y_true, duration
  y = y_pred, prob_dur
  """
  n = len(x)
  tot = n * (n - 1) // 2 

  perm = np.argsort(y)  # sort on y and convert y to dense ranks
  x, y = x[perm], y[perm]
  y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

  # stable sort on x and convert x to dense ranks
  perm = np.argsort(x, kind='mergesort')
  x, y = x[perm], y[perm]
  x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

  dis = _kendall_dis(x, y)

  obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
  cnt = np.diff(np.nonzero(obs)[0]).astype('int64', copy=False)

  ntie = (cnt * (cnt - 1) // 2).sum()  # joint ties
  xtie = count_rank_tie(x)     # ties in x, stats
  ytie = count_rank_tie(y)     # ties in y, stats

  if xtie == tot or ytie == tot:
    return np.nan

  # Note that tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
  #               = con + dis + xtie + ytie - ntie
  # con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
  # tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)

  # Limit range to fix computational errors
  # tau = min(1., max(-1., tau))

  con = tot - dis - xtie - ytie + ntie

  inv_ratio = dis / (con + dis)

  if con + dis == 0:
    inv_ratio = np.nan 
    # tau = np.nan

  return inv_ratio

def count_rank_tie(ranks):
  cnt = np.bincount(ranks).astype('int64', copy=False)
  cnt = cnt[cnt > 1]
  return (cnt * (cnt - 1) // 2).sum()
```

同样的也可以计算GroupTimeAUC


--------------

> [AUC的理解与计算](https://zhuanlan.zhihu.com/p/37576578)
> [模型排序中的逆序对计算](https://www.jianshu.com/p/e9813ac25cb6)

