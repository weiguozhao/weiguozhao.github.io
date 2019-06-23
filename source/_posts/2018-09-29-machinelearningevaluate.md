---
layout: post
title: 机器学习评价指标
date: 2018-09-29 12:10 +0800
categories: 评估指标
tags:
- 机器学习
mathjax: true
copyright: false
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


- 受试者曲线 ROC
    - 在ROC空间中，每个点的横坐标是 **FPR**，纵坐标是 **TPR**，这也就描绘了分类器在 TP(真正的正例) 和 FP(错误的正例) 间的 trade-off。
    - ROC的主要分析工具是一个画在ROC空间的曲线(ROC curve)，对于二分类问题，实例的值往往是连续值，我们通过设定一个阈值，将实例分类到正例或负例（比如大于阈值划分为正类）。
    - 因此可以变化阈值，根据不同的阈值进行分类，根据分类结果计算得到ROC空间中相应的点，连接这些点就形成ROC curve。
    - ROC curve经过(0,0)、(1,1)，实际上(0,0)和(1,1)连线形成的ROC curve实际上代表的是一个随机分类器；一般情况下，这个曲线都应该处于(0,0)和(1,1)连线的上方。


- 受试者曲线下面积 AUC
    - 用ROC curve来表示分类器的performance很直观好用。可是，人们总是希望能有一个数值来标志分类器的好坏。
    - 于是 AUC (Area Under roc Curve)就出现了。AUC的值就是处于ROC curve下方的那部分面积的大小；通常，AUC的值介于0.5到1.0之间，较大的AUC代表了较好的Performance。

