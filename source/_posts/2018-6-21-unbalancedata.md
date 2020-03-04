---
layout: post
title: 正负样本不平衡的处理方法
date: 2018-06-21 12:10 +0800
categories: 机器学习
tags:
- 特征选择
mathjax: true
copyright: true
---


### 1 通过过抽样和欠抽样解决样本不均衡

抽样是解决样本分布不均衡相对简单且常用的方法，包括过抽样和欠抽样两种。

**过抽样**

过抽样（也叫上采样、over-sampling）方法通过增加分类中少数类样本的数量来实现样本均衡，最直接的方法是简单复制少数类样本形成多条记录，这种方法的缺点是如果样本特征少而可能导致过拟合的问题；经过改进的过抽样方法通过在少数类中加入随机噪声、干扰数据或通过一定规则产生新的合成样本，例如SMOTE算法，它构造新的小类样本而不是产生小类中已有的样本的副本，即该算法构造的数据是新样本，原数据集中不存在的。该基于距离度量选择小类别下两个或者更多的相似样本，然后选择其中一个样本，并随机选择一定数量的邻居样本对选择的那个样本的一个属性增加噪声，每次处理一个属性。这样就构造了更多的新生数据。具体可以参见[原始论文](/posts_res/2018-04-03-interview/SMOTE_Synthetic Minority Over-sampling Technique.pdf)。

这里有SMOTE算法的多个不同语言的实现版本： 
* Python: [UnbalancedDataset](https://github.com/fmfn/UnbalancedDataset)模块提供了SMOTE算法的多种不同实现版本，以及多种重采样算法。
* R: [DMwR package](https://blog.csdn.net/heyongluoyao8/article/details/DMwR%20packagehttps://cran.r-project.org/web/packages/DMwR/index.html)。
* Weka: [SMOTE supervised filter](http://weka.sourceforge.net/doc.packages/SMOTE/weka/filters/supervised/instance/SMOTE.html)

**欠抽样**

欠抽样（也叫下采样、under-sampling）方法通过减少分类中多数类样本的样本数量来实现样本均衡，最直接的方法是随机地去掉一些多数类样本来减小多数类的规模，缺点是会丢失多数类样本中的一些重要信息。

总体上，过抽样和欠抽样更适合大数据分布不均衡的情况，尤其是第一种（过抽样）方法应用更加广泛。


------

### 2 通过正负样本的惩罚权重解决样本不均衡

通过正负样本的惩罚权重解决样本不均衡的问题的思想是在算法实现过程中，对于分类中不同样本数量的类别分别赋予不同的权重（一般思路分类中的小样本量类别权重高，大样本量类别权重低），然后进行计算和建模。

使用这种方法时需要对样本本身做额外处理，只需在算法模型的参数中进行相应设置即可。很多模型和算法中都有基于类别参数的调整设置，以scikit-learn中的SVM为例，通过在class_weight : {dict, 'balanced'}中针对不同类别针对不同的权重，来手动指定不同类别的权重。如果使用其默认的方法balanced，那么SVM会将权重设置为与不同类别样本数量呈反比的权重来做自动均衡处理，计算公式为：n_samples / (n_classes * np.bincount(y))。

如果算法本身支持，这种思路是更加简单且高效的方法。


----------

### 3 通过组合/集成方法解决样本不均衡

组合/集成方法指的是在每次生成训练集时使用所有分类中的小样本量，同时从分类中的大样本量中随机抽取数据来与小样本量合并构成训练集，这样反复多次会得到很多训练集和训练模型。最后在应用时，使用组合方法（例如投票、加权投票等）产生分类预测结果。

例如，在数据集中的正、负例的样本分别为100和10000条，比例为1:100。此时可以将负例样本（类别中的大量样本集）随机分为100份（当然也可以分更多），每份100条数据；然后每次形成训练集时使用所有的正样本（100条）和随机抽取的负样本（100条）形成新的数据集。如此反复可以得到100个训练集和对应的训练模型。

这种解决问题的思路类似于随机森林。在随机森林中，虽然每个小决策树的分类能力很弱，但是通过大量的“小树”组合形成的“森林”具有良好的模型预测能力。

如果计算资源充足，并且对于模型的时效性要求不高的话，这种方法比较合适。


-----------

### 4 通过特征选择解决样本不均衡

上述几种方法都是基于数据行的操作，通过多种途径来使得不同类别的样本数据行记录均衡。除此以外，还可以考虑使用或辅助于基于列的特征选择方法。

一般情况下，样本不均衡也会导致特征分布不均衡，但如果小类别样本量具有一定的规模，那么意味着其特征值的分布较为均匀，可通过选择具有显著型的特征配合参与解决样本不均衡问题，也能在一定程度上提高模型效果。


----------

### 5 尝试其它评价指标

* 混淆矩阵(Confusion Matrix)：使用一个表格对分类器所预测的类别与其真实的类别的样本统计，分别为：TP、FN、FP与TN。
* 精确度(Precision)
* 召回率(Recall)
* F1值(F1 Score)：精确度与找召回率的加权平均。
* Kappa ([Cohen kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa))
* ROC曲线(ROC Curves)：见[Assessing and Comparing Classifier Performance with ROC Curves](http://machinelearningmastery.com/assessing-comparing-classifier-performance-roc-curves-2/)


---------

### 6 尝试不同的分类算法

应该使用不同的算法对其进行比较，因为不同的算法使用于不同的任务与数据，具体请见[Why you should be Spot-Checking Algorithms on your Machine Learning Problems](https://machinelearningmastery.com/why-you-should-be-spot-checking-algorithms-on-your-machine-learning-problems/)

### 7 尝试一个新的角度理解问题 

我们可以从不同于分类的角度去解决数据不均衡性问题，我们可以把那些小类的样本作为异常点(outliers)，因此该问题便转化为异常点检测(anomaly detection)与变化趋势检测问题(change detection)。
 
* 异常点检测即是对那些罕见事件进行识别。如通过机器的部件的振动识别机器故障，又如通过系统调用序列识别恶意程序。这些事件相对于正常情况是很少见的。
* 变化趋势检测类似于异常点检测，不同在于其通过检测不寻常的变化趋势来识别。如通过观察用户模式或银行交易来检测用户行为的不寻常改变。 

将小类样本作为异常点这种思维的转变，可以帮助考虑新的方法去分离或分类样本。这两种方法从不同的角度去思考，让你尝试新的方法去解决问题。

推荐看论文[Learning from Imbalanced Data](/posts_res/2018-04-03-interview/Learning from Imbalanced Data.pdf)

主要包括四大类方法:
> 1. Sampling
2. Cost Sensitive Methods
3. Kernal-Based Methods and Active Learning Methods
4. One-Class Learning or Novelty Detection Methods


**极端情况下，只有正例(负例)如何做分类？**

*当作异常点检测问题或变化趋势检测问题*

> 识别异常点可以用四分位数(Tukey's text)方法，详见[这里](https://www.zhihu.com/question/38066650/answer/202282227)
[One-Class SVM介绍](https://zhuanlan.zhihu.com/p/32784067)


-------------

> 1. [CTR 预估正负样本 不平衡，请问怎么解决?](https://www.zhihu.com/question/27535832/answer/223882022)
2. [在分类中如何处理训练集中不平衡问题](https://blog.csdn.net/heyongluoyao8/article/details/49408131)
