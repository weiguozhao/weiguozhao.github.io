---
layout: post
title: 随机森林 Random Forest
date: 2018-04-21 20:10 +0800
categories: 机器学习
tags:
- 集成学习
- 算法
- 技术
mathjax: true
copyright: true
---

## <center> 随即森林 - Random Forest </center>

目录
* 基本概念
* 袋外错误率(out of bag error, oob error)
* 例子
* 特点及细节

> TODO 分布式实现原理 
> https://www.jianshu.com/p/d90189008864 
> https://www.ibm.com/developerworks/cn/opensource/os-cn-spark-random-forest/

------

### 基本概念

随机森林是由多颗CART树组成的，具体决策树部分知识[在这里](/2018/03/decision-tree/)

前面提到，随机森林中有许多的分类树。我们要将一个输入样本进行分类，我们需要将输入样本输入到每棵树中进行分类。
打个形象的比喻：森林中召开会议，讨论某个动物到底是老鼠还是松鼠，每棵树都要独立地发表自己对这个问题的看法，也就是每棵树都要投票。
该动物到底是老鼠还是松鼠，要依据**投票情况**来确定，获得票数最多的类别就是森林的分类结果。
森林中的每棵树都是独立的，99.9%不相关的树做出的预测结果涵盖所有的情况，这些预测结果将会彼此抵消。
少数优秀的树的预测结果将会超脱于芸芸“噪音”，做出一个好的预测。
将若干个弱分类器的分类结果进行投票选择，从而组成一个强分类器，这就是随机森林bagging的思想
（bagging的代价是不用单棵决策树来做预测，具体哪个变量起到重要作用变得未知，所以bagging改进了预测准确率但损失了解释性）。

有了树我们就可以分类了，但是森林中的每棵树是怎么生成的呢？

每棵树的按照如下规则生成：

（1）如果训练集大小为N，对于每棵树而言，**随机且有放回**地从训练集中的抽取N个训练样本（这种采样方式称为bootstrap sample方法），作为该树的训练集；从这里可以知道：每棵树的训练集都是不同的，而且里面包含重复的训练样本。

（2）如果每个样本的特征维度为M，指定一个常数m<<M，随机地从M个特征中选取m个特征子集，每次树(节点)进行分裂时，从这m个特征中选择最优的；
*具体意思是：每次进行分支的时候，都重新随机选择m个特征，计算这m个特征里，分裂效果最好的那个特征进行分类*

（3）每棵树都尽最大程度的生长，并且没有剪枝过程。

一开始我们提到的随机森林中的“随机”就是指的这里的两个随机性（采样数据、采样特征）。两个随机性的引入对随机森林的分类性能至关重要。由于它们的引入，使得随机森林不容易陷入过拟合，并且具有很好得抗噪能力（比如：对缺省值不敏感）。

随机森林分类效果（错误率）与两个因素有关：
* 森林中任意两棵树的相关性：相关性越大，错误率越大；
* 森林中每棵树的分类能力：每棵树的分类能力越强，整个森林的错误率越低。


-------

### 袋外错误率(out of bag error, oob error)

上面我们提到，构建随机森林的关键问题就是如何选择最优的m，要解决这个问题主要依据计算袋外错误率oob error。

随机森林有一个重要的优点就是，没有必要对它进行交叉验证或者用一个独立的测试集来获得误差的一个无偏估计。
它可以在内部进行评估，也就是说在生成的过程中就可以对误差建立一个无偏估计。

我们知道，在构建每棵树时，我们对训练集使用了不同的bootstrap sample（随机且有放回地抽取）。
所以对于每棵树而言（假设对于第k棵树），大约有1/3(\\( 1/e \approx 0.368 \\))的训练实例没有参与第k棵树的生成，它们称为第k棵树的oob样本。

而这样的采样特点就允许我们进行oob估计，它的计算方式如下：(note：以样本为单位)

（1）对每个样本，计算它作为oob样本的树对它的分类情况（约1/3的树）；

（2）然后以简单多数投票作为该样本的分类结果；

（3）最后用误分个数占样本总数的比率作为随机森林的oob误分率。

oob误分率是随机森林泛化误差的一个无偏估计，它的结果近似于需要大量计算的k折交叉验证。


------

### 例子

描述：根据已有的训练集已经生成了对应的随机森林，随机森林如何利用某一个人的年龄（Age）、性别（Gender）、教育情况（Highest Educational Qualification）、工作领域（Industry）以及住宅地（Residence）共5个字段来预测他的收入层次。

收入层次 :

  Band 1 : < 40,000

  Band 2: 40,000 – 150,000

  Band 3: > 150,000

随机森林中每一棵树都可以看做是一棵CART（分类回归树），这里假设森林中有5棵CART树，总特征个数N=5，我们取m=1（这里假设每个CART树对应一个不同的特征）。

CART 1 : Variable Age

![age](/posts_res/2018-04-21-randomforest/3-1.png)

CART 2 : Variable Gender

![gender](/posts_res/2018-04-21-randomforest/3-2.png)

CART 3 : Variable Education

![edu](/posts_res/2018-04-21-randomforest/3-3.png)

CART 4 : Variable Residence

![res](/posts_res/2018-04-21-randomforest/3-4.png)

CART 5 : Variable Industry

![industy](/posts_res/2018-04-21-randomforest/3-5.png)


我们要预测的某个人的信息如下：

1. Age : 35 years
2. Gender : Male
3. Highest Educational Qualification : Diploma holder
4. Industry : Manufacturing
5. Residence : Metro

根据这五棵CART树的分类结果，我们可以针对这个人的信息建立收入层次的分布情况：

![level](/posts_res/2018-04-21-randomforest/3-6.png)

最后，我们得出结论，这个人的收入层次70%是一等，大约24%为二等，6%为三等，所以最终认定该人属于一等收入层次（< 40,000）。


-------

### 特点及细节

* 在当前所有算法中，具有极好的准确率
* 能够有效地运行在大数据集上
* 能够处理具有高维特征的输入样本，而且不需要降维
* 能够评估各个特征在分类问题上的重要性
* 在生成过程中，能够获取到内部生成误差的一种无偏估计
* 对于缺省值问题也能够获得很好得结果

```
为什么要随机抽样训练集？
如果不进行随机抽样，每棵树的训练集都一样，那么最终训练出的树分类结果也是完全一样的，这样的话完全没有bagging的必要；

为什么要有放回地抽样？
如果不是有放回的抽样，那么每棵树的训练样本都是不同的，都是没有交集的，这样每棵树都是"有偏的"，都是"片面的"，也就是说每棵树训练出来都是有很大的差异的；而随机森林最后分类取决于多棵树（弱分类器）的投票表决，这种表决应该是"求同"，因此使用完全不同的训练集来训练每棵树这样对最终分类结果是没有帮助的，这样无异于是"盲人摸象"。
```

<br>

```
减小特征选择个数m，树的相关性和分类能力也会相应的降低；增大m，两者也会随之增大。
所以关键问题是如何选择最优的m（或者是范围），这也是随机森林唯一的一个参数。
```

<br>

```
RF特征选择
首先特征选择的目标有两个：
1：找到与分类结果高度相关的特征变量。
2：选择出数目较少的特征变量并且能够充分的预测应变量的结果。

特征选择的步骤：
（1）对于每一棵决策树，计算其oob error
（2）随机的修改OOB中的每个特征xi的值，计算oob error_2，再次计算重要性
（3）按照特征的重要性排序，然后剔除后面不重要的特征
（4）然后重复以上步骤，直到选出m个特征。
```

<br>

```
RF特征重要性的度量方法
（1）对于每一棵决策树，计算其oob error_0
（2）选取一个特征，随机对特征加入噪声干扰，再次计算oob error_1
（3）特征的重要性 = ∑(oob error_1-oob error_0) / 随机森林中决策树的个数
（4）对随机森林中的特征变量按照特征重要性降序排序。
（5）然后重复以上步骤，直到选出m个特征。

解释：
用这个公式来度量特征重要性，原因是：给某个特征随机的加入噪声后，如果oob error增大，说明这个特征对样本分类的结果影响比较大，说明重要程度比较高。
```


-------

### 参考

> [随机森林原理篇](https://blog.csdn.net/a819825294/article/details/51177435)

> [随机森林（Random Forest）](https://www.cnblogs.com/maybe2030/p/4585705.html)

> [Introduction to Random forest (博主：爱67)](http://www.cnblogs.com/Bfrican/p/4463292.html)

> [随即森林原理-简书](https://www.jianshu.com/p/57e862d695f2)

> [Introduction to Random forest – Simplified](https://www.analyticsvidhya.com/blog/2014/06/introduction-random-forest-simplified/)