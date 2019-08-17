---
layout: post
title: 朴素贝叶斯 Naive Bayes
date: 2018-03-08 10:10 +0800
categories: 机器学习
tags:
- 模型算法
mathjax: true
copyright: true
---

## <center>朴素贝叶斯 - Naive Bayes</center>

#### 目录
* 基本思想
* 算法
* 后验概率最大化的意义
* 特点
* 代码实现


------

### 1. 基本思想

朴素贝叶斯法通过训练数据集学习联合概率分布\\( P(X,Y) \\)。具体地，学习以下先验概率分布及条件概率分布。

先验概率分布：

\\[P(Y=c_k),\quad k=1,2,...,K\\]

条件概率分布: 
\\[P(X=x | Y=c\_k)=P(X^{(1)}=x^{(1)},...,X^{(n)}=x^{(n)} | Y=c\_k),\quad k=1,2,...,K\\]

于是通过贝叶斯公式\\(P(X,Y) = P(X | Y)\cdot P(Y)\\)，
可以学习到联合概率分布\\(P(X,Y)\\)。

**但是，条件概率分布\\(P(X=x \| Y=c\_k)\\)有指数级数量的参数，其估计实际是不可行的。事实上，假设\\(x^{(j)}\\) 可取值有\\(S\_j\\)个，\\(j=1,2,...,n\\)，\\(Y\\)可取值有\\(K\\)个，那么参数个数为$K\prod_{j=1}^nS\_j$。**

朴素贝叶斯法对条件概率分布作了条件独立性的假设，即假设特征之间相互独立，公式表达如下：
\\[P(X=x | Y=c\_k) \quad = \quad P(X^{(1)}=x^{(1)},...,X^{(n)}=x^{(n)} | Y=c\_k) \quad = \quad \prod\_{j=1}^{n}P(X^{(j)}=x^{(j)} | Y=c\_k)\\]


-----------

### 2. 算法

输入：

训练数据\\( T={((x_1, y_1)),...,(x\_N,y\_N)} \\)，其中\\( x\_i=(x\_i^{(1)},x\_i^{(2)},...,x\_i^{(n)})^T \\),\\( x\_i^{(j)} \\)是第\\( i \\)个样本的第\\( j \\)个特征，\\( x\_i^{(j)}\in {a\_{j1},a\_{j2},...,a\_{jS\_j}} \\), \\( a\_{jl} \\)是第\\( j \\)个特征可能取的第\\( l \\)个值，\\( j=1,2,...,n \quad l=1,2,...,S\_j, \quad y\_i \in {c\_1, c\_2, ..., c\_k} \\); 实例\\( x \\)

输出：

实例\\( x \\)的分类

（1）计算先验概率及条件概率

\\[P(Y=c_k) = \frac{\sum_{i=1}^N I(y_i=c\_k)}{N}, \quad k=1,2,...,K\\]

\\[P(X^{(j)}=a_{jl} \| Y=c_k)=\frac{\sum_{i=1}^N I(x_i^{(j)}, y_i=c_k)}{\sum_{i=1}^N I(y_i=c\_k)}\\]

\\[j=1,2,...,n; \quad l=1,2,...,S_j; \quad k=1,2,...,K\\]

（2）对于给定的实例\\( x=(x^{(1)}, x^{(2)},.., x^{(n)})^T \\)，计算

\\[P(Y=c_k) \prod_{j=1}^n P(X^{(j)}=x^{(j)} \| Y=c_k), \quad k=1,2,...,K\\]

（3）确定实例\\( x \\)的类
\\[ y = \mathop{\arg\max}\_{c\_k} P(Y=c\_k) \prod\_{j=1}^n P(X^{(j)} \| Y=c\_k)\\]


------

### 3. 后验概率最大化的意义

朴素贝叶斯法将实例分到后验概率最大的类中，这等价于期望风险最小化，假设选择0-1损失函数：
\\[ L(Y,f(X)) = \begin{cases}
1, & Y \neq f(X) \\\
0, & Y = f(X)
\end{cases} \\]

式中\\( f(X) \\)是决策函数，这时期望风险函数为
\\[ R\_{exp}(f)=E[L(Y,f(X)] \\]

期望是对联合分布 \\( P(X,Y) \\)取的，由此取条件期望
\\[
\begin{equation}
\begin{aligned}
R_{exp}(f)
& = \iint_{D_{xy}}L(y,f(x)) \cdot P(x,y) dxdy \\\
& = \int_{D_x} \int_{D_y} L(y, f(x)) \cdot P(y\|x) \cdot P(x) dxdy \\\
& = \int_{D_x} [\int_{D_y} L(x,f(x)) \cdot P(y\|x) dy] P(x) dx \\\
& = E\_X \sum\_{k=1}^K [L(c\_k, f(X)]P(c\_k|X)
\end{aligned}
\end{equation}
\\]

为了使期望风险最小化，只需要对 \\( X=x \\)逐个极小化，由此得到：
\\[
\begin{equation}
\begin{aligned}
f(x) 
& = \mathop{\arg\min}\_{y\in \mathbb{y}} \sum_{k=1}^K L(c_k,y)P(c_k|X=x) \\\
& = \mathop{\arg\min}\_{y\in \mathbb{y}} \sum_{k=1}^K P(y \neq c_k| X=x) \\\
& = \mathop{\arg\min}\_{y\in \mathbb{y}} (1-P(y=c_k|X=x) \\\
& = \mathop{\arg\max}\_{y\in \mathbb{y}} P(y=c_k|X=x)
\end{aligned}
\end{equation}
\\]

这样，根据期望奉献最小化准则就得到了后验概率最大化准则：
\\[ 
f(x)=\mathop{\arg\max}\_{y=c_k} P(c_k|X=x) 
\\]

即朴素贝叶斯法采用的原理。


------

### 4. 特点

* 优点：源于古典数学理论，有稳定的分类效率；小数据好，可增量训练；对缺失数据不敏感，算法简单。
* 缺点：独立性假设强，实际中根据属性间的关系强弱，效果不同；须知先验概率，但先验有时取决于假设；对数据的输入形式敏感。


------

### 5. 代码实现

[Naive Bayes 代码](/posts_res/2018-03-08-native-bayes/native_bayes.py)
    
    # coding:utf-8
    
    import numpy as np
    import time
    from datetime import timedelta
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    
    
    def time_consume(s_t):
        diff = time.time()-s_t
        return timedelta(seconds=int(diff))
    
    
    class NaiveBayes(object):
        def __init__(self, class_num=-1, features_dim=-1):
            self.class_num = class_num
            self.features_dim = features_dim
    
        def fit(self, X, y):
            self.nlen = X.shape[0]
            # shape meaning: class_num
            prior_prob = np.zeros(shape=self.class_num)
            # shape meaning: class_num, feature_dim, feature_value
            conditional_prob = np.zeros(shape=(self.class_num, self.features_dim, 2))
    
            for i in range(self.nlen):
                prior_prob[y[i]] += 1.0
                for j in range(self.features_dim):
                    conditional_prob[y[i]][j][X[i][j]] += 1.0
    
            for i in range(self.class_num):
                for j in range(self.features_dim):
                    p_0 = conditional_prob[i][j][0]
                    p_1 = conditional_prob[i][j][1]
                    prob_0 = p_0 / (p_0 + p_1)
                    prob_1 = p_1 / (p_0 + p_1)
                    conditional_prob[i][j][0] = prob_0
                    conditional_prob[i][j][1] = prob_1
    
            self.prior_prob = prior_prob # 没有除self.nlen，对结果无影响
            self.conditional_prob = conditional_prob
    
        def __calculate_prob__(self, sample, label):
            prob = int(self.prior_prob[label])
            for i in range(self.features_dim):
                prob *= self.conditional_prob[label][i][sample[i]]
            return prob
    
        def predict(self, X):
            if self.class_num == -1:
                raise ValueError("Please fit first.")
            y_pred = np.zeros(shape=X.shape[0])
            for i in range(X.shape[0]):
                label = 0
                prob = self.__calculate_prob__(X[i], 0)
                for j in range(1, self.class_num):
                    this_prob = self.__calculate_prob__(X[i], j)
                    if prob < this_prob:
                        prob = this_prob
                        label = j
                y_pred[i] = label
            return y_pred
    
        def accuracy(self, y_true, y_pred):
            return accuracy_score(y_true, y_pred)
    
    
    def load_data():
        """normalize value of data to {0, 1}"""
        data, target = load_breast_cancer(return_X_y=True)
        nlen = data.shape[0]
        ndim = data.shape[1]
        X = np.zeros(shape=data.shape, dtype=np.int32)
        for i in range(ndim):
            mean = np.mean(data[:, i])
            for j in range(nlen):
                if data[j][i] > mean:
                    X[j][i] = 1
        return X, target
    
    
    if __name__ == '__main__':
        start_time = time.time()
    
        data, target = load_data()
        train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.3, random_state=2048, shuffle=True)
    
        ml = NaiveBayes(class_num=2, features_dim=30)
        ml.fit(train_x, train_y)
        y_pred = ml.predict(test_x)
        accuracy = ml.accuracy(test_y, y_pred)
        print("Accuracy is ", accuracy)
        print("Timeusage: %s" % (time_consume(start_time)))


结果如下：

    Accuracy is  0.947368421053
    Timeusage: 0:00:00


------

### 6. 参考

> 1. 李航 - 《统计学习方法》
> 2. [刘建平Pinard's Blog](http://www.cnblogs.com/pinard/p/6069267.html)
> 3. [scikit-learn](http://scikit-learn.org/)
> 4. [WenDesi's Github](https://github.com/WenDesi/lihang_book_algorithm)
