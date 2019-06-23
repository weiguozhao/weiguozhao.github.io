---
layout: post
title: sklearn-GridSearchCV & hyperopt & hyperopt-sklearn 调参
date: 2018-04-23 21:10 +0800
categories: 工具
tags:
- trick
- 技术
mathjax: true
copyright: true
---


**目录**

* sklearn-GridSearchCV
* hyperopt
* hyperopt-sklearn

---------

### sklearn-GridSearchCV

#### 常用参数

[sklearn.model_selection.GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

<table>
 <tr>
  <th>参数</th> <th>含义</th> <th>其他</th>
 </tr>
 <tr>
  <td>estimator</td> <td>所使用的模型</td> <td>假定这是scikit-learn中模型接口。该模型可以提供score方法或scoring参数</td>
 </tr>
 <tr>
  <td>param_grid</td> <td>dict或list</td> <td>带有参数名称作为键的字典,例如param_grid=param_test, param_test={'n_estimators': range(1, 6)}</td>
 </tr>
 <tr>
  <td>scoring</td> <td>评价标准，默认为None</td> <td>字符串，或是可调用对象，需要其函数形式如：score(estimator, X, y)；如果是None，则使用estimator的误差估计函数</td>
 </tr>
 <tr>
  <td>cv</td> <td>交叉验证参数，默认为None，使用三折交叉验证</td> <td>整数指定交叉验证折数，也可以是交叉验证生成器</td>
 </tr>
 <tr>
  <td>refit</td> <td>默认为True</td> <td>在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集</td>
 </tr>
 <tr>
  <td>iid</td> <td>默认为True</td> <td>默认为各个样本fold概率分布一致，误差估计为所有样本之和，而非各个fold的平均</td>
 </tr>
 <tr>
  <td>verbose</td> <td>默认为0</td> <td>日志冗长度。0：不输出训练过程；1：偶尔输出；>1：对每个子模型都输出</td>
 </tr>
  <tr>
  <td>n_jobs</td> <td>并行数，int类型</td> <td>-1：跟CPU核数一致； 1:默认值</td>
 </tr>
 <tr>
  <td>pre_dispatch</td> <td>指定总共分发的并行任务数</td> <td>当n_jobs大于1时，数据将在每个运行点进行复制，这可能导致OOM，而设置pre_dispatch参数，则可以预先划分总共的job数量，使数据最多被复制pre_dispatch次</td>
 </tr>
</table>

scikit-learn内置可用评价标准如下：[scikit-learn model_evalution](http://scikit-learn.org/stable/modules/model_evaluation.html)

![img](/posts_res/2018-04-23-tuneparameters/1.png)

----------

#### 常用方法

<table>
 <tr>
  <th>方法</th> <th>含义</th>
 </tr>
 <tr>
  <td>grid.fit()</td> <td>运行网格搜索</td>
 </tr>
 <tr>
  <td>grid.grid_scores_</td> <td>给出不同参数情况下的评价结果</td>
 </tr>
 <tr>
  <td>grid.best_params_</td> <td>已取得最佳结果的参数的组合</td>
 </tr>
 <tr>
  <td>grid.best_score_</td> <td>优化过程期间观察到的最好的评分</td>
 </tr>
</table>

---------

#### 代码

```python3
# coding:utf-8

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

def grid_search(params, X, y):
    svc = svm.SVC()
    grid = GridSearchCV(estimator=svc,
                        param_grid=param_test,
                        scoring="accuracy",
                        cv=3,
                        refit=True,
                        iid=True,
                        verbose=True)
    grid.fit(X, y)
    return grid

if __name__ == '__main__':
    iris = datasets.load_iris()
    param_test = {'kernel':['linear', 'rbf'], 'C':[1, 10]}
    estimator = grid_search(param_test, iris.data, iris.target)

    print("grid_scores_:", estimator.grid_scores_)
    print("best_params_", estimator.best_params_)
    print("best_score_", estimator.best_score_)
    # Fitting 3 folds for each of 4 candidates, totalling 12 fits
    # [Parallel(n_jobs=1)]: Done  12 out of  12 | elapsed:    0.0s finished
    # grid_scores_: [mean: 0.98000, std: 0.01602, params: {'kernel': 'linear', 'C': 1}, 
                     mean: 0.97333, std: 0.00897, params: {'kernel': 'rbf', 'C': 1}, 
                     mean: 0.97333, std: 0.03697, params: {'kernel': 'linear', 'C': 10}, 
                     mean: 0.98000, std: 0.01601, params: {'kernel': 'rbf', 'C': 10}]
    # best_params_ {'kernel': 'linear', 'C': 1}
    # best_score_ 0.98
```

---------

### hyperopt

[Hyperopt-Github](https://github.com/hyperopt/hyperopt)

Hyheropt四个重要的因素：
* 指定需要最小化的函数(the objective function to minimize)；
* 搜索的空间(the space over which to search)；
* 采样的数据集(trails database)(可选)；
* 搜索的算法(可选)

**目标函数**，指定最小化的函数，比如要最小化函数\\( q(x,y) = x^2 + y^2 \\)

**搜索的算法**，即hyperopt的fmin函数的algo参数的取值。
当前支持的算法有随机搜索(hyperopt.rand.suggest)，模拟退火(hyperopt.anneal.suggest)，TPE算法。

**搜索(参数)空间设置**，例如优化函数q，输入 fmin(q，space=hp.uniform(‘a’，0，1))。
hp.uniform函数的第一个参数是标签，每个超参数在参数空间内必须具有独一无二的标签。
hp.uniform指定了参数的分布。

其他的参数分布：
* hp.choice返回一个选项，选项可以是list或者tuple.options可以是嵌套的表达式，用于组成条件参数。 
* hp.pchoice(label,p_options)以一定的概率返回一个p_options的一个选项。这个选项使得函数在搜索过程中对每个选项的可能性不均匀。 
* hp.uniform(label,low,high)参数在low和high之间均匀分布。 
* hp.quniform(label,low,high,q),参数的取值是round(uniform(low,high)/q)*q，适用于那些离散的取值。 
* hp.loguniform(label,low,high)绘制exp(uniform(low,high)),变量的取值范围是[exp(low),exp(high)] 
* hp.randint(label,upper) 返回一个在[0,upper)前闭后开的区间内的随机整数。 

搜索空间可以含有list和dictionary。
```python3
from hyperopt import hp

list_space = [hp.uniform('a', 0, 1), hp.loguniform('b', 0, 1)]
tuple_space = (hp.uniform('a', 0, 1), hp.loguniform('b', 0, 1))
dict_space = {'a': hp.uniform('a', 0, 1), 'b': hp.loguniform('b', 0, 1)}
```

**sample函数在参数空间内采样**

```python3
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
list_space = [hp.uniform('a', 0, 1), hp.loguniform('b', 0, 1), hp.randint('c', 8)]
print(sample(list_space))
# (0.7802721043817558, 1.6616883461586371, array(4))
```

---------

#### 简单的例子

```python3
from hyperopt import fmin, hp, tpe
def objective(args):
    x, y = args
    return x**2 + y**2 + 1
space = [hp.randint('x', 5), hp.randint('y', 5)]
best = fmin(objective, space=space, algo=tpe.suggest, max_evals=100)
print(best)
# {'y': 0, 'x': 0}
```

--------

#### Perceptron鸢尾花例子

使用感知器判别鸢尾花，使用的学习率是0.1，迭代40次得到了一个测试集上正确率为82%的结果；使用hyperopt优化参数，将正确率提升到了91%。

```python3
# coding:utf-8

from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from hyperopt import fmin, tpe, hp, partial

# load data set
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# data standard
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# create perceptron model
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

# show result before tune parameters
y_pred = ppn.predict(X_test_std)
print("="*15, "before tune", "="*15)
print(accuracy_score(y_test, y_pred))

# =============================================================================================

# define object which hyperopt use
def percept(args):
    global X_train_std, y_train, y_test
    ppn = Perceptron(n_iter=int(args["n_iter"]), eta0=args["eta"] * 0.01, random_state=0)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    return -accuracy_score(y_test, y_pred)

# define search space
space = {"n_iter": hp.choice("n_iter", range(30, 50)), "eta": hp.uniform("eta", 0.05, 0.5)}
# define search algorithm
algo = partial(tpe.suggest, n_startup_jobs=10)

best = fmin(percept, space, algo=algo, max_evals=100)
print("="*15, "after tune", "="*15)
print(best)
print(percept(best))
```

**结果**

由于使用tpe搜索算法，每次搜索的结果都不一样，不稳定。

```text
=============== before tune ===============
0.822222222222
=============== after tune ===============
{'n_iter': 14, 'eta': 0.12949436553904228}
-0.911111111111
```

--------------

#### xgboost癌症例子

```python3
# coding:utf-8

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_breast_cancer
from hyperopt import fmin, tpe, hp, partial


def loadData():
    d = load_breast_cancer()
    data, target = d.data, d.target
    minmaxscaler = MinMaxScaler()
    data_scal = minmaxscaler.fit_transform(data)

    indices = np.random.permutation(target.shape[0])
    train_indices = indices[: int(target.shape[0] * 0.8)]
    test_indices = indices[int(target.shape[0] * 0.8):]
    train_x = data[train_indices]
    test_x = data[test_indices]
    train_y = target[train_indices]
    test_y = target[test_indices]

    return train_x, test_x, train_y, test_y

train_x, test_x, train_y, test_y = loadData()
print("train_x:", train_x.shape, " test_x:", test_x.shape, " train_y:", train_y.shape, " test_y:", test_y.shape)


# define object function which need to minimize
def GBM(argsDict):
    max_depth = argsDict["max_depth"] + 5
    n_estimators = argsDict['n_estimators'] * 5 + 50
    learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
    subsample = argsDict["subsample"] * 0.1 + 0.7
    min_child_weight = argsDict["min_child_weight"] + 1
    print("="*20)
    print("max_depth:" + str(max_depth))
    print("n_estimator:" + str(n_estimators))
    print("learning_rate:" + str(learning_rate))
    print("subsample:" + str(subsample))
    print("min_child_weight:" + str(min_child_weight))
    global train_x, train_y

    gbm = xgb.XGBClassifier(nthread=4,  # 进程数
                            max_depth=max_depth,  # 最大深度
                            n_estimators=n_estimators,  # 树的数量
                            learning_rate=learning_rate,  # 学习率
                            subsample=subsample,  # 采样数
                            min_child_weight=min_child_weight,  # 孩子数
                            max_delta_step=10,  # 10步不降则停止
                            objective="binary:logistic")
    metric = cross_val_score(gbm, train_x, train_y, cv=5, scoring="roc_auc").mean()
    print("cross_val_score:", metric, "\n")
    return -metric


# define search space
space = {"max_depth": hp.randint("max_depth", 15),
         "n_estimators": hp.randint("n_estimators", 10),  # [0,1,2,3,4,5] -> [50,]
         "learning_rate": hp.randint("learning_rate", 6),  # [0,1,2,3,4,5] -> 0.05,0.06
         "subsample": hp.randint("subsample", 4),  # [0,1,2,3] -> [0.7,0.8,0.9,1.0]
         "min_child_weight": hp.randint("min_child_weight", 5),  # [0,1,2,3,4] -> [1,2,3,4,5]
         }
# define search algorithm
algo = partial(tpe.suggest, n_startup_jobs=1)
# search best parameters
best = fmin(GBM, space, algo=algo, max_evals=4)

print("best param:", best) # output best parameters
print("best result:", GBM(best)) # output model result in best parameters
```

**结果**

```text
train_x: (455, 30)  test_x: (114, 30)  train_y: (455,)  test_y: (114,)
====================
max_depth:16
n_estimator:60
learning_rate:0.07
subsample:0.7999999999999999
min_child_weight:5
cross_val_score: 0.988504345257 

====================
max_depth:6
n_estimator:60
learning_rate:0.15000000000000002
subsample:0.7
min_child_weight:5
cross_val_score: 0.990053680824 

====================
max_depth:8
n_estimator:95
learning_rate:0.13
subsample:0.8999999999999999
min_child_weight:5
cross_val_score: 0.988304093567 

====================
max_depth:12
n_estimator:60
learning_rate:0.09
subsample:0.7
min_child_weight:3
cross_val_score: 0.990361013789 

best param: {'n_estimators': 2, 'max_depth': 7, 'subsample': 0, 'learning_rate': 2, 'min_child_weight': 2}
====================
max_depth:12
n_estimator:60
learning_rate:0.09
subsample:0.7
min_child_weight:3
cross_val_score: 0.990361013789 

best result: -0.990361013789
```


-----------

### hyperopt-sklearn

[hyperopt-sklearn-Github](https://github.com/hyperopt/hyperopt-sklearn/tree/master)

目前hyperopt-sklearn只支持部分Classifiers\Regressors\Preprocessing，具体请见上面hyperopt-sklearn的Github主页。

#### 安装

```shell
git clone git@github.com:hyperopt/hyperopt-sklearn.git
(cd hyperopt-sklearn && pip install -e .)
```

#### 使用模版

```python3
from hpsklearn import HyperoptEstimator, svc
from sklearn import svm

# Load Data
# ...

if use_hpsklearn:
    estim = HyperoptEstimator(classifier=svc('mySVC'))
else:
    estim = svm.SVC()

estim.fit(X_train, y_train)

print(estim.score(X_test, y_test))
```

#### 鸢尾花例子

```python3
# coding:utf-8

from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from sklearn.datasets import load_iris
from hyperopt import tpe
import numpy as np

# Download the data and split into training and test sets
iris = load_iris()
X = iris.data
y = iris.target
test_size = int(0.2 * len(y))
np.random.seed(13)
indices = np.random.permutation(len(X))
train_x = X[indices[:-test_size]]
train_y = y[indices[:-test_size]]
test_x = X[indices[-test_size:]]
test_y = y[indices[-test_size:]]

# Instantiate a HyperoptEstimator with the search space and number of evaluations
estim = HyperoptEstimator(classifier=any_classifier('my_clf'),
                          preprocessing=any_preprocessing('my_pre'),
                          algo=tpe.suggest,
                          max_evals=100,
                          trial_timeout=120)

# Search the hyperparameter space based on the data
estim.fit(train_x, train_x)

# Show the results
print(estim.score(test_x, test_y))
# 1.0
print(estim.best_model())
# {'learner': ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
#           max_depth=3, max_features='log2', max_leaf_nodes=None,
#           min_impurity_decrease=0.0, min_impurity_split=None,
#           min_samples_leaf=1, min_samples_split=2,
#           min_weight_fraction_leaf=0.0, n_estimators=13, n_jobs=1,
#           oob_score=False, random_state=1, verbose=False,
#           warm_start=False), 'preprocs': (), 'ex_preprocs': ()}
```

#### MNIST例子

```python3
# coding:utf-8

from hpsklearn import HyperoptEstimator, extra_trees
from sklearn.datasets import fetch_mldata
from hyperopt import tpe
import numpy as np

# Download the data and split into training and test sets
digits = fetch_mldata('MNIST original')

X = digits.data
y = digits.target
test_size = int(0.2 * len(y))
np.random.seed(13)
indices = np.random.permutation(len(X))
X_train = X[indices[:-test_size]]
y_train = y[indices[:-test_size]]
X_test = X[indices[-test_size:]]
y_test = y[indices[-test_size:]]

# Instantiate a HyperoptEstimator with the search space and number of evaluations
estim = HyperoptEstimator(classifier=extra_trees('my_clf'),
                          preprocessing=[],
                          algo=tpe.suggest,
                          max_evals=10,
                          trial_timeout=300)

# Search the hyperparameter space based on the data
estim.fit(X_train, y_train)

# Show the results
print(estim.score(X_test, y_test))
# 0.962785714286
print(estim.best_model())
# {'learner': ExtraTreesClassifier(bootstrap=True, class_weight=None, criterion='entropy',
#           max_depth=None, max_features=0.959202875857,
#           max_leaf_nodes=None, min_impurity_decrease=0.0,
#           min_impurity_split=None, min_samples_leaf=1,
#           min_samples_split=2, min_weight_fraction_leaf=0.0,
#           n_estimators=20, n_jobs=1, oob_score=False, random_state=3,
#           verbose=False, warm_start=False), 'preprocs': (), 'ex_preprocs': ()}
```


---------

### 参考

>
1. [sklearn-GridSearchCV,CV调节超参使用方法](https://blog.csdn.net/u012969412/article/details/72973055)
2. [python调参神器hyperopt-简略](http://www.cnblogs.com/gczr/p/7156270.html)
3. [python调参神器hyperopt-详细](https://blog.csdn.net/qq_34139222/article/details/60322995)

