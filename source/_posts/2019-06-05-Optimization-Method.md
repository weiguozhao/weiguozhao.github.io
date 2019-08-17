---
title: Optimization Method
tags:
  - 优化算法
  - 深度学习
mathjax: true
copyright: true
date: 2019-06-05 18:05:24
categories: 优化算法
---

对一些优化算法进行总结：

## 1.1 Batch Gradient Descent

$$
\theta_t  = \theta_{t-1} - \eta \cdot \nabla_{\theta} J(\theta )
$$

```python
for i in range(nb_epochs):
    theta_grad = evaluate_gradient(loss_function, data, theta)
    theta = theta - learning_rate * theta_grad
```

每次按照全局梯度的负方向前进，步数由learning_rate控制。批量梯度下降(GD): 1. 可以保证精度; 2. 防止过拟合要加正则项。在线梯度下降(OGD): 1. 着重处理稀疏性。



## 1.2 Stochastic Gradient Descent

$$
\theta_t = \theta_{t-1} - \eta \cdot \nabla_{\theta} J(\theta; x^{(i)}; y^{(i)})
$$

```python
for i in range(nb_epochs):
    np.random.shuffle(data)
    for example in data:
        theta_grad = evaluate_gradient(loss_function, example, theta)
        theta = theta - learning_rate * theta_grad
```

每次随机按照选定的样本梯度负方向前进，步数由learning_rate控制。


## 1.3 Mini-Bach Gradient Descent

$$
\theta_t = \theta_{t-1} - \eta \cdot \nabla_{\theta} J(\theta; x^{(i:i+n)}; y^{(i:i+n)})
$$

```python
for i in range(nb_epoches):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    theta_grad = evaluate_gradient(loss_function, batch, theta)
    theta = theta - learning_rate * theta_grad
```

每次选取小的batch，按照batch的梯度负方向前进，步数由learning_rate控制。


**上述算法存在的问题：**
- 很难调整出一个合适的learning_rate
- learning_rate的变化规则往往是预定义的，很难适应不同的数据
- 所有的特征共享相同的learning_rate
- 局部最有解的问题


---------------------------------------------------------------

## 2.1 Gradient Descent with Momentum

$$
\begin{aligned}
v_t &= \beta v_{t-1} + (1 - \beta) \nabla_{\theta} J(\theta) \\
\theta_t &= \theta_{t-1} - \alpha v_t
\end{aligned}
$$

```python
for i in range(nb_epoches):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    theta_grad = evaluate_gradient(loss_function, batch, theta)
    v_t = beta * v_t + (1 - beta) * theta_grad
    theta = theta - alpha * v_t
```

- theta_grad $\Longleftrightarrow \nabla_{\theta} J(\theta)$
- v_t $\Longleftrightarrow v_t$
- beta $\Longleftrightarrow \beta$
- alpha $\Longleftrightarrow \alpha$

对梯度进行指数加权平均，这样使当前梯度不仅与当前方向有关，还与之前的方向有关，这样处理让梯度前进方向更加平滑，减少振荡，能够更快地到达最小值处。经验值 $\beta=0.9$。


## 2.2 Nesterov Accelerated Gradient

$$
\begin{aligned}
v_t &= \beta v_{t-1} + \nabla_{\theta} J(\theta_{t-1} - \alpha \beta v_{t-1}) \\
\theta_t &= \theta_{t-1} - \alpha v_t
\end{aligned}
\tag{2.2-1}
$$

$$\Updownarrow$$

$$
\begin{aligned}
v_t &= \beta v_{t-1} + \nabla J(\theta_{t-1}) + \beta [ \nabla (\theta_{t-1}) - \nabla (\theta_{t-2})] \\
\theta_t &= \theta_{t-1} - \alpha v_t
\end{aligned}
\tag{2.2-2}
$$

```python
for i in range(nb_epoches):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    theta_grad = evaluate_gradient(loss_function, batch, theta)
    v_t = beta * v_t + theta_grad
    theta = theta - alpha * v_t
```

> 如果这次的梯度比上次的梯度变大了，那么有理由相信它会继续变大下去，那就把预计要增大的部分提前加进来；如果相比上次变小了，也是类似的情况。这个多加上去的项就是在近似目标函数的二阶导，所以NAG本质上是多考虑了目标函数的二阶导信息，因此可以加速收敛了！所谓“往前看”的说法，在牛顿法这样的二阶方法中也是经常提到的，比喻起来是说“往前看”，数学本质上则是利用了目标函数的二阶导信息。
> 在原始形式中，Nesterov Accelerated Gradient（NAG）算法相对于Momentum的改进在于，以“向前看”看到的梯度而不是当前位置梯度去更新。经过变换之后的等效形式中，NAG算法相对于Momentum多了一个本次梯度相对上次梯度的变化量，这个变化量本质上是对目标函数二阶导的近似。由于利用了二阶导的信息，NAG算法才会比Momentum具有更快的收敛速度。

**$Prof: (2.2-1) \Rightarrow (2.2-2)$**

$$
\begin{aligned}
\theta_t - \alpha \beta v_t 
&= \theta_{t-1} - \alpha v_t - \alpha \beta v_t \\
&= \theta_{t-1} - \alpha (\beta + 1) v_t \\
&= \theta_{t-1} - \alpha (\beta + 1) [ \beta v_{t-1} + \nabla_{\theta} J(\theta_{t-1} - 
\alpha \beta v_{t-1}) ] \\
&= \theta_{t-1} - \alpha \beta v_{t-1} - \alpha [ \beta^2 v_{t-1} + (\beta + 1) \nabla_{\theta} J(\theta_{t-1} - 
\alpha \beta v_{t-1}) ]
\end{aligned}
$$

定义alias：
$$
\begin{aligned}
\hat{\theta_t} &= \theta_{t} - \alpha \beta v_{t} \rightarrow \hat{\theta_{t-1}} = \theta_{t-1} - \alpha \beta v_{t-1} \\
\hat{v_t} &= \beta^2 v_{t-1} + (\beta + 1) \nabla_{\theta} J(\theta_{t-1} - \alpha \beta v_{t-1}) \\ 
&= \beta^2 v_{t-1} + (\beta + 1) \nabla_{\theta} J(\hat{ \theta_{t-1} })
\end{aligned}
$$

则：
$$
\begin{aligned}
\hat{\theta_t} &= \theta_t - \alpha \beta v_t = \hat{\theta_t} - \alpha \hat{v_t}
\end{aligned}
$$

$$
\begin{aligned}
\hat{v_t} 
&= \beta^2 v_{t-1} + (\beta + 1) \nabla_{\theta} J(\hat{ \theta_{t-1} }) \\
&= (\beta + 1) \nabla_{\theta} J(\hat{ \theta_{t-1} }) + \beta^2 (\beta v_{t-2} + \nabla_{\theta} J(\theta_{t-2} - \alpha \beta v_{t-2})) \\
&= (\beta + 1) \nabla_{\theta} J(\hat{ \theta_{t-1} }) + \beta^2 ( \beta v_{t-2} + \nabla_{\theta} J(\hat{ \theta_{t-2} }) ) \\
&= (\beta + 1) \nabla_{\theta} J(\hat{ \theta_{t-1} }) + \beta^2 \nabla_{\theta} J(\hat{ \theta_{t-2} }) + \beta^3 v_{t-2} \\
&= (\beta + 1) \nabla_{\theta} J(\hat{ \theta_{t-1} }) + \beta^2 \nabla_{\theta} J(\hat{ \theta_{t-2} }) + \beta^3 \nabla_{\theta} J(\hat{ \theta_{t-3} }) + \beta^4 v_{t-3} \\
&= \cdots \\
&= \cdots \\
&= (\beta + 1) \nabla_{\theta} J(\hat{ \theta_{t-1} }) + \beta^2 \nabla_{\theta} J(\hat{ \theta_{t-2} }) + \beta^3 \nabla_{\theta} J(\hat{ \theta_{t-3} }) + \beta^4 \nabla_{\theta} J(\hat{ \theta_{t-4} }) + \beta^5 \nabla_{\theta} J(\hat{ \theta_{t-5} }) + \cdots \\
\\
\beta \hat{v_{t-1}}
&= \beta (\beta + 1) \nabla_{\theta} J(\hat{ \theta_{t-2} }) + \beta^3 \nabla_{\theta} J(\hat{ \theta_{t-3} }) + \beta^4 \nabla_{\theta} J(\hat{ \theta_{t-4} }) + \beta^5 \nabla_{\theta} J(\hat{ \theta_{t-5} }) + \beta^6 \nabla_{\theta} J(\hat{ \theta_{t-6} }) + \cdots \\
\\
\hat{v_t} - \beta \hat{v_{t-1}}
&= (\beta + 1) \nabla_{\theta} J(\hat{ \theta_{t-1} }) - \beta \nabla_{\theta} J(\hat{ \theta_{t-2} }) \\
&= \nabla_{\theta} J(\hat{ \theta_{t-1} }) + \beta [ \nabla_{\theta} J(\hat{ \theta_{t-1} }) - \nabla_{\theta} J(\hat{ \theta_{t-2} })]
\end{aligned}
$$

最终得到：
$$
\begin{aligned}
\hat{v_t} &= \beta \hat{v_{t-1}} + \nabla_{\theta} J(\hat{ \theta_{t-1} }) + \beta [ \nabla_{\theta} J(\hat{ \theta_{t-1} }) - \nabla_{\theta} J(\hat{ \theta_{t-2} })] \\
\hat{\theta_t} &= \hat{\theta_t} - \alpha \hat{v_t}
\end{aligned}
$$


## 2.3 AdaGrad

定义alias:
$$
g_{t,i} = \nabla_{\theta_t} J(\theta_{t, i})
$$

$$
\begin{aligned}
\theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{ \sqrt{G_{t,ii} + \epsilon}}  \cdot g_{t, i} \quad \Rightarrow \quad
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_{t\cdot}
\end{aligned}
$$

其中$G_t \in \mathbb{R}^{d \times d}$是一个对角矩阵，其中当$t$步时的参数为$\theta_i$，则$G_{t,ii}$就表示$t$步时，第$i$个参数的历史梯度平方和。$\epsilon$是平滑因子，为避免分母为$0$通常设置$\epsilon=1e-8$，当没有平方根操作的时候效果会很差。

$Adagrad$ 最主要的贡献是不再需要人为调整learning_rate，大多数应用中只需要设置为默认$0.01$即可。由于每个增加的梯度平方都是非负数，$G_t$在训练中会不断增长，会导致learning_rate最终变得很小，此时模型无法学习到新的knowledge。


----------------------------------------


## 3.1 AdaDelta


## 3.2 RMSprop


## 3.3 Adam


## 3.4 AdaMax


## 3.5 Nadam



-----------------------

## 4.1 简单截断法

- 将不满足阈值的系数设置为0

## 4.2 Truncated Gradient

- 简单截断法的改进

## 4.3 FOBOS

- 微调标准梯度下降的结果

## 4.4 RDA

- 历史梯度加权平均
- 正则项对特征稀疏化
- 严格递增序列

## 4.5 FTRL

- 确保新的权重和历史权重不偏离太远
- L1正则稀疏性约束


------------

> - [An overview of gradient descent optimization algorithms]()
> - [在线最优化求解(Online Optimization)-冯扬]()
> - [比Momentum更快：揭开Nesterov Accelerated Gradient的真面目](https://zhuanlan.zhihu.com/p/22810533)
