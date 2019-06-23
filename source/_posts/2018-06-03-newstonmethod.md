---
layout: post
title: 牛顿法&拟牛顿法
date: 2018-06-03 12:10 +0800
categories: 优化算法
tags:
- 算法
- 技术
mathjax: true
copyright: true
---

目录

* 1.原始牛顿法
* 2.阻尼牛顿法
* 3.拟牛顿条件
* 4.DFP算法
* 5.BFGS算法
* 6.L-BFGS算法


------

考虑如下无约束的极小化问题

$$ min_x f(x) $$

其中 $ x=(x_1, x_2, ..., x_n)^T \in R^n $，由于本文不对收敛性进行讨论，因此对目标函数 $ f: R^n \rightarrow R $做一个较苛刻的假设。
这里假定$ f $为凸函数，且二阶连续可微，此外记极小化问题的解为 $ x^{\ast} $ .


---------

### 1.原始牛顿法

牛顿法的基本思想是：**在现有极小点估计值的附近对 $ f(x) $ 做二阶泰勒展开，进而找到极小点的下一个估计值。**

为简单起见，首先考虑$ n=1 $的情况，设$ x\_k $ 为当前的极小点估计值，则

$$ \phi (x) = f(x_k) + f'(x_k)(x-x_k) + \frac{1}{2}f''(x_k) (x-x_k)^2 $$

表示$ f(x) $ 在 $ x\_k $附近的二阶泰勒展开式（略去了关于$ x-x\_k $ 的高阶项）。由于求的是最值，由极值必要条件可知，$ \phi(x) $应该满足

$$ \phi ' (x) = 0 \quad \Rightarrow \quad f'(x_k) + f^{''}(x_k) (x - x_k) = 0 $$

从而求得：

\\[
x = x\_k - \frac{f'(x\_k)}{f^{''}(x\_k)}
\\]

于是，若给定初始值$ x\_0 $，则可以构造如下的迭代公式：

$$ x_{k+1} = x_k - \frac{f'(x_k)}{f^{''}(x_k)}, \quad k=0,1,... $$

产生序列$ \lbrace x\_k \rbrace $来逼近 $ f(x) $的极小点，在一定条件下，$ \lbrace x\_k \rbrace $可以收敛到$ f(x) $的极小点。

<br>

对于 $ n>1 $的情形，二阶泰勒展开式可以做推广，此时：

$$ \phi(X) = f(X_k) + \nabla f(X_k) \cdot (X - X_k) + \frac{1}{2} \cdot (X - X_k)^T \nabla^2 f(X_k) \cdot (X - X_k) $$

其中$ \nabla f $ 为 $ f $的梯度向量，$ \nabla^2 f $为 $ f $的海森矩阵（Hessian matrix），其定义分别为：

$$ \nabla f = \left[ \begin{matrix} \frac{\partial f}{\partial x_1} \\\ \frac{\partial f}{\partial x_2} \\\ \cdots \\\ \frac{\partial f}{\partial x_n}  \end{matrix} \right] $$

$$
\nabla^2 f = 
\left[
\begin{matrix} 
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} &  \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x^2_2} &  \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\\
 &  &  \ddots &  \\\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} &  \cdots & \frac{\partial^2 f}{\partial x_n^2} \\\
\end{matrix}
\right]
$$

注意，$ \nabla f $ 和 $ \nabla^2 f $ 中的元素均为关于$ x $ 的函数，以下分别将其简记为$g$和$H$，特别地，若$f$的混合偏导数可交换次序，则海森矩阵$H$为对称矩阵。
而$ \nabla f(X\_k) $ 和 $ \nabla^2 f(X\_k) $则表示将$X$取为$X\_k$后得到的实值向量和矩阵，以下分别将其简记为$ g\_k $和$H\_k$（这里字母$g$表示$gradient$，$H$表示$Hessian$）。

同样地，由于是求极小点，极值必要条件要求它为$ \phi (X)$的驻点，即

$$ \nabla \phi (X) = 0 \quad \Rightarrow \quad g_k + H_k \cdot (x - x_k) = 0 $$

进一步，若矩阵$ H\_k $非奇异，则可解得

$$ X = X_k - H_k^{-1} \cdot g_k $$

于是，给定初始值$ X\_0 $，则同样可以构造出迭代格式

$$ X_{k+1} = X_k - H_k^{-1} \cdot g_k, \quad k=0,1,2... $$

这就是原始的牛顿迭代法，其迭代格式中的搜索方向$ d\_k = - H\_k^{-1} \cdot  g\_k $称为**牛顿方向**，下面给出牛顿法的完整算法描述。

>
1. 给定初值$ x\_0 $和精度阈值$ \epsilon $，并令$ k=0$；
2. 计算$ g\_k $和$ H\_k $；
3. 若$ \|\| gk \|\| < \epsilon $，则停止迭代；否则确定搜索方向$ d\_k = -H\_k^{-1} \cdot g\_k $；
4. 计算新的迭代点 $ x\_{k+1} = x\_k + d\_k $；
5. 令$ k = k+1 $，转至第2步。 

原始牛顿法由于迭代公式中没有步长因子，而是定步长迭代，对于非二次型目标函数，有时会使函数值上升，即出现$ f(x\_{k+1} > f(x\_k) $的情况，
这表明原始牛顿法不能保证函数值稳定地下降，在严重的情况下甚至可能造成迭代点列$ \lbrace x\_k \rbrace $的发散而导致计算失败。


-------------

### 2.阻尼牛顿法

为消除原始牛顿法中的弊端，提出了“阻尼牛顿法”，阻尼牛顿法每次迭代的方向仍采用 $ d\_k $，但每次迭代需沿此方向做一维搜索(line search)，寻求最优的步长因子$ \lambda\_k $，即

$$ \lambda_k = \arg\min_{\lambda \in R} f(x_k + \lambda d_k) \tag{*} $$

下面给出一个阻尼牛顿法的完整算法描述：

>
1. 给定初值$ x\_0 $和精度阈值$ \epsilon $，并令$ k=0$；
2. 计算$ g\_k $和$ H\_k $；
3. 若$ \|\| gk \|\| < \epsilon $，则停止迭代；否则确定搜索方向$ d\_k = -H\_k^{-1} \cdot g\_k $；
4. 利用$(*)$式得到步长$ \lambda\_k $，并令 $ x\_{k+1} = x\_k + \lambda\_k d\_k $；
5. 令$ k = k+1 $，转至第2步。 


**小结**

牛顿法是梯度(下降)发的进一步发展，梯度法利用目标函数的一阶偏导数信息，以负梯度方向作为搜索方向，只考虑目标函数在迭代点的局部性质；
而牛顿法不仅使用目标函数的一阶偏导数，还进一步利用了目标函数的二阶偏导数，这样就考虑了梯度变化的趋势，因而能更全面地确定合适的搜索方向以加快收敛。

但是牛顿法主要存在以下两个缺点：
1. 对目标函数有较严格的要求，函数必须具有连续的一、二阶偏导数，海森矩阵必须正定；
2. 计算相当复杂，除需计算梯度以外，还需计算二阶偏导数矩阵和它的逆矩阵。计算量、存储量均很大，且均以维数$ n $的平方比增加，当$n$很大时这个问题尤为突出。


-----------

### 3.拟牛顿条件

上面说到了牛顿法的两个缺点，为了客服这两个问题，人们提出了拟牛顿法，这个方法的基本思想是：**不用二阶偏导数而构造出可以近似海森矩阵（或海森矩阵的逆）的正定对称阵，在“拟牛顿”的条件下优化目标函数。**

不同的构造方法就产生了不同的拟牛顿法，都是对牛顿法中用来计算搜索方向的海森矩阵（或海森矩阵的逆）作了近似计算罢了。

下面先推到拟牛顿条件，或者叫拟牛顿方程、割线条件，它是用来提供理论指导，指出了用来近似的矩阵应该满足的条件。

为了明确起见，下文中用$ B $ 表示对海森矩阵$H$本身的近似，而用$D$表示对海森矩阵的逆$ H^{-1} $的近似，即 $ B \approx H, D \approx H^{-1} $。

<br>

假设经过 $ k+1 $ 次迭代后得到 $ x_{k+1} $，此时将目标函数 $ f(x) $ 在 $ x_{k+1} $附近作泰勒展开，取二阶近似，得到：

$$ f(x) \approx f(x_{k+1} + \nabla f(x_{k+1}) \cdot (x - x_{k+1}) + \frac{1}{2} \cdot (x - x_{k+1}^T \cdot \nabla^2 f(x_{k+1}) \cdot (x - x_{k+1}) \tag{**} $$

在$ (**) $式两边同时作用一个梯度算子 $ \nabla $， 可得

$$ \nabla f(x) \approx \nabla f(x_{k+1}) + H_{k+1} \cdot (x - x_{k+1}) \tag{***} $$

在$ (***) $式中取$ x = x_k$，并整理可得：

$$ g_{k+1} - g_k = H_{k+1} \cdot ( x_{k+1} - x_k ) $$

引入新的记号：

$$ s_k = x_{k+1} - x_k, \quad y_k = g_{k+1} - g_k $$

则上式可以重新表示为：

$$ y_k \approx H_{k+1} \cdot s_k \qquad \Longleftrightarrow \qquad s_k \approx H_{k+1}^{-1} \cdot y_k $$

这就是所谓的**拟牛顿条件**，它对迭代过程中的海森矩阵$ H_{k+1}$作约束，因此，对$ H_{k+1} $做近似的$B_{k+1}$，以及对$ H_{k+1}^{-1} $做近似的$  D_{k+1}$ 可以将

$$ y_k = B_{k+1} \cdot s_k \qquad \Longleftrightarrow \qquad s_k = D_{k+1} \cdot y_k $$

作为指导。


-----------------

### 4.DFP算法

该算法的核心是：**通过迭代的方法，对$ H_{k+1}^{-1}$做近似，迭代格式为：**

$$ D_{k+1} = D_k + \Delta D_k, \quad k=0,1,2...$$

其中$ D_0$通常取为单位矩阵$I$，因此，关键是每一步的校正矩阵$ \Delta D_k$如何构造。

这里采用“待定法”，即首先将$\Delta D_k$待定成某种形式，然后结合拟牛顿条件来推到。这里我们将$ \Delta D_k $ 待定为：

$$ \Delta D_k = \alpha u u^T + \beta v v^T $$

其中$ \alpha, \beta$为待定系数，$ u, v \in R^n $为待定向量，从形式上来看，这种待定公式至少保证了矩阵$ \Delta D_k $的对称性（因为$uu^T$和$vv^T$都是对称矩阵）


$$
\begin{cases}
D_{k+1} = D_k + \Delta D_k \\
\Delta D_k = \alpha u u^T + \beta v v^T \\
s_k = D_{k+1} \cdot y_k \\
\end{cases}
\Longleftrightarrow
s_k = D_k y_k + \alpha u u^T y_k + \beta v v^T y_k
$$

调整上面的结论：

$$
\begin{equation}
\begin{aligned}
s_k
& = D_k y_k + u (\alpha u^T y_k) + v(\beta v^T y_k) \\
& = D_k y_k + (\alpha u^T y_k) u + (\beta v^T y_k) v
\end{aligned}
\end{equation}
\tag{+}
$$

括号中的$\alpha u^T y_k$和$\beta v^T y_k$是两个数，既然是数，我们不妨作如下简单赋值：

$$ \alpha u^T y_k = 1, \qquad \beta v^T y_k = -1  \tag{++}$$

即

$$ \alpha = \frac{1}{u^T y_k}, \qquad \beta = - \frac{1}{v^T y_k} $$

将$ (++) $式代入 $ (+) $，得到：

$$ u - v = s_k - D_k y_k $$

要使上式成立，不妨就直接取

$$ u = s_k, \qquad v = D_k y_k $$

再将$(21)$代入$(19)$，便得

$$ \alpha = \frac{1}{s_k^T y_k}, \qquad \beta = - \frac{1}{(D_k y_k)^T y_k} = - \frac{1}{y_k^T D_k y_k}$$

其中第二个等式用到了$D_k$的对称性。

至此，我们已经可以将校正矩阵$ \Delta D_k $构造出来了，

$$ \Delta D_k = \frac{s_k s_k^T}{s_k^T y_k} - \frac{D_k y_k y_k^T D_k}{y_k^T D_k y_k} $$

综上，我们给出DFP算法的一个完整算法描述：

>
1. 给定初值$x_0$和精度阈值$\epsilon$，并令$ D_0=I, \quad k=0$，
2. 确定搜索方向$ d_k = - D_k \cdot g_k $，
3. 利用$(*)$式得到步长$\lambda_k$，令$ s_k = \lambda_k d_k, \quad x_{k+1}=x_k + s_k $，
4. 若$ \|\| g_{k+1} \|\| < \epsilon$，则算法结束，
5. 计算 $ y_k = g_{k+1} - g_k $，
6. 计算$ D_{k+1} = D_k + \Delta D_k = D_k + \frac{s_k s_k^T}{s_k^T y_k} - \frac{D_k y_k y_k^T D_k}{y_k^T D_k y_k} $，
7. 令$ k = k+1$，转至第2步。 


-------------

### 5.BFGS算法

与DFP算法相比，BFGS算法性能更佳，目前它已成为求解无约束非线性优化问题最常用的方法之一。
BFGS算法已有较完善的局部收敛理论，对其全局收敛性的研究也取得了重要成果。

BFGS算法中核心公式的推到和DFP算法完全类似，只是互换了其中$s_k$和$y_k$的位置。
需要注意的是，BFGS算法是直接逼近海森矩阵，即$ B_k \approx H_k$，仍采用迭代方法，设迭代格式为：

$$ B_{k+1} = B_k + \Delta B_k, \qquad k=0,1,2... $$

其中的 $ B_0 $也常取为单位矩阵$I$，因此，关键是每一步的校正矩阵$\Delta B_k$如何构造。同样，将其待定为：

$$ \Delta B_k = \alpha u u^T + \beta v v^T $$

类比DFP算法，可得

$$ y_k = B_k s_k + (\alpha u^T s_k)u + (\beta v^T s_k) v $$

通过令 $ \alpha u^T s_k = 1,\qquad \beta v^T s_k = -1 $，以及 $ u = y_k, \qquad v = B_k s_k $，可以算得：

$$ \alpha = \frac{1}{y_k^T s_k}, \qquad \beta = - \frac{1}{s_k^T B_k s_k} $$

综上，便得到了如下校正矩阵$ \Delta B_k $的公式：

$$ \Delta B_k = \frac{y_k y_k^T}{y_k^T s_k} - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k} $$

先讲把矩阵$ \Delta B_k $ 和 $ \Delta D_k $拿出来对比一下，是不是除了将$D$换成$B$外，其他只是将$s_k$和$y_k$互调了一下位置呢？

最后，我们给出BFGS算法的一个完整算法描述：

>
1. 给定初值$x_0$和精度阈值$\epsilon$，并令$ B_0=I, \quad k=0$，
2. 确定搜索方向$ d_k = - B_k^{-1} \cdot g_k $，
3. 利用$(*)$式得到步长$\lambda_k$，令$ s_k = \lambda_k d_k, \quad x_{k+1}=x_k + s_k $，
4. 若$ \|\| g_{k+1} \|\| < \epsilon$，则算法结束，
5. 计算 $ y_k = g_{k+1} - g_k $，
6. 计算$ B_{k+1} = B_k + \Delta B_k = B_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k} $，
7. 令$ k = k+1$，转至第2步。 


上面算法中的第2步通常是通过求解线性代数方程组$ B_k d_k = -g_k $来进行，然而更一般的做法是通过对第6步中的递推关系，
应用$Sherman-Morrison$公式，直接给出$B_{k+1}^{-1}$和$B_k^{-1}$之间的关系：

$$ B_{k+1}^{-1} = \left( I - \frac{s_k y_k^T}{y_k^T s_k} \right) B_k^{-1} \left( I - \frac{y_k s_k^T}{y_k^T s_k} \right) + \frac{s_k s_k^T}{y_k^T s_k} $$

利用上式，我们很容易将上述BFGS算法进行改进，为了避免出现矩阵求逆符号，我们统一将$B_i^{-1}$换用$D_i$（这样做仅仅只是符号上看起来舒服起见）。
这样，整个算法中不再需要求解线性代数方程组，由矩阵-向量运算就可以完成了。

改进后的BFGS算法如下：

>
1. 给定初值$x_0$和精度阈值$\epsilon$，并令$ D_0=I, \quad k=0$，
2. 确定搜索方向$ d_k = - D_k \cdot g_k $，
3. 利用$(*)$式得到步长$\lambda_k$，令$ s_k = \lambda_k d_k, \quad x_{k+1}=x_k + s_k $，
4. 若$ \|\| g_{k+1} \|\| < \epsilon$，则算法结束，
5. 计算 $ y_k = g_{k+1} - g_k $，
6. 计算$ D_{k+1} = \left( I - \frac{s_k y_k^T}{y_k^T s_k} \right) D_k \left( I - \frac{y_k s_k^T}{y_k^T s_k} \right) + \frac{s_k s_k^T}{y_k^T s_k} $，
7. 令$ k = k+1$，转至第2步。 

至此，关于DFP算法和BFGS算法的介绍就完成了。

最后再补充一下一维搜索(line search)的问题，在之前几个算法描述中，为简单起见，均采用了$(*)$时来计算步长$ \lambda_k $，其实这是一种精确搜索；
实际应用中，还有像Wolfe型搜索、Armijo搜索以及满足Goldstein条件的非精确搜索。这里我们以Wolfe搜索为例，简单介绍。

设$ \tilde{\beta} \in (0, \frac{1}{2}), \beta \in (\tilde{\beta}, 1)$，所谓的Wolfe搜索是指$ \lambda_k $满足如下**Wolfe条件**

$$
\begin{cases}
f(x_k + \lambda_k d_k) \qquad \leq f(x_k) + \tilde{\beta} \lambda_k d_k^T g_k; \\
d_k^T g(x_k + \lambda_k d_k) \geq \beta d_k^T g_k
\end{cases}
$$

带非精确搜索的拟牛顿法的研究是从1976年Powell的工作开始的，他证明了带Wolfe搜索的BFGS算法的全局收敛性和超线性收敛性。


------------------

### 6.L-BFGS算法

在BFGS算法中，需要用到一个$NxN$的矩阵$D_k$，当 N 很大时，存储这个矩阵将变得很耗计算机资源，考虑 N 为 10万 的情形，且 用double型（8字节）来存储$D_k$，需要多大的内存呢？来计算一下：

$$ \frac{N阶矩阵的字节数}{1GB的字节数} \qquad = \qquad \frac{10^5 \times 10^5 \times 8}{2^{10} \times 2^{10} \times 2^{10}} \qquad = \qquad 74.5(GB) $$

即使考虑到矩阵$D_k$的对称性，内存还可以降一半，但是这对于一般的服务器仍然是难以承受的。况且我们还只是假定了 N=10W 的情况，在实际机器学习问题中，这只能算是中小规模。

L-BFGS(Limited-memory BFGS)算法就是通过对BFGS算法进行改造，从而减少其迭代过程中所需内存开销的算法。
它对BFGS算法进行了近似，其基本思想是：**不再存储完整的矩阵$D_k$，而是存储计算过程中的向量序列$\{s_i\}，\{y_i\}$，需要矩阵$D_k$时，利用向量序列$\{s_i\}，\{y_i\}$的计算来代替。**
而且，向量序列$\{s_i\}，\{y_i\}$也不是所有的都存，而是固定存最新的$m$个（参数$m$可由用户根据机器的内存自行指定），每次计算$D_k$时，只利用最新的$m$个$\{s_i\}$和$m$个$\{y_i\}$，
显然，这样我们将存储由原来的$O(n^2)$降到了$O(mN)$。

<br>

接下来，讨论L-BFGS算法的具体实现过程，我们的出发点是改进的BFGS算法中的第6步中的迭代式：

$$ D_{k+1} = \left( I - \frac{s_k y_k^T}{y_k^T s_k} \right) D_k \left( I - \frac{y_k s_k^T}{y_k^T s_k} \right) + \frac{s_k s_k^T}{y_k^T s_k} $$

若记$ \rho_k = \frac{1}{y_k^T s_k}，V_k = I - \rho_k y_k s_k^T $，则上式可写成

$$ D_{k+1} = V_k^T D_k V_k + \rho_k s_k s_k^T \tag{6.*} $$

如果给定初始矩阵$D_0$（通常为正定的对角矩阵，如$D_0 = I$），则可利用$(6.*)$式，依次可得：

$$
\begin{equation}
\begin{aligned}
D_1 &= V_0^T D_0 V_0 + \rho_0 s_0 s_0^T; \\
D_2 &= V_1^T D_1 V_1 + \rho_1 s_1 s_1^T \\
&= v_1^T (V_0^T D_0 V_0 + \rho_0 s_0 s_0^T) V_1 + \rho_1 s_1 s_1^T \\
&= V_1^T V_0^T D_0 V_0 V_1 + V_1^T \rho_0 s_0 s_0^T V_1 + \rho_1 s_1 s_1^T \\
D_3 &= V_2^T D_2 V_2 + \rho_2 s_2 s_2^T \\
&= V_2^T (V_1^T V_0^T D_0 V_0 V_1 + V_1^T \rho_0 s_0 s_0^T V_1 + \rho_1 s_1 s_1^T) V_2 + \rho_2 s_2 s_2^T \\
&= v_2^T V_1^T V_0^T D_0 V_0 V_1 V_2 + V_2^T V_1^T \rho_0 s_0 s_0^T V_1 V_2 + V_2^T \rho_1 s_1 s_1^T V_2 + \rho_2 s_2 s_2^T \\
\cdots
\end{aligned}
\end{equation}
$$

一般地，我们有：

$$
\begin{equation}
\begin{aligned}
D_{k+1} = 
&\ (V_k^T V_{k-1}^T \cdots V_1^T V_0^T) D_0 (V_0 V_1 \cdots V_{k-1} V_k) \\
&+ (V_k^T V_{k-1}^T \cdots V_2^T V_1^T) (\rho_0 s_0 s_0^T) (V_1 V_2 \cdots V_{k-1} V_k) \\
&+ (V_k^T V_{k-1}^T \cdots V_3^T V_2^T) (\rho_1 s_1 s_1^T) (V_2 V_3 \cdots V_{k-1} V_k) \\
&+ \cdots \\
&+ (V_k^T V_{k-1}^T) (\rho_{k-2} s_{k-2} s_{k-2}^T) (V_{k-1} V_k) \\
&+ V_k^T (\rho_{k-1} s_{k-1} s_{k-1}^T) V_k \\
&+ \rho_k s_k s_k^T
\end{aligned}
\end{equation} \tag{6.**}
$$

由上式可以看到，计算$D_{k+1}$需要用到$ \lbrace s_i, y_i \rbrace_{i=0}^k $，因此，若从$ s_0, y_0 $开始连续存储 $ m $ 组的话，只能存储到$ s_{m-1}, y_{m-1} $，
亦即只能依次计算 $ D_1, D_2, \cdots $，直到$ D_m $，那$ D_{m+1}, D_{m+2} $该如何计算呢？

自然地，如果一定要丢弃一些向量，那么肯定优先考虑那些最早生成的向量，具体来说，计算$D_{m+1}$时，我们保存$ \lbrace s_i, y_i \rbrace_{i=1}^m $，丢掉$ \lbrace s_0, y_0 \rbrace;$ 
计算$ D_{m+2} $时，我们保存$ \lbrace s_i, y_i\rbrace_{i=2}^{m+1} $，丢掉了$ \lbrace s_i, y_i \rbrace_{i=0}^1; \cdots $

但是舍弃掉一些向量后，就只能近似计算了，当$k+1 > m$时，仿照$(6.**)$式，可以构造近似计算公式：

$$
\begin{equation}
\begin{aligned}
D_{k+1} = 
&\ (V_k^T V_{k-1}^T \cdots V_{k-m+2}^T V_{k-m+1}^T) D_0 (V_{k-m+1} V_{k-m+2} \cdots V_{k-1} V_k) \\
&+ (V_k^T V_{k-1}^T \cdots V_{k-m+3}^T V_{k-m+2}^T) (\rho_0 s_0 s_0^T) (V_{k-m+2} V_{k-m+3} \cdots V_{k-1} V_k) \\
&+ (V_k^T V_{k-1}^T \cdots V_{k-m+4}^T V_{k-m+3}^T) (\rho_1 s_1 s_1^T) (V_{k-m+3} V_{k-m+4} \cdots V_{k-1} V_k) \\
&+ \cdots \\
&+ (V_k^T V_{k-1}^T) (\rho_{k-2} s_{k-2} s_{k-2}^T) (V_{k-1} V_k) \\
&+ V_k^T (\rho_{k-1} s_{k-1} s_{k-1}^T) V_k \\
&+ \rho_k s_k s_k^T
\end{aligned}
\end{equation} \tag{6.***}
$$

$ (6.\*\*) $ 和 $ (6.\*\*\*) $ 被称为$ special BFGS matrices $，若引入 $ \hat{m} = min\lbrace k, m-1 \rbrace $，则还可以将两式合并成

$$
\begin{equation}
\begin{aligned}
D_{k+1} = 
&\ (V_k^T V_{k-1}^T \cdots V_{k-\hat{m}+1}^T V_{k-\hat{m}}^T) D_0 (V_{k-\hat{m}} V_{k-\hat{m}+1} \cdots V_{k-1} V_k) \\
&+ (V_k^T V_{k-1}^T \cdots V_{k-\hat{m}+2}^T V_{k-\hat{m+1}}^T) (\rho_0 s_0 s_0^T) (V_{k-\hat{m}+1} V_{k-\hat{m}+2} \cdots V_{k-1} V_k) \\
&+ (V_k^T V_{k-1}^T \cdots V_{k-\hat{m}+3}^T V_{k-\hat{m+2}}^T) (\rho_1 s_1 s_1^T) (V_{k-\hat{m}+2} V_{k-\hat{m}+3} \cdots V_{k-1} V_k) \\
&+ \cdots \\
&+ (V_k^T V_{k-1}^T) (\rho_{k-2} s_{k-2} s_{k-2}^T) (V_{k-1} V_k) \\
&+ V_k^T (\rho_{k-1} s_{k-1} s_{k-1}^T) V_k \\
&+ \rho_{k} s_{k} s_{k}^T
\end{aligned}
\end{equation}
$$

事实上，由BFGS算法流程易知，$D_k$的作用仅用来计算$ D_k g_k $ 获取搜索方向，因此，若能利用上式设计出一种计算$ D_k g_k $的快速算法，则大功告成。

> **($D_k \cdot g_k$的快速算法)**
> 
Step 1 初始化
$$
\delta = 
\begin{cases}
0, \qquad k \leq m \\
k-m, \qquad k > m
\end{cases};
\qquad
L = 
\begin{cases}
k, \qquad k \leq m \\
m, \qquad k > m
\end{cases};
\qquad
q_L = g_k
$$
>
Step 2 后向循环
$$
FOR \quad i=L-1, L-2, \cdots, 1, 0 \quad DO \\
\{ \\
\qquad j=i+\delta; \\
\qquad \alpha_i = \rho_j s_j^T q_{i+1}; // \alpha_i 需要保存下来，前向循环要用！！ \\
\qquad q_i = q_{i+1} - \alpha_i y_j \\
\}
$$
>
Step 3 前向循环
$$
r_0 = D_0 \cdot q_0; \\
FOR \quad i=0,1,\cdots,L-2,L-1 \quad DO \\
\{ \\
\qquad j = i+\delta; \\
\qquad \beta_j = \rho_j y_j^T r_i; \\
\qquad r_{i+1} = r_i + (\alpha_i - \beta_i) s_j \\
\}
$$


最后算出的$r_L$即为$H_k \cdot g_k $的值。


---------

>
[牛顿法与拟牛顿法学习笔记](https://blog.csdn.net/itplus/article/details/21896453)

