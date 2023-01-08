# HW3

PB20111689 蓝俊玮

## 3.2

$$
y=\frac{1}{1+e^{-(\pmb{w}^T\pmb{x}+b)}}\quad(3.18)\\
l(\pmb{\beta})=\sum_\limits{i=1}^{m}(-y_i\pmb{\beta}^T\hat{\pmb{x_i}}+\ln(1+e^{\pmb{\beta}^T\hat{\pmb{x_i}}}))\quad(3.27)
$$

 那么可以计算 
$$
\frac{\part y}{\part \pmb{w}}=\frac{\pmb{x}e^{-(\pmb{w}^T\pmb{x}+b)}}{{(1+e^{-(\pmb{w}^T\pmb{x}+b)})}^2}\\
\frac{\part^2y}{\part\pmb{w}\part{\pmb{w}^T}}=\pmb{x}\pmb{x}^Te^{-(\pmb{w}^T\pmb{x}+b)}\frac{-1+e^{-(\pmb{w}^T\pmb{x}+b)}}{(1+e^{-(\pmb{w}^T\pmb{x}+b)})^3}
$$
则可以看出 $\frac{\part^2y}{\part\pmb{w}^2}$ 在 $e^{-(\pmb{w}^T\pmb{x}+b)}=1$ 时取 0，则其有正有负，所以它的海森矩阵不是半正定的，因此它不是凸函数。

根据课本式子 
$$
\frac{\part^2l(\pmb{\beta})}{\part{\pmb{\beta}\part{\pmb{\beta}}^T}}=\sum_\limits{i=1}^{m}\hat{\pmb{x}_i}\hat{\pmb{x}_i}^Tp_1(\hat{\pmb{x}_i};\pmb{\beta})(1-p_1(\hat{\pmb{x}_i};\pmb{\beta}))\quad(3.31)
$$
因为概率 $p_1(\hat{\pmb{x}_i};\pmb{\beta})\in[0,1],\quad\frac{\part^2l(\pmb{\beta})}{\part{\pmb{\beta}\part{\pmb{\beta}}^T}}\ge0$，所以式子 (3.31) 是半正定的，所以 $l(\pmb{\beta})$ 是凸函数。

## 3.7

> 令码长为 9，类别数为 4，试给出海明距离意义下理论最优的 ECOC 二元码并证明之

这里的分歧我认为是这个定义不够准确，网上绝大多数的定义为正反码两个意义下的海明距离都要最优。

根据助教在 gitee 所言。如果只考虑正码之间两两海明距离最小的情况，而不考虑反码下的话，则有：

|      |  f1  |  f2  |  f3  |  f4  |  f5  |  f6  |  f7  |  f8  |  f9  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  c1  |  1   |  1   |  1   |  1   |  1   |  1   |  1   |  1   |  1   |
|  c2  |  1   |  1   |  1   |  1   |  1   |  1   |  -1  |  -1  |  -1  |
|  c3  |  1   |  1   |  1   |  -1  |  -1  |  -1  |  1   |  1   |  1   |
|  c4  |  -1  |  -1  |  -1  |  1   |  1   |  1   |  1   |  1   |  1   |

则可以取到最小海明距离为 6 的 ECOC 二元码。

但如果考虑正反码意义下的两两之间海明距离最小的情况，则根据 Exhausted Code 可以给出

|      |  f1  |  f2  |  f3  |  f4  |  f5  |  f6  |  f7  |  f8  |  f9  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  c1  |  1   |  1   |  1   |  1   |  1   |  1   |  1   |  1   |  1   |
|  c2  |  -1  |  -1  |  -1  |  -1  |  1   |  1   |  1   |  1   |  -1  |
|  c3  |  -1  |  -1  |  1   |  1   |  -1  |  -1  |  1   |  1   |  1   |
|  c4  |  -1  |  1   |  -1  |  1   |  -1  |  1   |  -1  |  1   |  1   |

则可以取到最小海明距离为 4 的 ECOC 二元码。

## 补充题

> 在 LDA 多分类情形下，试计算类间散度矩阵 $S_b$ 的秩并证明

在 LDA 多分类情形下，类间散度矩阵 $S_b=\sum\limits_{i=1}^{N}m_i(\pmb{\mu}_i-\pmb{\mu})(\pmb{\mu}_i-\pmb{\mu})^T$。因为有引理 $\rank(AB)\le \min{(\rank{A},\rank{B})}$， 所以可以知道 $(\pmb{\mu}_i-\pmb{\mu})(\pmb{\mu}_i-\pmb{\mu})^T$ 的秩为 $1$。同时因为 $\pmb{\mu}=\frac{1}{m}\sum\limits_{i=1}^{N}m_i\pmb{\mu}_i$，所以对 $S_b$ 中的所有矩阵，取 $\forall c_i=\frac{1}{m}$，将每一个矩阵求和乘上系数后有 
$$
\sum\limits_{i=1}^{N}c_im_i(\pmb{\mu}_i-\pmb{\mu})(\pmb{\mu}_i-\pmb{\mu})^T=\frac{1}{m}\sum\limits_{i=1}^{N}m_i(\pmb{\mu}_i-\pmb{\mu})(\pmb{\mu}_i-\pmb{\mu})^T=\\
(\frac{m\pmb{\mu}}{m}-\frac{1}{m}\sum\limits_{i=1}^{N}m_i\pmb{\mu})(\pmb{\mu}_i-\pmb{\mu})^T=(\pmb{\mu}-\pmb{\mu})(\pmb{\mu}_i-\pmb{\mu})^T=0
$$
使得 $S_b$ 中只有 $N-1$ 个无关的矩阵，所以 $S_b$ 的秩至多为 $N-1$，即有 $\rank{(S_b)}\le N-1$。

> 给出公式 3.45 的推导证明

此处也固定分母为 1，则可得到该优化问题
$$
\min\limits_{w}\quad-\tr(W^TS_bW)\\
s.t.\quad \tr(W^TS_wW)=1
$$
由拉格朗日乘子法，则
$$
L(W,\lambda)=-\tr(W^TS_bW)+\lambda\tr(W^TS_wW)-\lambda
$$
则求导可得
$$
\frac{\part L(W,\lambda)}{\part W}=-(S_b+S_b^T)W+\lambda(S_w+S_w^T)W=0\\
\frac{\part L(W,\lambda)}{\part\lambda}=\tr(W^TS_wW)-1=0\\
-2S_bW+2\lambda S_wW=0\\
S_bW=\lambda S_wW
$$

> 证明 $X(X^TX)^{-1}X^T$ 是投影矩阵，并对线性回归模型从投影角度解释

$X(X^TX)^{-1}X^T\cdot X(X^TX)^{-1}X^T=X(X^TX)^{-1}(X^TX)(X^TX)^{-1}X^T=X(X^TX)^{-1}X^T$

所以 $X(X^TX)^{-1}X^T$ 是幂等矩阵，所以它是投影矩阵。

对于线性回归模型有 $f(\hat{\pmb{x}_i})=\pmb{x}_i\hat{\pmb{w}}$，通过求其偏导使得 $\nabla_{\hat{\pmb{w}}}E(\hat{\pmb{w}})=0$ 得到 $\hat{\pmb{w}}=(X^TX)^{-1}X^Ty$，则带入可以得到 $f(\hat{\pmb{x}_i})=\pmb{x}_i(X^TX)^{-1}X^Ty$，即有 $\hat{Y}=X(X^TX)^{-1}X^TY$，所以可以这样解释 $X(X^TX)^{-1}X^T$ 这个矩阵将原来的数据集的标注矩阵 $Y$ 映射到了测试集的预测矩阵 $\hat{Y}$ 上，即通过这个矩阵将其映射到了一个线性集上，因此可以认为是将空间中的标注投影到这个线性集上。所以从投影角度分析，这个投影矩阵就是将所有的数据通过投影的方式而映射到一个线性集上。同时根据这个是幂等矩阵，因此其无论投影多少次，其投影结果还是一致的。



​    





