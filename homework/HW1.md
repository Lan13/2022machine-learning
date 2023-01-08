# HW1

PB20111689 蓝俊玮

## T1

$\frac{\partial\ln\det(\textbf{A})}{\partial x}=\frac{\partial\ln\det(\textbf{A})}{\partial\det(\textbf{A})}\frac{\partial\det(\textbf{A})}{\partial x}=\frac{1}{\det(\textbf{A})}\frac{\partial\det(\textbf{A})}{\partial x}$

由链式法则 $\frac{\partial\det(\textbf{A})}{\partial x}=\sum_i\sum_j\frac{\partial\det(\textbf{A})}{\part a_{ij}}\frac{\part a_{ij}}{\part x}$ 而 $\frac{\part \det(\textbf{A})}{\part a_{ij}}=\det(\textbf{A})\textbf{A}^{-1}_{ji},\frac{\part\det(\textbf{A})}{\part x}=\sum_i\sum_j\det(\textbf{A})\textbf{A}^{-1}_{ji}(\frac{\part\textbf{A}}{\part x})_{ij}=\det(\textbf{A})\sum_i\sum_j\textbf{A}^{-1}_{ji}(\frac{\part\textbf{A}}{\part x})_{ij}=\det(\textbf{A})\tr(\textbf{A}^{-1}\frac{\part \textbf{A}}{\part x})$

所以结果为 $\frac{\partial\ln\det(\textbf{A})}{\partial x}=\tr(\textbf{A}^{-1}\frac{\part \textbf{A}}{\part x})$

## T2

根据给定的西瓜数据，西瓜的色泽共有 2 种，根蒂和敲声有 3 种。所以对于单个合取式来说共有 $2\times3\times3=18$ 种，那么对于所有的析取范式来说，其上限为 $\pmatrix{18\\0}+\pmatrix{18\\1}+\pmatrix{18\\2}+...+\pmatrix{18\\18}=2^{18}$ 种。当 $k=9$ 的时候，去重所有可能的结果后就可以使假设空间达到上限。

## T3

因为 $\textbf{x}=[\textbf{x}_1, \textbf{x}_2]\sim\mathcal{N}(\mu,\Sigma)$，则 $\mu=\pmatrix{\mu_1\\\mu_2},\Sigma=\pmatrix{\Sigma_{11}\quad\Sigma_{12}\\\Sigma_{21}\quad\Sigma_{22}}$，则可以计算得到条件分布 $(\textbf{x}_1|\textbf{x}_2)\sim\mathcal{N}(\overline{\mu},\overline{\Sigma})$ 其中 $\overline{\mu}=\mu_1+\Sigma_{12}\Sigma_{22}^{-1}(\textbf{x}_2-\mu_2),\overline{\Sigma}=\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$。而对于边缘分布，由其性质易知边缘分布的参数就是与自己本身参数所相关的一维正态分布，所以其分布为 $\textbf{x}_1\sim\mathcal{N}(\mu_1,\Sigma_{11})$。因此得到了这些分布后，其概率也就显然可得
$$
P(\textbf{x}_1)=\frac{1}{\sqrt{{(2\pi)}^n\det({\Sigma_{11}})}}\exp{(-\frac{1}{2}(\textbf{x}_1-\mu_1)^T\Sigma_{11}^{-1}(\textbf{x}_1-\mu_1))} \\
P(\textbf{x}_1|\textbf{x}_2)=\frac{1}{\sqrt{{(2\pi)}^n\det({\overline{\Sigma}})}}\exp{(-\frac{1}{2}(\textbf{x}_1-\overline\mu)^T{\overline\Sigma}^{-1}(\textbf{x}_1-\overline\mu))} 
$$

## T4

首先范数 ${\Vert\textbf{x}\Vert}_p$ 的定义式为 ${\Vert\textbf{x}\Vert}_p=\sqrt[p]{(\sum_{i=1}^{i=n}{\vert x_i\vert}^p)}$，那么根据凸函数的定义 $\forall \textbf{x},\textbf{y}\in\R^n,\lambda\in[0,1]$ 有 $f(\lambda\textbf{x}+(1-\lambda)\textbf{y})\le \lambda f(\textbf{x})+(1-\lambda)f(\textbf{y})$。那么可以根据范数的性质 ${\lambda\Vert\textbf{x}\Vert}_p=\lambda{\Vert\textbf{x}\Vert}_p$ 和三角性质 ${\Vert\textbf{x}+\textbf{y}\Vert}_p\le{\Vert\textbf{x}\Vert}_p+{\Vert\textbf{y}\Vert}_p$ 可以得到 ${\Vert\lambda\textbf{x}+(1-\lambda)\textbf{y}\Vert}_p\le{\Vert\lambda\textbf{x}\Vert}_p+{\Vert(1-\lambda)\textbf{y}\Vert}_p=\lambda{\Vert\textbf{x}\Vert}_p+(1-\lambda){\Vert\textbf{y}\Vert}_p$，所以范数 ${\Vert\textbf{x}\Vert}_p$ 满足凸函数的性质，因此范数 ${\Vert\textbf{x}\Vert}_p$ 是凸函数。

## T5

充分性证明：

因为有 $f(\lambda\textbf{x}+(1-\lambda)\textbf{y})\le \lambda f(\textbf{x})+(1-\lambda)f(\textbf{y}),\forall \textbf{x},\textbf{y}\in\R^n,\forall\lambda\in[0,1]$，所以可得
$$
f(\lambda\textbf{y}+(1-\lambda)\textbf{x})\le \lambda f(\textbf{y})+(1-\lambda)f(\textbf{x})\\
f(\textbf{x}+\lambda(\textbf{y}-\textbf{x}))\le \lambda f(\textbf{y})+(1-\lambda)f(\textbf{x})\\
f(\textbf{y})\ge f(\textbf{x})+\frac{f(\textbf{x}+\lambda(\textbf{y}-\textbf{x}))-f(\textbf{x})}{\lambda}\\
\lim_{\lambda\rightarrow0} f(\textbf{y})\ge \lim_{\lambda\rightarrow0} (f(\textbf{x})+\frac{f(\textbf{x}+\lambda(\textbf{y}-\textbf{x}))-f(\textbf{x})}{\lambda})\\
f(\textbf{y})\ge f(\textbf{x})+\lim_{\lambda\rightarrow0}\frac{f(\textbf{x}+\lambda(\textbf{y}-\textbf{x}))-f(\textbf{x})}{\lambda}\\
f(\textbf{y})\ge f(\textbf{x})+\nabla {f(\textbf{x})}^T(\textbf{y}-\textbf{x})
$$
必要性证明：

因为有 $f(\textbf{y})\ge f(\textbf{x})+\nabla {f(\textbf{x})}^T(\textbf{y}-\textbf{x}),\forall \textbf{x},\textbf{y}\in\R^n$，所以假设并得到：
$$
\textbf{z}=t\textbf{y}+(1-t)\textbf{x}\\
f(\textbf{y})\ge f(\textbf{z})+\nabla {f(\textbf{z})}^T(\textbf{y}-\textbf{z})\\
f(\textbf{x})\ge f(\textbf{z})+\nabla {f(\textbf{z})}^T(\textbf{x}-\textbf{z})\\
$$
接着对 1 式和 2 式分别乘 $t$ 和 $1-t$ 并相加得到
$$
tf(\textbf{y})+(1-t)f(\textbf{x})\ge tf(\textbf{z})+t\nabla {f(\textbf{z})}^T(\textbf{y}-\textbf{z})+(1-t)f(\textbf{z})+(1-t)\nabla {f(\textbf{z})}^T(\textbf{x}-\textbf{z})\\
=f(\textbf{z})+\nabla {f(\textbf{z})}^T(t(\textbf{y}-\textbf{x})+(\textbf{x}-\textbf{z}))\\
=f(\textbf{z})+\nabla {f(\textbf{z})}^T(t(\textbf{y}-\textbf{x})+(\textbf{x}-t\textbf{y}-(1-t)\textbf{x}))=f(\textbf{z})
$$
因此得到了 $tf(\textbf{y})+(1-t)f(\textbf{x})\ge f(t\textbf{y}+(1-t)\textbf{x})=f(\textbf{z})$。

综上所述，可以得到凸函数的 0 阶和 1 阶条件相互等价。