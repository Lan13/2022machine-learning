# HW9

PB20111689 蓝俊玮

## T1

> 给定任意的两个相同长度向量 $\pmb{x},\pmb{y}$，其余弦距离为 $1-\frac{\pmb{x}^T\pmb{y}}{|\pmb{x}||\pmb{y}|}$，证明余弦距离不满足传递性，而余弦夹角 $\arccos(\frac{\pmb{x}^T\pmb{y}}{|\pmb{x}||\pmb{y}|})$ 满足

取反例 $A=(1,0),B=(1,1),C=(0,1)$，则可以计算得到 $dist(A,B)=1-\frac{\sqrt{2}}{2},\ dist(B,C)=1-\frac{\sqrt{2}}{2},\ dist(A,C)=1$，得到 $2-\sqrt{2}=dist(A,B)+dist(B,C)<dist(A,C)=1$，所以余弦距离在某些情况下是不满足传递性的。

而对于余弦夹角来说，要证明 $\arccos(\frac{\pmb{x}^T\pmb{y}}{|\pmb{x}||\pmb{y}|})\le\arccos(\frac{\pmb{x}^T\pmb{z}}{|\pmb{x}||\pmb{z}|})+\arccos(\frac{\pmb{z}^T\pmb{y}}{|\pmb{z}||\pmb{y}|})$ 。

由 $\cos(\alpha+\beta)=\cos\alpha\cos\beta-\sin\alpha\sin\beta=\cos\alpha\cos\beta-\sqrt{1-\cos^2\alpha}\sqrt{1-\cos^2\beta}$ 可以得到： $\alpha+\beta=\arccos(\cos\alpha\cos\beta-\sqrt{1-\cos^2\alpha}\sqrt{1-\cos^2\beta})$

则等价证明为：$\arccos(\frac{\pmb{x}^T\pmb{y}}{|\pmb{x}||\pmb{y}|})\le\arccos(\frac{\pmb{x}^T\pmb{z}}{|\pmb{x}||\pmb{z}|})+\arccos(\frac{\pmb{z}^T\pmb{y}}{|\pmb{z}||\pmb{y}|})=\arccos(\frac{\pmb{x}^T\pmb{z}}{|\pmb{x}||\pmb{z}|}\frac{\pmb{z}^T\pmb{y}}{|\pmb{z}||\pmb{y}|}-\sqrt{1-(\frac{\pmb{x}^T\pmb{z}}{|\pmb{x}||\pmb{z}|})^2}\sqrt{1-(\frac{\pmb{z}^T\pmb{y}}{|\pmb{z}||\pmb{y}|})^2})$ 即证明：$\arccos(\frac{\pmb{x}^T\pmb{y}}{|\pmb{x}||\pmb{y}|})\le \arccos(\frac{\pmb{x}^T\pmb{z}}{|\pmb{x}||\pmb{z}|}\frac{\pmb{z}^T\pmb{y}}{|\pmb{z}||\pmb{y}|}-\sqrt{1-(\frac{\pmb{x}^T\pmb{z}}{|\pmb{x}||\pmb{z}|})^2}\sqrt{1-(\frac{\pmb{z}^T\pmb{y}}{|\pmb{z}||\pmb{y}|})^2})$。由于 $\arccos$ 函数在 $[-1,1]$ 上是单调递减的，则需证明：$\frac{\pmb{x}^T\pmb{y}}{|\pmb{x}||\pmb{y}|}\ge \frac{\pmb{x}^T\pmb{z}}{|\pmb{x}||\pmb{z}|}\frac{\pmb{z}^T\pmb{y}}{|\pmb{z}||\pmb{y}|}-\sqrt{1-(\frac{\pmb{x}^T\pmb{z}}{|\pmb{x}||\pmb{z}|})^2}\sqrt{1-(\frac{\pmb{z}^T\pmb{y}}{|\pmb{z}||\pmb{y}|})^2}$，两边同时乘 $|\pmb{x}||\pmb{y}||\pmb{z}|^2$ ，即需要证明：$\sqrt{|\pmb{x}|^2|\pmb{z}|^2-(\pmb{x}^T\pmb{z})^2}\sqrt{|\pmb{z}|^2|\pmb{y}|^2-(\pmb{z}^T\pmb{y})^2}\ge\pmb{x}^T\pmb{z}\pmb{z}^T\pmb{y}-|\pmb{z}|^2\pmb{x}^T\pmb{y}$，两边平方得到：$(|\pmb{x}|^2|\pmb{z}|^2-(\pmb{x}^T\pmb{z})^2)(|\pmb{z}|^2|\pmb{y}|^2-(\pmb{z}^T\pmb{y})^2)\ge(\pmb{x}^T\pmb{z}\pmb{z}^T\pmb{y}-|\pmb{z}|^2\pmb{x}^T\pmb{y})^2$。即需要证明这个不等式成立。

设 $\pmb{x}=(x_1,x_2,...,x_n)^T,\pmb{y}=(y_1,y_2,...,y_n)^T,\pmb{z}=(z_1,z_2,...,z_n)^T$
$$
|\pmb{x}|^2|\pmb{z}|^2-(\pmb{x}^T\pmb{z})^2=(x_1^2+x_2^2+...+x_n^2)(z_1^2+z_2^2+...+z_n^2)-(x_1z_1+x_2z_2+...+x_nz_n)^2\\
=\sum\limits_{i=1}^{n-1}\sum\limits_{j=i+1}^{n}(x_iz_j-x_jz_i)^2\\
|\pmb{z}|^2|\pmb{y}|^2-(\pmb{z}^T\pmb{y})^2=\sum\limits_{i=1}^{n-1}\sum\limits_{j=i+1}^{n}(z_iy_j-z_jy_i)^2\\
(\pmb{x}^T\pmb{z}\pmb{z}^T\pmb{y}-|\pmb{z}|^2\pmb{x}^T\pmb{y})^2=((x_1z_1+...+x_nz_n)(z_1y_1+...+z_ny_n)-(z_1^2+...+z_n^2)(x_1y_1+...+x_ny_n))^2\\
=(\sum\limits_{i=1}^{n-1}\sum\limits_{j=i+1}^{n}(x_iz_j-x_jz_i)(z_iy_j-z_jy_i))^2
$$
则由柯西不等式
$$
(\sum\limits_{i=1}^{n-1}\sum\limits_{j=i+1}^{n}(x_iz_j-x_jz_i)^2)(\sum\limits_{i=1}^{n-1}\sum\limits_{j=i+1}^{n}(z_iy_j-z_jy_i)^2)\ge(\sum\limits_{i=1}^{n-1}\sum\limits_{j=i+1}^{n}(x_iz_j-x_jz_i)(z_iy_j-z_jy_i))^2
$$
即得到证明：
$$
(|\pmb{x}|^2|\pmb{z}|^2-(\pmb{x}^T\pmb{z})^2)(|\pmb{z}|^2|\pmb{y}|^2-(\pmb{z}^T\pmb{y})^2)\ge(\pmb{x}^T\pmb{z}\pmb{z}^T\pmb{y}-|\pmb{z}|^2\pmb{x}^T\pmb{y})^2
$$
因此证明出余弦夹角满足传递性。

## T2

> 证明 k-means 算法的收敛性

k-means 的损失函数为 $E=\sum\limits_{i=1}^{k}\sum\limits_{\pmb{x}\in C_i}||\pmb{x}-\pmb{\mu}_i||^2_2$，则有 $\frac{\part E}{\part \pmb{\mu}_i}=2\sum_\limits{\pmb{x}\in C_i}(\pmb{x}-\pmb{\mu}_i)=0$ 得到 $\pmb{\mu}_k=\frac{1}{|C_i|}\sum\limits_{\pmb{x}\in C_i}\pmb{x}=\pmb{\mu}_k'$，可以得知在更新之后的均值向量 $\pmb{\mu}_k'$ 是损失函数最小值的一个极值点。那么就说明了，在每次更新中心点为均值向量时，都能让损失函数 $E$ 变得更小。因此 k-means 算法的更新能够让损失函数 $E$ 单调递减，同时又因为 $E\ge0$ 是有界的，因此 k-means 算法具有收敛性。 

## T3

> 在 k-means 算法中替换欧式距离为其他任意的度量，请问“聚类簇”中心如何计算？

当不再采用欧式距离时，则可以通过计算每个聚类簇中的距离度量之和最小的点作为中心点。即对于一个簇 $C_i$，选取 $x_0=\arg\min\limits_{x_0\in C_i}{\sum_\limits{x\in C_i}dist(x_0,x)}$，其中 $dist(x_0,x)$ 为新的度量方式。即从一个聚类簇中，选取这样一个点：它到这个聚类簇中其它的所有点的距离度量之和最小，并将这个点作为该聚类簇的中心点。

