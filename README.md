# GAN loss with pytorch

f-divergence 的定义：

$$
D_f(P||Q)=\int_\Omega f(\frac{dP}{dQ})dQ
$$

其中，$f$ 为下凸函数，满足$f(1)=0$；$P=p(x), Q=q(x)$ ，进一步可将上式转化为：

$$
\int_\Omega f(\frac{p(x)dx}{q(x)dx})q(x)dx=\int_\Omega f(\frac{p(x)}{q(x)})q(x)dx
$$

通过合理的选择不同的 $f$, 可以转化为不同的 divergence，常用的例子有：KL- divergence，JS divergence 等。

## [KL divergence](./KLD.py)

KL 散度又称为相对熵，信息散度，信息增益。KL 散度是是两个概率分布 P 和 Q 差别的非对称性的度量。 KL 散度是用来度量使用基于 Q 的编码来编码来自P的样本平均所需的额外的位元数。 典型情况下，**P 表示数据的真实分布，Q 表示数据的理论分布、模型分布、或 P 的近似分布**。

另 $f=xlogx$，f-divergence 可以被转化为 KL-divergence：

$$
KL(P||Q)=D_f(P||Q)=\int_\Omega \frac{p(x)}{q(x)}log{\frac{p(x)}{q(x)}}q(x)dx=\int_\Omega p(x)log{\frac{p(x)}{q(x)}}dx=\sum_{x}p(x)logp(x)-\sum_{x}p(x)logq(x)
$$

因为对数函数是凸函数，所以 KL divergence 散度的值为非负数。

有时会将 KL 散度称为 KL 距离，**但它并不满足距离的性质**（第1点和第2点）：

1. KL散度不是对称的，即 $KL(P||Q)\neq KL(Q||P)$。因此不能将它视为“距离”，它衡量的是一个分布相比另一个分布的信息损失；

2. KL散度不满足三角不等式，即在三角形中两边之和大于第三边；

3. 非负性，即$KL(P||Q)>0$。

**在GAN的训练中，使用 BCE 损失来充当**：

$$
BCE(output, target)=-\frac{1}{n}\sum_i(t[i]*log(o[i])+(1-t[i])*log(1-o[i]))
$$

可以简单推导出：

$$
BCE(output, 1)=-\frac{1}{n}\sum_ilog(o[i])=KL(P=1||Q=output)
$$

$$
BCE(output,0)=-\frac{1}{n}\sum_ilog(1-o[i])=KL(P=0||Q=output)
$$

注意，当 target=0 时，在 KL 散度里面应该修改定义为：

$$KL(P||Q)=\sum_{x}(1-p(x))log(1-p(x))-\sum_{x}(1-p(x))log(1-q(x))$$

伪代码如下：

```python
adversarial_loss = torch.nn.BCELoss()

valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
# -----------------
#  Train Generator
# -----------------
g_loss = adversarial_loss(validity, valid)

g_loss.backward()
optimizer_G.step()

# -----------------
#  Train Discriminator
# -----------------
d_real_loss = adversarial_loss(real_pred, valid)
d_fake_loss = adversarial_loss(fake_pred, fake)
d_loss = (d_real_loss + d_fake_loss) / 2

d_loss.backward()
optimizer_D.step()
```



## [JS divergence](./JSD.py)

**JS散度度量了两个概率分布的相似度，基于KL散度的变体，解决了KL散度非对称的问题。一般地，JS散度是对称的，其取值是0到1之间。**

定义如下：

$$
JS(P||Q)=\frac{1}{2}KL(P||\frac{P+Q}{2})+\frac{1}{2}KL(Q||\frac{P+Q}{2})
$$

**JS散度主要有两个性质：**

- 对称性，即 $JS(P||Q)=JS(Q||P)$
- 值域范围：JS 散度的值域范围为 [0, 1]，相同为0，相反为1。

**KL散度和JS散度度量的时候有一个问题：**
如果两个分配P,Q离得很远，完全没有重叠的时候，那么KL散度值是没有意义的，而JS散度值是一个常数。这在学习算法中是比较致命的，这就意味这这一点的梯度为0。梯度消失了。



## 为什么在原始 GAN 的训练中使用的是类似交叉熵的损失？

这里，我们可以先看一下交叉熵的定义：

$$
H(p,q)=\sum_{i=1}^{N}p(x_i)log\frac{1}{q(x_i)}
$$

交叉熵和相对熵（KL散度）的关系如下：

$$
KL(p,q)=H(p,q)-H(p)
$$

因为训练数据的分布是已知的，故H(p)=0， $KL(p,q)$ 就直接等于 $H(p,q)$ 。所以交叉熵与KL散度的意义十分类似。



## Reference

https://zhuanlan.zhihu.com/p/341461665

https://blog.csdn.net/leviopku/article/details/81388306

https://blog.csdn.net/Marilynmontu/article/details/89260109
