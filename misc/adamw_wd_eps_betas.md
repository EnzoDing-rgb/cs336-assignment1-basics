# AdamW 里三个不起眼超参：weight decay / eps / betas

用本仓库 `configs/tinystories_small.yaml` 的数过一遍，方便对照 `cs336_basics/model/optimizer.py`。

公式约定：行内 `$...$`，独立成行 `$$...$$`。

本配置：

$$
\alpha = 3\times 10^{-4},\quad
\lambda = 0.1,\quad
(\beta_1,\beta_2)=(0.9,\ 0.95),\quad
\varepsilon = 10^{-8}
$$

---

## 1. Weight decay（$\lambda$）

### 1.1 代码在干什么

```python
p.mul_(1 - lr * weight_decay)   # θ ← θ * (1 - α λ)
```

代入本配置：

$$
\alpha \lambda = (3\times 10^{-4})\times 0.1 = 3\times 10^{-5}
$$

$$
1 - \alpha \lambda = 0.99997
$$

### 1.2 具体前后对比

取某一个权重 $\theta = 2.0$，**只做这一行**（暂时不算后面的 Adam 更新）：

$$
\begin{aligned}
\lambda &= 0   &\Rightarrow\quad & 2.0 \times 1 = 2.0 \\
\lambda &= 0.1 &\Rightarrow\quad & 2.0 \times 0.99997 = 1.99994
\end{aligned}
$$

一万步量级、若梯度长期几乎推不动这个权重：

$$
2.0 \times (0.99997)^{10000} \approx 1.48
$$

幅度被往 0 收。这就是 weight decay 要做的事：参数别越训越大、死记训练集。

### 1.3 AdamW 的「解耦」用数说话

本实现：收缩直接乘在 $\theta$ 上，**不**把 $\lambda\theta$ 加进 `grad` 再喂给 $m$、$v$。

若改成旧式 L2（`grad ← grad + λθ`），同一组数：

$$
\theta=2,\ \lambda=0.1 \Rightarrow \lambda\theta=0.2
$$

`grad` 多了 $0.2$，再进 Adam 的 $m/\sqrt{v}$。大参数、小 $v$ 时，这 $0.2$ 被自适应步长放大或缩小，**衰减强度和自适应缠在一起**。AdamW 把「每步乘 $0.99997$」单独拿出来，衰减比例就是 $\alpha\lambda$，跟 $\sqrt{v}$ 无关。

### 1.4 容易忘

本行用的是 group 里的 $\alpha$（`lr`）。  
下面 Adam 那一步用的才是带 bias correction 的 $\alpha_t$（代码里的 `lr_t`）。

---

## 2. `eps`（$\varepsilon=10^{-8}$）

### 2.1 代码在干什么

```python
p.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)
# 即  θ ← θ - α_t * m / (√v + ε)
```

### 2.2 具体前后对比

危险情形：$m=10^{-3}$（还有一点动量），$v=0$（二阶还没攒起来）：

$$
\begin{aligned}
\varepsilon &= 0
&\Rightarrow\quad
\frac{10^{-3}}{0} &\rightarrow \mathrm{Inf}/\mathrm{NaN}
\quad\text{参数坏掉} \\[0.5em]
\varepsilon &= 10^{-8}
&\Rightarrow\quad
\frac{10^{-3}}{10^{-8}} &= 10^{5}
\quad\text{步子大但有限}
\end{aligned}
$$

正常情形：$\sqrt{v}=0.1$，同样 $m=10^{-3}$：

$$
\frac{10^{-3}}{0.1 + 10^{-8}} \approx 0.01
$$

这时 $\varepsilon$ 可以当成不存在。它是分母地板，通常不用扫；别随手改成 $10^{-4}$ 这种会明显拖小步长的量级。

---

## 3. `betas`，尤其是 $\beta_2$

### 3.1 代码在干什么

```python
m.mul_(beta1).add_(grad, alpha=1 - beta1)           # m ← β1 m + (1-β1) g
v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # v ← β2 v + (1-β2) g²
```

本配置：$\beta_1=0.9$，$\beta_2=0.95$。  
经典 Adam 默认常常是 $\beta_2=0.999$。

### 3.2 用数看 $\beta_2$ 差在哪

$v$ 每步保留旧值的比例就是 $\beta_2$。旧信息衰减到原来的 $e^{-1}\approx 0.37$ 大约需要

$$
\frac{1}{1-\beta_2}\ \text{步的量级}
$$

$$
\begin{aligned}
\beta_2 &= 0.95  &\Rightarrow\quad & 1/(1-0.95) = 20 \text{ 步量级} \\
\beta_2 &= 0.999 &\Rightarrow\quad & 1/(1-0.999) = 1000 \text{ 步量级}
\end{aligned}
$$

$\beta_2=0.95$：$v$（「梯度有多大声」）跟得快。  
$\beta_2=0.999$：$v$ 记很久以前的尺度。

语言模型梯度噪声大，常用略小的 $\beta_2$（如 $0.95$），让分母更跟当前梯度幅度。

### 3.3 和 bias correction 绑在一起（具体算）

代码：

$$
\alpha_t = \alpha \cdot \frac{\sqrt{1-\beta_2^{t}}}{1-\beta_1^{t}}
$$

取 $t=1$，$\alpha=3\times 10^{-4}$，$\beta_1=0.9$：

$$
\begin{aligned}
\beta_2=0.95:
&\quad
\sqrt{1-0.95}=\sqrt{0.05}\approx 0.224,\quad
1-\beta_1=0.1 \\
&\quad
\alpha_t \approx 3\times 10^{-4} \times \frac{0.224}{0.1} \approx 6.7\times 10^{-4}
\\[0.75em]
\beta_2=0.999:
&\quad
\sqrt{1-0.999}=\sqrt{0.001}\approx 0.0316 \\
&\quad
\alpha_t \approx 3\times 10^{-4} \times \frac{0.0316}{0.1} \approx 9.5\times 10^{-5}
\end{aligned}
$$

同是第 1 步，换 $\beta_2$ 会改变 $\alpha_t$ 的校正幅度。yaml 写了 `0.95` 就和默认 `0.999` 当两套设定。

$\beta_1=0.9$ 几乎人人用，扫的时候优先想 $\beta_2$。

---

## 4. 三行对照（复习用）

$$
\begin{aligned}
&\textbf{weight decay:}
&&
\theta \leftarrow \theta(1-\alpha\lambda)
&&
\text{本配置每步}\times 0.99997
\\
&\textbf{eps:}
&&
\theta \leftarrow \theta - \alpha_t \frac{m}{\sqrt{v}+\varepsilon}
&&
\varepsilon=10^{-8}\ \text{垫分母}
\\
&\textbf{betas:}
&&
m,v\ \text{滑动平均}
&&
\beta_2=0.95\ \text{比 }0.999\ \text{跟得快}
\end{aligned}
$$

对应代码顺序：先 `p.mul_(1 - lr * weight_decay)`，再更新 $m$、$v$，再 `addcdiv_`（含 $\varepsilon$ 与 $\alpha_t$）。
