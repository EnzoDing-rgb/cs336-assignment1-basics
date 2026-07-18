# Cross-Entropy（交叉熵）讲义笔记

符号一律跟 CS336 assignment 讲义走。

公式约定：行内 `$...$`，独立成行 `$$...$$`。

---

## 0. 语言模型在算什么？

给定一段 token 序列。讲义把序列写成

$$
x = (x_1, x_2, \ldots, x_{m+1})
$$

其中：

- $x_t$：第 $t$ 个位置上的 token（词表里的一个离散编号）
- $m+1$：整段序列长度；模型要做 $m$ 次「下一个词」预测

在位置 $i$ 上，模型只看前文

$$
x_{1:i} = (x_1, \ldots, x_i)
$$

并给出「下一个词是谁」的条件分布

$$
p_{\theta}(x_{i+1} \mid x_{1:i})
$$

其中：

- $\theta$：模型全部可训练参数
- $x_{i+1}$：位置 $i$ 之后那个真实 token（训练时已知）
- $p_{\theta}(\cdot \mid x_{1:i})$：参数为 $\theta$、以 $x_{1:i}$ 为条件的概率分布

一次 Transformer 前向，在位置 $i$ 会产出一串 **logits**

$$
o_i \in \mathbb{R}^{\text{vocab\_size}}
$$

其中：

- $o_i$：位置 $i$ 的 logit 向量
- $\text{vocab\_size}$：词表大小（有多少个可候选 token）
- $o_i[k]$：向量 $o_i$ 的第 $k$ 个分量，对应词表编号 $k$ 的原始分数

softmax 把 $o_i$ 变成概率（讲义式 17）：

$$
p_{\theta}(x_{i+1} \mid x_{1:i})
= \mathrm{softmax}(o_i)[x_{i+1}]
= \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{\text{vocab\_size}} \exp(o_i[a])}
$$

其中：

- $\mathrm{softmax}(o_i)$：与 $o_i$ 同形状的概率向量，各分量非负且和为 1
- $\mathrm{softmax}(o_i)[x_{i+1}]$：该概率向量在下标 $x_{i+1}$ 处的值，即模型分给正确答案的概率
- $\exp(\cdot)$：自然指数
- 求和指标 $a$：遍历词表里每一个候选 token 编号

位置 $i$ 上的逐步损失定义为

$$
\ell_i = -\log \mathrm{softmax}(o_i)[x_{i+1}]
$$

其中：

- $\ell_i$：位置 $i$ 的标量损失
- $\log$：自然对数（与信息论、最大似然同一套）

下文按 **熵 → 交叉熵 → 训练目标 → 稳定实现** 的顺序展开这条式子。

---

## 1. Entropy（熵）：一个分布有多不确定？

先只谈一个概率分布 $p$。

$p$ 给每个互斥结果 $a$ 一个概率 $p(a)$，且

$$
\sum_a p(a) = 1,\qquad p(a) \ge 0
$$

### 1.1 直觉

- 公平骰子：六个面各 $1/6$ → 很难事先确定结果 → 不确定度大
- 几乎总出 6 的骰子 → 很容易事先确定 → 不确定度小

熵用一个非负数概括「这个分布有多乱」。

### 1.2 定义

$$
H(p) = -\sum_{a} p(a)\,\log p(a)
$$

其中：

- $H(p)$：分布 $p$ 的熵（entropy）
- 求和扫过 $p$ 的全部结果 $a$
- $-\log p(a)$：结果 $a$ 的惊讶度；概率越小，惊讶度越大
- 外层再乘 $p(a)$ 并求和：按真实出现频率做平均惊讶度

两端情形：

| 分布形态 | 熵 |
|----------|----|
| 某个 $p(a)=1$，其余为 0（完全确定） | $H(p)=0$ |
| 均匀分布（每个结果一样可能） | 熵取到该支撑集上的最大值 |

**熵 $H(p)$ 只描述分布 $p$ 自身的不确定度。**

---

## 2. Cross-Entropy（交叉熵）：用 $q$ 描述 $p$ 有多费劲？

交叉熵同时用到两个分布：

- $p$：参照分布（真实 / 目标）
- $q$：用来「讲述」$p$ 的分布（模型预测）

定义：

$$
H(p, q) = -\sum_{a} p(a)\,\log q(a)
$$

其中：

- $H(p,q)$：以 $p$ 为权重、以 $\log q$ 打分的交叉熵
- $p(a)$：决定「哪个结果有多重要」
- $\log q(a)$：决定「模型觉得这个结果有多不惊讶」

名字里的 **cross**：权重来自 $p$，对数来自 $q$，两条分布写进同一条式子。

### 2.1 语言模型里的 $p$：独热（Dirac）

训练时正确答案是某一个具体 token $x_{i+1}$。对应的真分布是：

- $a = x_{i+1}$ 时，$p(a) = 1$
- 其余编号上，$p(a) = 0$

讲义脚注：交叉熵就是这个独热分布与 $\mathrm{softmax}(o_i)$ 之间的交叉熵。

代入定义，只有 $a = x_{i+1}$ 那一项留下：

$$
H(p, q) = -\log q(x_{i+1})
$$

令模型分布

$$
q = \mathrm{softmax}(o_i)
$$

即得

$$
\ell_i = -\log \mathrm{softmax}(o_i)[x_{i+1}]
$$

### 2.2 与 KL 散度的关系

KL 散度定义为

$$
D_{\mathrm{KL}}(p \parallel q) = H(p,q) - H(p)
$$

其中：

- $D_{\mathrm{KL}}(p \parallel q)$：用 $q$ 近似 $p$ 时的额外代价
- $H(p)$：真分布自身的熵
- $H(p,q)$：交叉熵

独热时 $H(p)=0$，因此

$$
H(p,q) = D_{\mathrm{KL}}(p \parallel q)
$$

最小化 $\ell_i$ 就是让 $q = \mathrm{softmax}(o_i)$ 贴近独热真分布。

---

## 3. 为什么用 $-\log q(x_{i+1})$ 当逐步损失？

位置 $i$ 上，模型输出的是整张词表上的概率向量；监督信号是离散正确答案 $x_{i+1}$。自然目标是：

> 把 $q(x_{i+1}) = \mathrm{softmax}(o_i)[x_{i+1}]$ 尽量做大。

把「越大越好」写成「最小化」时，常用

$$
\ell_i = -\log q(x_{i+1})
$$

### 3.1 $-\log$ 的形状

设 $q = q(x_{i+1}) \in (0,1]$。

| $q$ | $-\log q$（约） |
|-----|------------------|
| $1$ | $0$ |
| $0.7$ | $0.36$ |
| $0.05$ | $3.0$ |
| 趋近 $0$ | 趋向 $+\infty$ |

要点：

- 正确答案概率接近 1 → 损失接近 0
- 正确答案概率很小 → 损失很大
- 损失对概率光滑，便于反向传播

### 3.2 与最大似然同一条目标

讲义把训练集记为 $\mathcal{D}$，每条序列长度相关的预测步数为 $m$。整体损失（讲义式 16）：

$$
\ell(\theta; \mathcal{D})
= \frac{1}{|\mathcal{D}|\,m}
\sum_{x\in\mathcal{D}}
\sum_{i=1}^{m}
-\log p_{\theta}(x_{i+1} \mid x_{1:i})
$$

其中：

- $\mathcal{D}$：训练序列的集合
- $|\mathcal{D}|$：集合里有多少条序列
- 外层求和：扫过每条训练序列 $x$
- 内层求和：扫过该序列上每一个预测位置 $i=1,\ldots,m$
- 前面的 $\dfrac{1}{|\mathcal{D}|\,m}$：对「序列数 × 每序列预测步数」取平均

这条式子就是：在数据上最大化真实下一个词的似然，写成最小化负对数似然；在独热监督下，它与交叉熵一致。

### 3.3 与「只看 argmax」的对比

只看「概率最大的类别是否等于 $x_{i+1}$」时，反馈是离散的对/错。

交叉熵直接使用连续值 $q(x_{i+1})$：

- $q(x_{i+1})=0.51$ 与 $q(x_{i+1})=0.99$ 都会产生不同损失，模型会继续把概率推向 1
- 错误预测时，损失大小随 $q(x_{i+1})$ 连续变化，梯度信息更细

### 3.4 数值例子

词表 $\{A,B,C\}$，正确答案 $B$。

模型较好：

$$
\mathrm{softmax}(o_i) = (0.1,\ 0.7,\ 0.2)
\quad\Rightarrow\quad
\ell_i = -\log 0.7 \approx 0.357
$$

模型较差：

$$
\mathrm{softmax}(o_i) = (0.6,\ 0.05,\ 0.35)
\quad\Rightarrow\quad
\ell_i = -\log 0.05 \approx 3.0
$$

---

## 4. 从 logits 到可实现的 $\ell_i$

### 4.1 展开 softmax

把定义直接展开：

$$
\ell_i = -\log\left(
\frac{\exp(o_i[x_{i+1}])}{\sum_a \exp(o_i[a])}
\right)
$$

logit 很大时，$\exp(o_i[a])$ 容易溢出。

### 4.2 消掉成对的 log / exp

对数运算法则给出：

$$
\ell_i
= -o_i[x_{i+1}] + \log\sum_a \exp(o_i[a])
$$

实现时直接算这一行（讲义：Cancel out log and exp whenever possible）。

### 4.3 减最大值保证稳定

令

$$
m = \max_a o_i[a]
$$

这里的 $m$ 是向量 $o_i$ 各分量的最大值（与损失符号 $\ell$ 无关）。

恒等式：

$$
\sum_a \exp(o_i[a])
= \exp(m)\sum_a \exp(o_i[a]-m)
$$

因而

$$
\log\sum_a \exp(o_i[a])
= m + \log\sum_a \exp(o_i[a]-m)
$$

代入得稳定形式：

$$
\ell_i
= -(o_i[x_{i+1}] - m) + \log\sum_a \exp(o_i[a]-m)
$$

此时每个 $\exp(o_i[a]-m) \le 1$，指数计算安全（讲义：Subtract the largest element for numerical stability）。

### 4.4 Batch 上取平均

作业实现：`inputs` 为 logits，最后一维是词表；`targets` 为正确类别下标；对所有样本维上的 $\ell$ 做 **mean**。

这与讲义 (16) 中的

$$
\frac{1}{|\mathcal{D}|\,m}\sum\sum \cdots
$$

同一精神：先对每个位置算 $\ell_i$，再对 batch（及序列位置）取平均。

维度约定（§3.2）：batch 类维度在前，词表维在最后。

---

## 5. 总览

1. **$H(p)$**：分布 $p$ 的不确定度。
2. **$H(p,q)$**：用 $q$ 描述 $p$ 的交叉代价。
3. **LM 监督是独热** → $\ell_i = -\log \mathrm{softmax}(o_i)[x_{i+1}]$。
4. **训练**最小化平均 $\ell_i$，即最大似然 / 交叉熵。
5. **实现**用 $-o_i[x_{i+1}] + \log\sum\exp$，并减 $\max o_i$ 保稳定，最后 mean。

---

## 6. 作业对照

目标：

$$
\ell_i = -\log \mathrm{softmax}(o_i)[x_{i+1}]
$$

要点：

1. Subtract the largest element for numerical stability
2. Cancel out log and exp whenever possible
3. Handle batch dims；return the **average** across the batch

```bash
uv run pytest -k test_cross_entropy
```

adapter：`adapters.run_cross_entropy`  
`inputs` = 未归一化 logits，`targets` = 正确类别下标。
