# ZeRO：数据并行里「模型状态」怎么切、为什么省显存

本文把 ZeRO 四档（Baseline → $P_{os}$ → $P_{os+g}$ → $P_{os+g+p}$）从零推到尾。  
数字一边用一套标准算例（$\Psi=7.5\mathrm{B}$、$N_d=64$、$K=12$），一边用本仓库 `TransformerLM` / TinyStories / GPT-2 XL 配置算一遍。

**本文在讲什么：**

> 全程只处理 **模型状态（model states）**：参数 + 梯度 + 优化器状态。  
> 三者都按参数个数 $\Psi$（本仓库记作 $P$）缩放。  
> 文中的 **state** 就是这三样合在一起。

训练显存里还有激活 $A$（随 $B,L$ 变），那是另一笔账，见 §1 的四桶表；**ZeRO 的切分公式只动 $\Psi$ 那一侧。**

---

## 0. 符号表（全文统一）

| 符号 | 含义 |
|------|------|
| $\Psi$ | 可训练参数个数（元素个数）。标准算例取 $\Psi=7.5\times 10^9$。本仓库 TinyStories small 约 $\Psi\approx 2.31\times 10^7$；`reports/flops.md` 的 GPT-2 XL 配置 $\Psi=1\,640\,452\,800$。 |
| $N_d$ | 数据并行 GPU 数（degree of data parallelism）。标准算例取 $N_d=64$。 |
| $K$ | **每个参数**对应的优化器状态字节数（mixed-precision Adam 常见记账）。标准算例取 $K=12$。 |
| $P$ / $\Psi$ | 同一件事：参数元素个数。本仓库 `activiations.md` 里用 $P$，本文算例多用 $\Psi$。 |
| $B,L,d,h,N,V,d_{\mathrm{ff}}$ | batch、序列长、`d_model`、头数、层数、词表、FFN 宽（与 `activiations.md` 相同） |
| Parameters | 参数（权重等） |
| Gradients | 梯度 |
| Optimizer states | 优化器状态（Adam 的 $m$、$v$ 等） |

字节约定（**mixed precision 记账**）：

| 东西 | 每参数多少字节 | 为什么 |
|------|----------------|--------|
| FP16 参数副本 | $2$ | 前向/反传常用半精度权重 |
| FP16 梯度 | $2$ | 反传得到的梯度也按半精度记 |
| 优化器状态 | $K=12$ | 见 §2：FP32 master + Adam 的 $m$、$v$ |

因此「一份完整模型状态」是：

$$
(2 + 2 + K)\,\Psi = 16\Psi \quad\text{（字节）}
$$

---

## 1. 训练时 GPU 上有几桶东西？ZeRO 动哪几桶？

一次 `train` 步（`cs336_basics/train.py`）里，显存至少有四类：

$$
M_{\mathrm{peak}}
\;\approx\;
M_{\mathrm{params}}
+ M_{\mathrm{grads}}
+ M_{\mathrm{optim}}
+ M_{\mathrm{activations}}
$$

（还有临时 buffer、碎片等，这里忽略。）

| 桶 | 是什么 | 形状跟谁走 | 本仓库落点 | ZeRO 是否切它 |
|----|--------|------------|------------|---------------|
| Parameters | 可训练权重 $W$、RMSNorm 的 $g$ 等 | 只跟模型结构有关，按 $\Psi$ | `model.parameters()` | 是（ZeRO-3） |
| Gradients | $\partial L/\partial p$ | 与参数同形状，按 $\Psi$ | `p.grad` | 是（ZeRO-2 起） |
| Optimizer states | AdamW 的 $m$、$v$ 等 | 与参数同形状，按 $\Psi$ | `optimizer.state[p]` | 是（ZeRO-1 起） |
| Activations | 前向中间张量 | 按 $B,L$（及 $L^2$） | 计算图里的 $Q,K,V,\mathrm{logits},\ldots$ | 否；另算 |

合称 **model states** 的是前三桶。  
每档「每卡显存」都是这三桶在每卡上的字节数，自变量只有 $\Psi$、$K$、$N_d$。

---

## 2. 优化器状态为什么按「每参数 $K=12$ 字节」这么贵？

Adam 的 $m$、$v$ 和参数 **一一对应、同形状**，元素个数都是 $\Psi$。  
贵，是因为 **每个参数要多存好几份同形状的数，而且常用更高精度**。

用本仓库 AdamW 语言（`misc/optimizer_state_and_groups.md`）：

```text
p              = 参数本身
p.grad         = 本步梯度 g
state[p]["m"]  = 一阶动量（与 p 同形状）
state[p]["v"]  = 二阶动量（与 p 同形状）
```

Mixed-precision 训练里常见一整套是：

| 份 | 精度 | 字节/参数 | 角色 |
|----|------|-----------|------|
| 计算用参数 | FP16 | 2 | 前向、反传 |
| 梯度 | FP16 | 2 | `backward` |
| master 参数 | FP32 | 4 | 真正被 Adam 更新的「高精度副本」 |
| $m$ | FP32 | 4 | 动量 |
| $v$ | FP32 | 4 | 二阶矩 |

于是：

$$
K = 4+4+4 = 12
$$

参数按 $2$ 字节记、优化器按 $K=12$ 记时，**每个参数上优化器这块是参数的 6 倍字节**。  
所以 ZeRO 往往先切优化器状态。

若按本仓库 `activiations.md` 的 **全 FP32** 记账（作业那套）：

| 桶 | 字节 |
|----|------|
| 参数 | $4P$ |
| 梯度 | $4P$ |
| AdamW（$m$+$v$） | $8P$ |

此时优化器是参数的 **2 倍**——参数本身也按 4 字节计。  
**两种记账差在「参数算 2 还是 4、master 算不算进 $K$」**；本文标准算例用 mixed 那套 $2+2+K$。

---

## 3. 激活 $A$ 有多大？（对照用，ZeRO 公式不依赖它）

激活元素个数（`activiations.md`）：

$$
A
= N\bigl(8BLd + 2BhL^{2} + 4BL\,d_{\mathrm{ff}}\bigr)
+ BLd
+ 2BLV
$$

$$
M_{\mathrm{activations}} = 4A \quad\text{（FP32 作业约定）}
$$

$$
M_{\mathrm{optim}} = 8P \quad\text{（FP32，$m$+$v$）}
$$

$P$ 只跟结构走；$A$ 跟 $B$、$L$ 走。所以：

- 小模型 + 大 batch → $M_{\mathrm{activations}}$ 往往远大于 $M_{\mathrm{optim}}$；
- 大模型 + 小 batch → model states（尤其优化器）更显眼。

### TinyStories small（本仓库）

`configs/tinystories_small.yaml`：

- $V=10000,\ L=256,\ N=4,\ d=512,\ h=16$
- $d_{\mathrm{ff}}=\mathrm{compute\_d\_ff}(512)=1408$
- 常用 $B=64$

$$
P
= 2Vd + N\bigl(4d^{2}+3d\,d_{\mathrm{ff}}+2d\bigr)+d
= 23\,089\,664
$$

| 桶（全 FP32） | 约多少 |
|---------------|--------|
| 参数 $4P$ | $\approx 0.092\,\mathrm{GB}$ |
| 梯度 $4P$ | $\approx 0.092\,\mathrm{GB}$ |
| AdamW $8P$ | $\approx 0.185\,\mathrm{GB}$ |
| 激活 $4A$（$B=64$） | $\approx 6.04\,\mathrm{GB}$ |

同一时刻两边都可以很大；**ZeRO 管的是上面三行（$\Psi$ 侧），激活是下面那一行。**

### GPT-2 XL（`reports/flops.md`）

$P=1\,640\,452\,800$。全 FP32：

| 桶 | 约多少 |
|----|--------|
| 参数+梯度+AdamW $=$ $16P$ | $\approx 26.2\,\mathrm{GB}$ |
| 其中仅 AdamW $8P$ | $\approx 13.1\,\mathrm{GB}$ |
| 激活 $B=1,L=1024$ | $\approx 16.4\,\mathrm{GB}$ |
| 激活 $B=4$ | $\approx 65.5\,\mathrm{GB}$ |

ZeRO 要解决的问题是：**多卡数据并行时，每张卡还要不要各存一整份 model states（$\Psi$ 侧）。**

---

## 4. 普通数据并行（Baseline）：每卡一份完整 $\Psi$ 状态

### 4.1 数据并行在干什么

$N_d$ 张卡：

1. 每张卡有一份 **完整模型**；
2. 每张卡吃不同的数据 micro-batch；
3. `backward` 后各卡梯度不同，做 **all-reduce（平均）**，让每张卡的 `grad` 变成全局平均；
4. 每张卡用同一套平均梯度各自 `optimizer.step()`。

通信对象主要是 **梯度**；参数和优化器状态默认 **每卡一份完整拷贝**。

### 4.2 Baseline 公式

每张卡存：

$$
\underbrace{2\Psi}_{\text{FP16 params}}
+
\underbrace{2\Psi}_{\text{FP16 grads}}
+
\underbrace{K\Psi}_{\text{optim}}
=
(2+2+K)\Psi
$$

标准算例：$\Psi=7.5\mathrm{B},\ K=12$：

$$
16 \times 7.5\times 10^9\ \text{bytes}
= 120\times 10^9\ \text{bytes}
= 120\,\mathrm{GB}
$$

**与 $N_d$ 无关**：加到 64 卡，每卡 model states 仍是 120GB——吞吐上去了，单卡仍要装得下整份状态。

本仓库 TinyStories 在 Baseline、mixed 记账下：

$$
16P \approx 0.37\,\mathrm{GB}
$$

作业模型远小于 $7.5\mathrm{B}$ 那个量级。

---

## 5. 核心点子：切开 $\Psi$ 侧的冗余 + reduce-scatter 等价

核心思路：

> 把贵的 model states 切开；通信上利用 all-reduce ≡ reduce-scatter + all-gather。

两层意思：

1. **切：** 第 $i$ 张卡只常驻第 $i$ 片优化器状态 / 梯度 / 参数（按档位逐步加深）。  
2. **通信：** 普通 DP 对梯度做 all-reduce；all-reduce 在信息上等价于 **reduce-scatter 再 all-gather**（见 `misc/allreduce_reducescatter_allgather.md`）。  
   ZeRO 让「每卡最终只留下自己那一片」，从而少存完整副本。

下面三档：先切优化器状态，再切梯度，最后切参数——**全程都是对 $\Psi$ 分区。**

---

## 6. ZeRO-1：$P_{os}$（只切 Optimizer States）

### 6.1 每卡常驻什么

- 参数、梯度：每卡完整一份；
- 优化器状态：切成 $N_d$ 段，第 $i$ 卡只留第 $i$ 段。

### 6.2 公式

$$
2\Psi + 2\Psi + \frac{K\Psi}{N_d}
=
2\Psi + 2\Psi + \frac{K}{N_d}\Psi
$$

标准算例：

$$
\bigl(4 + 12/64\bigr)\times 7.5\mathrm{B}
=
4.1875\times 7.5\mathrm{B}
\approx 31.4\,\mathrm{GB}
$$

### 6.3 一步里发生了什么

- 每卡仍有完整 `p` 和完整 `p.grad`（all-reduce 之后梯度一致）；
- `state[p]["m"]`、`state[p]["v"]`（以及 FP32 master）**按参数下标分区**：  
  GPU 0 管 $[0,\Psi/N_d)$，GPU 1 管下一段，……
- 更新完自己负责的那一段参数后，再 **all-gather / 广播** 新参数，保证下一轮前向每卡仍有完整权重。

优化器状态从 $K\Psi$ 变成 $K\Psi/N_d$：120GB → 31.4GB。

---

## 7. ZeRO-2：$P_{os+g}$（再切 Gradients）

### 7.1 每卡常驻什么

- 参数：完整一份；
- 梯度、优化器状态：都切成 $N_d$ 段。

### 7.2 公式

$$
2\Psi + \frac{(2+K)\Psi}{N_d}
$$

标准算例：

$$
\bigl(2 + 14/64\bigr)\times 7.5\mathrm{B}
=
2.21875\times 7.5\mathrm{B}
\approx 16.6\,\mathrm{GB}
$$

### 7.3 梯度为什么也能按 $\Psi$ 切？

更新时 GPU $i$ 只负责第 $i$ 片参数的 Adam，因此它 **只需要第 $i$ 片梯度**。

- 用 **reduce-scatter**：每卡最终只拿到自己那一片段的平均梯度；
- 梯度显存从 $2\Psi$ 降到 $2\Psi/N_d$。

参数仍整份：每卡还要自己跑完整前向/反传。

---

## 8. ZeRO-3：$P_{os+g+p}$（参数也切）

### 8.1 每卡常驻什么

参数、梯度、优化器状态都切开——每卡只有 $1/N_d$。

### 8.2 公式

$$
\frac{(2+2+K)\Psi}{N_d}
=
\frac{16\Psi}{N_d}
$$

标准算例：

$$
120\,\mathrm{GB}/64 = 1.875\,\mathrm{GB} \approx 1.9\,\mathrm{GB}
$$

### 8.3 前向时临时凑齐参数

1. all-gather 当前层（或当前片）需要的参数；
2. 算完前向/反传，丢掉自己不拥有的参数副本；
3. 梯度按分片 reduce-scatter；
4. 只在自己拥有的参数分片上跑 Adam。

显存最省，通信量最大（反复 gather 参数）。  
120GB → 1.9GB，靠的是把 $\Psi$ 侧冗余副本砍光。

---

## 9. 四档对照（全是 $\Psi$ 的式子）

$\Psi=7.5\mathrm{B},\ K=12,\ N_d=64$：

| 档位 | 每卡存什么 | 公式（字节） | 每卡显存 |
|------|------------|--------------|----------|
| Baseline | 参数+梯度+优化器全份 | $(2+2+K)\Psi$ | 120 GB |
| $P_{os}$（ZeRO-1） | 参数+梯度全份，优化器切开 | $2\Psi+2\Psi+K\Psi/N_d$ | 31.4 GB |
| $P_{os+g}$（ZeRO-2） | 参数全份，梯度+优化器切开 | $2\Psi+(2+K)\Psi/N_d$ | 16.6 GB |
| $P_{os+g+p}$（ZeRO-3） | 三者全切开 | $(2+2+K)\Psi/N_d$ | 1.9 GB |

同一套公式套到本仓库 GPT-2 XL（$\Psi\approx 1.64\mathrm{B}$）：Baseline mixed 约 $16\Psi\approx 26.2\,\mathrm{GB}$；ZeRO-3、$N_d=64$ 时 model states 约 $26.2/64\approx 0.41\,\mathrm{GB}$（仍要另加激活）。

---

## 10. 和本仓库代码的对应

### 10.1 参数 $\Psi$（即 $P$）

$$
P
= 2Vd
+ N\bigl(4d^{2} + 3d\,d_{\mathrm{ff}} + 2d\bigr)
+ d
$$

TinyStories small：$P=23\,089\,664$。  
GPT-2 XL 配置：$P=1\,640\,452\,800$。

### 10.2 梯度

`loss.backward()` → 每个 `p.grad` 与 `p` 同形状。  
Baseline / ZeRO-1：通信后常驻完整 grad；ZeRO-2/3：常驻分片。

### 10.3 优化器状态

AdamW 的 `state[p]` 里至少有与 `p` 同形状的 $m$、$v$。  
$K=12$ 再计入 FP32 master 等。

### 10.4 激活

$Q\in\mathbb{R}^{B\times L\times d}$、`logits`$\in\mathbb{R}^{B\times L\times V}$ 等计入 $A$。  
若 $B$ 很大，$M_{\mathrm{activations}}$ 仍可能打满 GPU；那要用更小 batch、activation checkpointing、序列并行等，和 ZeRO 切 $\Psi$ 是并行的两条线。

---

## 11. 收束

| 问题 | 答案 |
|------|------|
| 本文在讲什么？ | 多卡时如何按 $\Psi$ 切开并少存 model states（参数+梯度+优化器） |
| state 是什么？ | parameters + gradients + optimizer states |
| 优化器状态为什么贵？ | 每参数 $K=12$ 字节（master+$m$+$v$），相对 FP16 参数约 6 倍 |
| Baseline → ZeRO-3？ | 120 → 31.4 → 16.6 → 1.9 GB（标准算例） |
| 激活呢？ | 另按 $A(B,L,\ldots)$ 算；ZeRO 公式里没有 $A$ |

- **Baseline：** 每卡完整参数 + 梯度 + 优化器状态。  
- **ZeRO-1：** 只切优化器状态。  
- **ZeRO-2：** 切优化器状态 + 梯度。  
- **ZeRO-3：** 切参数 + 梯度 + 优化器状态。  
- 优化器按 $\Psi$ 很贵，激活按 $B,L$ 也可以很大——**两笔账；本文只处理 $\Psi$ 那一笔。**

---
---

# 附录：四张卡上到底发生了什么（All-Reduce → ZeRO）

前面用字节和分档把「切参数 / 梯度 / 优化器」说完了。下面换一种讲法：  
**固定 4 张 GPU、4 份数据、一个只有 4 个数字的玩具参数**，把  
`p` / `p.grad` / Adam 的 `m,v` / All-Reduce / Reduce-Scatter / All-Gather  
在每张卡上长什么样、谁更新谁，从头走一遍。

和上文的衔接（很快带过，不断裂）：

| 上文概念 | 附录里对应什么 |
|----------|----------------|
| Parameters | 每卡上的 `w = [w0,w1,w2,w3]` |
| Gradients | 每卡上的 `g = w.grad` |
| Optimizer states | 每卡上的 `m`、`v`（Adam） |
| Baseline | 四卡各存完整 `w,g,m,v`，用 All-Reduce 同步梯度 |
| ZeRO-1/2/3 | 四卡开始只常驻自己那一「格」的优化器 / 梯度 / 参数 |

真实 `TransformerLM` 只是把 4 个数字换成几千万个；**分工和通信逻辑一样。**

---

## A. 玩具舞台：4 卡 × 4 个参数

### A.1 参数长什么样

假装整个模型就一个长度为 4 的向量（你可以把它想成某层 `Linear` 权重摊平后的前 4 格）：

```text
w = [w0, w1, w2, w3]
```

四张卡启动时 **参数相同**（从同一份初始化拷过来）：

```text
GPU0 / GPU1 / GPU2 / GPU3 上都是：
  w = [1.0, 1.0, 1.0, 1.0]
```

### A.2 数据怎么分

全局有一大堆训练样本。这一步把它们切成 4 份（data parallel）：

```text
GPU0 ← batch0
GPU1 ← batch1
GPU2 ← batch2
GPU3 ← batch3
```

每张卡只在自己的 batch 上做前向 + 反传。  
（激活张量每卡自己有一份，跟 ZeRO 切参数无关，这里不展开。）

### A.3 梯度长什么样

`loss.backward()` 之后，每个参数格子有一个梯度。玩具里写成：

```text
g = [g0, g1, g2, g3]    # 就是代码里的 w.grad，和 w 同形状
```

因为 batch 不同，**四卡本地梯度一开始不一样**，例如：

```text
GPU0:  g = [0.4, 0.4, 0.4, 0.4]
GPU1:  g = [0.8, 0.8, 0.8, 0.8]
GPU2:  g = [1.2, 1.2, 1.2, 1.2]
GPU3:  g = [1.6, 1.6, 1.6, 1.6]
```

（数字故意简单；真实训练每个位置都会不同。）

### A.4 Adam 的 `m`、`v` 长什么样

和本仓库 AdamW 一样：每个参数格子配一对动量。第一步之前通常是全 0：

```text
m = [0, 0, 0, 0]    # state[w]["m"]，与 w 同形状
v = [0, 0, 0, 0]    # state[w]["v"]，与 w 同形状
```

一步更新（忽略 bias correction / weight decay，只看骨架）：

```text
m ← β1 * m + (1-β1) * g
v ← β2 * v + (1-β2) * (g * g)      # 逐元素平方
w ← w - lr * m / (sqrt(v) + eps)
```

**要点：** `m`、`v` 和 `w`、`g` 都是「4 格」；优化器状态贵，是因为你多存了两整份（再加高精度 master 就更贵）。

---

## B. 通信三件套：All-Reduce = Reduce-Scatter + All-Gather

数据并行要先让大家的梯度变成 **同一份全局平均**，否则四卡会往四个方向更新。  
集体通信就三种姿势，关系是：

```text
All-Reduce  ≡  Reduce-Scatter  +  All-Gather
```

下面用「每卡一个长度为 4 的向量」把三种姿势钉死。  
（数字与 `misc/allreduce_reducescatter_allgather.md` 一致。）

### B.1 起点：四卡各有一份本地向量

```text
GPU0:  [0, 1, 2, 3]
GPU1:  [1, 2, 3, 4]
GPU2:  [2, 3, 4, 5]
GPU3:  [3, 4, 5, 6]
```

想得到的「按位置求和」是：

```text
位置0: 0+1+2+3 = 6
位置1: 1+2+3+4 = 10
位置2: 2+3+4+5 = 14
位置3: 3+4+5+6 = 18
→ 全局和向量 [6, 10, 14, 18]
```

### B.2 Reduce-Scatter：求和，但每人只留一段

```text
Reduce  = 同一位置跨卡加总
Scatter = 加总结果切开，每人拿走自己负责的那一格
```

做完之后：

```text
GPU0 只拿到位置0 → [6]
GPU1 只拿到位置1 → [10]
GPU2 只拿到位置2 → [14]
GPU3 只拿到位置3 → [18]
```

**没有一张卡持有完整 [6,10,14,18]。**  
这也是 ZeRO-2/3「梯度切开存」时喜欢的通信形态。

### B.3 All-Gather：把各人那一段拼回完整向量，且人人都有

接上一步：

```text
GPU0 有 [6]，GPU1 有 [10]，GPU2 有 [14]，GPU3 有 [18]
        │
        ▼  All-Gather
四张卡全都变成：
  [6, 10, 14, 18]
```

### B.4 All-Reduce：一步到位，人人都有完整全局结果

对同一组输入直接 All-Reduce（SUM）后：

```text
GPU0 / GPU1 / GPU2 / GPU3 全都是：
  [6, 10, 14, 18]
```

实现上常见路径就是先 Reduce-Scatter、再 All-Gather，所以：

```text
All-Reduce  =  Reduce-Scatter  +  All-Gather
```

训练里还经常要 **平均**：SUM 之后每人再 `/ 4`，或直接用 `AVG` 的 All-Reduce。  
上面例子平均后是 `[1.5, 2.5, 3.5, 4.5]`。

**人话对照：**

| 原语 | 每卡最后手里有什么 |
|------|-------------------|
| Reduce-Scatter | 全局和的 **一块** |
| All-Gather | 把各块拼成 **完整向量，且人人一份** |
| All-Reduce | 直接 **人人一份完整全局和/平均** |

---

## C. Baseline 数据并行：四卡完整走一轮

这是「不加 ZeRO」的默认 DP。和上文 Baseline 同一逻辑。

### C.1 开始时每卡存什么（完整参数 + 梯度槽位 + 优化器）

```text
每张 GPU 上都有：
  w = [1, 1, 1, 1]          # 完整参数
  g = （还没有）             # 反传后才有
  m = [0, 0, 0, 0]          # 完整 Adam m
  v = [0, 0, 0, 0]          # 完整 Adam v
```

### C.2 各卡用自己的数据算本地梯度

```text
GPU0 + batch0 → g = [0.4, 0.4, 0.4, 0.4]
GPU1 + batch1 → g = [0.8, 0.8, 0.8, 0.8]
GPU2 + batch2 → g = [1.2, 1.2, 1.2, 1.2]
GPU3 + batch3 → g = [1.6, 1.6, 1.6, 1.6]
```

此时四卡的 `w` 仍相同，但 `g` 不同。

### C.3 All-Reduce（AVG）同步梯度

对 `g` 做 All-Reduce 平均。每个位置：

```text
(0.4 + 0.8 + 1.2 + 1.6) / 4 = 1.0
```

之后：

```text
GPU0 / GPU1 / GPU2 / GPU3 的 g 全都变成：
  g = [1.0, 1.0, 1.0, 1.0]
```

若拆开看通信，就是：

```text
1) Reduce-Scatter：每卡先只拿到「全局和」的一格
     GPU0←[4.0], GPU1←[4.0], GPU2←[4.0], GPU3←[4.0]   # 若先 SUM
2) All-Gather：拼回完整 [4,4,4,4]，再 /4 → [1,1,1,1]
   （或库直接做 AVG 的 All-Reduce，对外一行 dist.all_reduce）
```

**没有「先把所有梯度堆到 GPU0 再广播」这一步**——是每卡就地改自己的 `w.grad`。

### C.4 四卡各自用同一份 `g` 跑 Adam，更新完整 `w`

取 `β1=0.9, β2=0.999, lr=0.1, eps≈0` 做示意（数字只为看得清）：

```text
每张卡上（g 已是 [1,1,1,1]）：
  m ← 0.9*0 + 0.1*1  = [0.1, 0.1, 0.1, 0.1]
  v ← 0.999*0 + 0.001*1 = [0.001, 0.001, 0.001, 0.001]
  w ← 1 - 0.1 * 0.1 / sqrt(0.001)
    ≈ 1 - 0.1 * 3.162
    ≈ [0.684, 0.684, 0.684, 0.684]
```

因为输入的 `g` 相同、`m/v` 初值相同，**四卡更新后的 `w` 仍然一致**。  
下一轮继续：各吃各的 batch → All-Reduce 梯度 → 各自 Adam。

### C.5 Baseline 一句话

```text
数据切开（省的是算力重叠）
参数/梯度/优化器 每卡一整份（显存按 Ψ 重复 N_d 次）
通信：All-Reduce 梯度
```

这就是上文 120GB 那种「加卡也不减每卡 model states」的根。

---

## D. 同一舞台上的 ZeRO：谁负责哪一格？

参数只有 4 格、正好 4 张卡，约定：

```text
GPU0 负责下标 0（w0 / g0 / m0 / v0）
GPU1 负责下标 1
GPU2 负责下标 2
GPU3 负责下标 3
```

真实模型是「按参数块/层切片」，逻辑相同。

### D.1 ZeRO-1（只切优化器状态）：完整 `w`、完整 `g`，但 `m,v` 切开

**常驻显存：**

```text
         w（参数）        g（梯度）           m,v（优化器）
GPU0   [w0,w1,w2,w3]   [g0,g1,g2,g3]     只存 m0,v0
GPU1   [w0,w1,w2,w3]   [g0,g1,g2,g3]     只存 m1,v1
GPU2   [w0,w1,w2,w3]   [g0,g1,g2,g3]     只存 m2,v2
GPU3   [w0,w1,w2,w3]   [g0,g1,g2,g3]     只存 m3,v3
```

**一步流程：**

1. 各卡前向/反传（每卡完整 `w`）→ 本地完整 `g`  
2. **All-Reduce** `g` → 四卡都有相同的完整平均梯度  
3. **只有主人更新自己那一格：**  
   - GPU0 用 `g0` 更新 `m0,v0`，写出新的 `w0`  
   - GPU1 更新 `w1`，……  
4. **All-Gather 新参数**：把新的 `w0,w1,w2,w3` 拼回，四卡又都有完整 `w`

省的是优化器：`m,v` 从「每卡 4 格」变成「每卡 1 格」。

### D.2 ZeRO-2（再切梯度）：完整 `w`，但 `g` 和 `m,v` 都切开

Adam 更新时 GPU0 只要 `g0`，所以通信可以停在 Reduce-Scatter：

```text
         w（参数）           g（梯度）      m,v（优化器）
GPU0   完整 [w0..w3]      只常驻 g0      只存 m0,v0
GPU1   完整 [w0..w3]      只常驻 g1      只存 m1,v1
GPU2   完整 [w0..w3]      只常驻 g2      只存 m2,v2
GPU3   完整 [w0..w3]      只常驻 g3      只存 m3,v3
```

**一步流程：**

1. 各卡反传 → 先有本地完整 `g`（临时）  
2. **Reduce-Scatter** 平均梯度 → 每卡只留下自己那一格平均 `g_i`（完整 `g` 可以丢掉）  
3. 主人用 `g_i` 更新 `m_i,v_i` 和 `w_i`  
4. **All-Gather** 新 `w`，保证下一轮前向人人有完整参数

这里用上了「All-Reduce = Reduce-Scatter + All-Gather」里的前半段：  
**只做 Reduce-Scatter 就够更新**；后半段 All-Gather 用在参数上，而不是再把完整梯度 gather 回来。

### D.3 ZeRO-3（参数也切）：`w / g / m,v` 全都只留自己那一格

ZeRO-3 和 ZeRO-2 的差别只有一点：**连 `w` 也切开常驻。**  
下面用同一组 `w0..w3 / g0..g3 / m0..v3`，把一步训练从头走到尾。

#### 时刻 0：每卡硬盘里常驻的只有自己那一格

```text
GPU0 常驻：  w0 ,  m0 ,  v0
GPU1 常驻：  w1 ,  m1 ,  v1
GPU2 常驻：  w2 ,  m2 ,  v2
GPU3 常驻：  w3 ,  m3 ,  v3
```

数字接续 Baseline 那套初值，方便对照：

```text
w0=w1=w2=w3 = 1.0
m0=m1=m2=m3 = 0
v0=v1=v2=v3 = 0
```

此时 **没有任何一张卡** 手里有完整 `[w0,w1,w2,w3]`。  
前向要用完整 `w`，所以先通信。

#### 时刻 1：All-Gather 参数 —— 每人把自己的 `wi` 发给所有人

```text
GPU0 拿出 w0=1.0
GPU1 拿出 w1=1.0
GPU2 拿出 w2=1.0
GPU3 拿出 w3=1.0
        │
        ▼  All-Gather
四张卡各自得到一份完整副本：
  GPU0:  [w0,w1,w2,w3] = [1,1,1,1]
  GPU1:  [w0,w1,w2,w3] = [1,1,1,1]
  GPU2:  [w0,w1,w2,w3] = [1,1,1,1]
  GPU3:  [w0,w1,w2,w3] = [1,1,1,1]
```

注意：`w1` 对 GPU0 来说是 **借来的**；`w0` 对 GPU1 来说也是借来的。  
常驻归属没变：GPU0 仍然只「拥有」`w0`（以及 `m0,v0`）。

#### 时刻 2：各卡用完整 `w` + 自己的 batch 做前向、反传

```text
GPU0 + batch0 → 本地梯度  g^(0) = [0.4, 0.4, 0.4, 0.4]
GPU1 + batch1 → 本地梯度  g^(1) = [0.8, 0.8, 0.8, 0.8]
GPU2 + batch2 → 本地梯度  g^(2) = [1.2, 1.2, 1.2, 1.2]
GPU3 + batch3 → 本地梯度  g^(3) = [1.6, 1.6, 1.6, 1.6]
```

写成「每张卡对四个参数格各有一个本地导数」：

```text
          对 w0 的本地导    对 w1    对 w2    对 w3
GPU0:        0.4             0.4      0.4      0.4
GPU1:        0.8             0.8      0.8      0.8
GPU2:        1.2             1.2      1.2      1.2
GPU3:        1.6             1.6      1.6      1.6
```

#### 时刻 3：Reduce-Scatter 梯度 —— 按列求平均，结果只送给该列的主人

对 `w0` 这一列（只有 GPU0 需要留下）：

```text
g0 = (0.4 + 0.8 + 1.2 + 1.6) / 4 = 1.0   → 只放进 GPU0
```

对 `w1` 列 → 只放进 GPU1；`w2` → GPU2；`w3` → GPU3。同样都是 `1.0`。

做完之后：

```text
GPU0 手里只留  g0 = 1.0
GPU1 手里只留  g1 = 1.0
GPU2 手里只留  g2 = 1.0
GPU3 手里只留  g3 = 1.0
```

没有一张卡再保留完整 `[g0,g1,g2,g3]`。  
这就是 ZeRO-2 里已经用过的 Reduce-Scatter；ZeRO-3 梯度侧完全一样。

#### 时刻 4：丢掉借来的参数，只留自己拥有的那一格

前向已经结束，GPU0 不再需要 `w1,w2,w3`：

```text
GPU0 删除借来的 w1,w2,w3，只留自己的 w0
GPU1 删除借来的 w0,w2,w3，只留自己的 w1
GPU2 删除借来的 w0,w1,w3，只留自己的 w2
GPU3 删除借来的 w0,w1,w2，只留自己的 w3
```

此刻每卡显存回到：

```text
GPU0:  w0,  g0=1.0,  m0,  v0
GPU1:  w1,  g1=1.0,  m1,  v1
GPU2:  w2,  g2=1.0,  m2,  v2
GPU3:  w3,  g3=1.0,  m3,  v3
```

#### 时刻 5：各卡只更新自己拥有的那一格（Adam）

仍用 `β1=0.9, β2=0.999, lr=0.1`：

```text
GPU0（只动下标 0）：
  m0 ← 0.9*0 + 0.1*g0 = 0.1
  v0 ← 0.999*0 + 0.001*(g0^2) = 0.001
  w0 ← 1 - 0.1 * m0 / sqrt(v0) ≈ 0.684

GPU1（只动下标 1）：同样得到 m1=0.1, v1=0.001, w1≈0.684
GPU2（只动下标 2）：同样得到 m2,v2,w2
GPU3（只动下标 3）：同样得到 m3,v3,w3
```

**没有任何一张卡去改别人的 `wi`。**  
例如 GPU0 不会写 `w1`；`w1` 的新值只在 GPU1 上算出来。

#### 时刻 6：一步结束时的常驻状态

```text
GPU0:  w0≈0.684 ,  m0=0.1 ,  v0=0.001
GPU1:  w1≈0.684 ,  m1=0.1 ,  v1=0.001
GPU2:  w2≈0.684 ,  m2=0.1 ,  v2=0.001
GPU3:  w3≈0.684 ,  m3=0.1 ,  v3=0.001
```

下一 step 又从时刻 1 开始：再次 All-Gather，把新的 `w0..w3` 拼成完整 `[0.684,0.684,0.684,0.684]` 给四卡前向用。

#### 串起来看（ZeRO-3 一步 = 六拍）

```text
常驻只有自己的 wi,mi,vi
  → All-Gather：四卡都暂时拥有 [w0,w1,w2,w3]
  → 各卡用自己的 batch 算出本地完整梯度 g^(0)..g^(3)
  → Reduce-Scatter：GPU0 只留 g0，GPU1 只留 g1，…
  → 删掉借来的 w（GPU0 删 w1,w2,w3，…）
  → GPU0 用 g0 更新 m0,v0,w0；GPU1 用 g1 更新 m1,v1,w1；…
```

和 ZeRO-2 比：ZeRO-2 的 `w` 一直完整挂在每卡上，所以不需要时刻 1 / 时刻 4；  
ZeRO-3 用这两次「借参数 / 还参数」换来更小的常驻显存（标准算例里到约 1.9GB），代价是每步都要再 All-Gather 一次 `w0..w3`。

---

## E. 四卡对照总表（贴在显示器旁边看）

同一时刻「谁常驻什么」：

| | Baseline | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|--|----------|--------|--------|--------|
| GPU$i$ 的 `w` | 完整 4 格 | 完整 4 格 | 完整 4 格 | **只第 $i$ 格** |
| GPU$i$ 的 `g` | 完整 4 格（All-Reduce 后） | 完整 4 格 | **只第 $i$ 格** | **只第 $i$ 格** |
| GPU$i$ 的 `m,v` | 完整 4 格 | **只第 $i$ 格** | **只第 $i$ 格** | **只第 $i$ 格** |
| 梯度通信 | All-Reduce | All-Reduce | Reduce-Scatter | Reduce-Scatter |
| 参数通信 | 通常不需要 | 更新后 All-Gather | 更新后 All-Gather | 用时 All-Gather、用完可丢 |

同一轮「算法顺序」压缩版：

```text
Baseline:
  各卡 forward/backward
  → All-Reduce(g)
  → 各卡用完整 g 更新完整 w,m,v

ZeRO-1:
  各卡 forward/backward
  → All-Reduce(g)
  → 各卡只更新自己那一格的 w,m,v
  → All-Gather(w)

ZeRO-2:
  各卡 forward/backward
  → Reduce-Scatter(g)     # 每卡只留自己的 g_i
  → 各卡更新自己的 w_i,m_i,v_i
  → All-Gather(w)

ZeRO-3:
  All-Gather：每人得到 [w0,w1,w2,w3]
  → 各卡 forward/backward → 本地完整 g
  → Reduce-Scatter：GPU0 只留 g0，GPU1 只留 g1，…
  → 删掉借来的 w（GPU0 只留 w0，…）
  → GPU0 用 g0 更新 m0,v0,w0；其余卡同理
```

---

## F. 和真实训练循环对齐（你们的 `train.py` 骨架）

单卡时你们熟悉的是：

```text
zero_grad → forward → loss → backward → (clip) → optimizer.step
```

四卡 Baseline 只是在 `backward` 和 `step` 之间多一步：

```text
zero_grad
→ forward（本卡 batch）
→ loss / backward          # 写出本卡 w.grad
→ All-Reduce(w.grad)       # 四卡梯度变成同一个平均
→ optimizer.step()         # 读 w.grad，改 w 和 state[w] 的 m,v
```

ZeRO 不改「要算梯度、要用 Adam」这件事，改的是：

```text
哪些格子的 m,v（以及 g、w）常驻在本卡
以及通信用 All-Reduce 还是只做到 Reduce-Scatter
```

把玩具里的 4 格换成 TinyStories 的两千万参数，或标准算例的 $7.5\mathrm{B}$，只是格子变多；  
**GPU0..3 的分工、All-Reduce = Reduce-Scatter + All-Gather、谁更新哪一段**——就是这一附录讲的全部内容。

---

## G. 万亿参数时：前向那一刻显存里到底要装什么？

问题是这样的：

> ZeRO-3 常驻可以只留 `1/N_d` 的参数；但 All-Gather 之后，前向要用权重，那一刻是不是仍得在**单张卡**上放下**全部**参数？一万亿、两万亿参数的模型，单卡装不下整模，那一步会不会直接爆？

### 答案（就事论事）

**前向确实需要「当前这一层要用的权重」在本卡上算矩阵乘；但那不是「整模一次性灌进单卡」。**

附录玩具里只有 4 个数字，所以 All-Gather 一次就得到了完整 `[w0,w1,w2,w3]`——看起来像「整模都来了」。  
真实大模型里，ZeRO-3 的 gather **按层（或按参数块）做**：

```text
算第 ℓ 层之前：
  All-Gather 只要第 ℓ 层的权重分片 → 本卡暂时有完整「第 ℓ 层」
  用第 ℓ 层做完前向（或反传）
  立刻丢掉第 ℓ 层里不属于自己的分片

再算第 ℓ+1 层：
  再 All-Gather 第 ℓ+1 层
  ……
```

因此单卡峰值大致是：

```text
常驻：自己拥有的参数分片 + 对应的 m,v（再加激活等）
峰值额外：当前正在算的那一层（或那一块）的完整权重
```

**峰值跟「一整模」脱钩，跟「一层有多大」挂钩。**  
一层塞不进单卡，再往下拆这一层（见下）。

### 和玩具例子对齐（把 4 格想成 4 层）

若 `w0,w1,w2,w3` 分别是四层的权重，ZeRO-3 实际节奏是：

```text
All-Gather 只拉 w0 → 四卡都有完整第 0 层 → 算完 → 非主人丢掉 w0 的副本
All-Gather 只拉 w1 → 算第 1 层 → 丢掉
All-Gather 只拉 w2 → ……
All-Gather 只拉 w3 → ……
```

不是：

```text
一次 All-Gather 把 w0+w1+w2+w3 整模堆进每张卡再开算
```

玩具写成一次 gather，只是因为 4 个数太小，合成一步更好看；**工程上的 ZeRO-3 是一层一层借、一层一层还。**

### 万亿级还要再加什么

当「单层」也大于单卡显存时，光靠 ZeRO-3（数据并行 + 按参数分片）不够。  
训练超大模型时，通常叠几类并行，让**任何时刻单卡都只看见模型的一块**：

| 手段 | 单卡看见什么 |
|------|----------------|
| ZeRO-3（参数分片） | 常驻 `1/N_d` 参数；算某层时暂时多出该层完整权重 |
| 张量并行（Tensor Parallel） | 同一层矩阵切列/切行，多卡各算一块再通信 |
| 流水线并行（Pipeline Parallel） | 不同卡常驻不同层，激活在卡间传递 |

Anthropic / OpenAI 量级的训练，是这些东西（再加优化器分片、激活重计算、专家并行等）**叠在一起**：  
**没有「某一时刻单卡必须盛下全部一万亿参数」这个前提。**  
前提是：单卡盛得下「当前正在算的那一块」（一层的一块、或一层里张量并行后的一块），算完就释放或传走。

### 一句话

```text
ZeRO-3：省的是「常驻整模」；
前向：要的是「当前层完整权重（可再被张量并行切开）」；
万亿模：靠「按层 gather + 张量/流水线并行」，保证单卡从不需要放下整个 Ψ。
```

---

## H. 大模型训练里最常见的五种并行

经典教科书里常先讲三种：**数据并行、张量并行、流水线并行**。  
今天训稠密大模型、长上下文、MoE，实务上还会经常用到另外两种：**序列（上下文）并行、专家并行**。  

下面按「最常见、都在用」列 **5 种**。  
ZeRO / FSDP 算数据并行的加强版（参数/梯度/优化器怎么存），不单开第六种。

约定一个小舞台，方便举例：

```text
模型：4 层 Transformer（层号 0,1,2,3）
每层里有大矩阵乘法（可想成 QKV / 输出投影 / FFN）
序列长度 L（token 数），batch 里有多条样本
GPU 编号：0,1,2,3,…（具体用几张看例子）
```

---

### 1. 数据并行（Data Parallelism, DP）

**切什么：** 切 **数据**。模型（至少逻辑上）每份副本结构相同。  
**谁算什么：** 每张卡拿不同 micro-batch，各算各的前向/反传，再同步梯度（All-Reduce；ZeRO 则改成分片存 + Reduce-Scatter 等）。

**例子（4 卡）：**

```text
全局 batch = 256 条样本
GPU0 ← 样本 0..63
GPU1 ← 样本 64..127
GPU2 ← 样本 128..191
GPU3 ← 样本 192..255

四卡各自算出一份梯度 → 平均成同一份 → 再更新参数
```

本文前面 Baseline / ZeRO 讲的都是这一族。  
**省的是时间（吞吐）；经典 DP 不省「每卡一份完整模型状态」的显存，ZeRO 才在 DP 里继续省显存。**

---

### 2. 张量并行（Tensor Parallelism, TP）

**切什么：** 切 **同一层里的大矩阵**（一行/一列切开），多卡合力算这一层。  
**谁算什么：** 例如把 `W` 按列切成 `W_a | W_b`，GPU0 算 `X @ W_a`，GPU1 算 `X @ W_b`，再通信拼成完整输出（或下一层需要的布局）。

**例子（2 卡张量并行，只看一层线性层）：**

```text
输入 X 形状 [B, L, d]，d=4096
权重 W 形状 [4096, 4096]

TP=2：
  GPU0 只存并计算 W 的左半：W[:, 0:2048]
  GPU1 只存并计算 W 的右半：W[:, 2048:4096]

GPU0 得到输出的前 2048 维，GPU1 得到后 2048 维
需要完整向量时：一次 All-Gather（或按实现做 Reduce-Scatter）
```

注意力里也可以切头：GPU0 算一部分 head，GPU1 算另一部分 head。  
**单层太大、塞不进一张卡时，TP 是直接手段。** 卡间通信密、通常要求高速互联（同机 NVLink 等）。

---

### 3. 流水线并行（Pipeline Parallelism, PP）

**切什么：** 切 **层**。不同卡常驻模型的不同深度。  
**谁算什么：** 激活像流水一样在卡间传递：算完第 0–1 层交给下一张卡算第 2–3 层。

**例子（4 层模型，PP=4）：**

```text
GPU0 常驻 layer0
GPU1 常驻 layer1
GPU2 常驻 layer2
GPU3 常驻 layer3

一条样本前向：
  GPU0: x → layer0 → 把激活传给 GPU1
  GPU1: → layer1 → 传给 GPU2
  GPU2: → layer2 → 传给 GPU3
  GPU3: → layer3 → logits / loss

反传时激活/梯度反向传回来
```

为了不让后面的卡空转，通常把 batch 切成多个 micro-batch 填满流水线（1F1B 等调度）。  
**层数很多、整模太深时用 PP；通信是「层与层之间的激活」，比 TP 稀疏，但要处理好气泡（bubble）。**

---

### 4. 序列并行 / 上下文并行（Sequence / Context Parallelism）

**切什么：** 切 **序列长度 L**（或长上下文的 token 段）。  
**谁算什么：** 同一层里，不同卡负责序列的不同区段；注意力要跨段看上下文时再通信（例如环状传 KV）。

**例子（序列长 L=8k，4 卡序列并行）：**

```text
一条样本的 8192 个 token 切开：
  GPU0 负责 token 0..2047 的激活/部分注意力计算
  GPU1 负责 token 2048..4095
  GPU2 负责 token 4096..6143
  GPU3 负责 token 6144..8191

算注意力时：每卡需要其他段的 K/V（或等价信息）
→ 按实现 All-Gather、或 ring 传递 KV 块
```

和张量并行常一起出现：TP 切「隐藏维/头」，序列并行切「长度维」，避免 LayerNorm、dropout 等在 TP 下重复存整段激活。  
**上下文特别长（几十万 token）、激活按 L 爆显存时，这一类几乎必用。**

---

### 5. 专家并行（Expert Parallelism, EP）——主要给 MoE

**切什么：** 切 **MoE 里的专家网络**（不同专家放在不同卡上）。  
**谁算什么：** 路由器决定每个 token 去哪些专家；token 被派送到持有该专家的卡上算 FFN，再送回。

**例子（8 专家，4 卡，每卡 2 个专家）：**

```text
GPU0: Expert0, Expert1
GPU1: Expert2, Expert3
GPU2: Expert4, Expert5
GPU3: Expert6, Expert7

某个 token 路由到 Expert5：
  激活从当前卡发到 GPU2 → Expert5 算完 → 结果送回
```

稠密模型（你们作业里的 `TransformerLM`）没有「专家」，一般用不到 EP。  
**Mixtral、一大类万亿稀疏模型：总参数极大，但每个 token 只激活少数专家——EP 是标配。**

---

### 五张对照表

| 并行 | 切开的轴 | 典型目的 | 一句话例子 |
|------|----------|----------|------------|
| 数据并行 DP | batch / 样本 | 提高吞吐；ZeRO 再省状态显存 | 4 卡各吃 64 条，梯度平均 |
| 张量并行 TP | 一层内的矩阵维 / 头 | 单层太大塞不进一卡 | 4096 维权重左右劈给 2 卡 |
| 流水线并行 PP | 层号（深度） | 模型太深，按深度分卡 | 4 卡各挂 1 层，激活顺次传 |
| 序列/上下文并行 | 序列长度 L | 超长上下文，激活按 L 爆炸 | 8k token 切成 4 段分 4 卡 |
| 专家并行 EP | MoE 专家编号 | 稀疏超大模型 | 8 专家分 4 卡，token 按路由跑腿 |

### 它们怎么叠在一起（诚实版）

实务里很少只用一种：

```text
常见组合（示意）：
  DP  ×  TP  ×  PP              ← 稠密大模型的「老三样 + 数据」
  再加上 Sequence/Context Parallel ← 上下文变长时
  再叠加 Expert Parallel          ← 换成 MoE 时
```

例如：

```text
64 张卡可以这样分：
  PP = 4     → 深度切成 4 段
  TP = 8     → 每一段内部 8 卡一起算一层大矩阵
  DP = 2     → 上面这套「模型切片」再复制两份吃不同数据
  （长上下文再把 TP 组里加序列并行；
    MoE 再在 FFN 专家维加 EP）
```

**所以：不是「只有三种」；三种是底座，后面两种在长上下文和 MoE 上同样常见。**  
本文前面的 ZeRO，属于 **数据并行怎么存、怎么同步** 那一条线；张量/流水线/序列/专家是 **模型与序列怎么切开算** 的另外几条线，可以和 ZeRO 叠用。
