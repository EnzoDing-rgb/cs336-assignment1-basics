# ZeRO：数据并行里「模型状态」怎么切、为什么省显存

本文把一张 ZeRO 讲义图（Baseline → $P_{os}$ → $P_{os+g}$ → $P_{os+g+p}$）从零推到尾。  
**不引用外部论文；** 数字一边用图上的例子（$\Psi=7.5\mathrm{B}$、$N_d=64$、$K=12$），一边用本仓库 `TransformerLM` / TinyStories / GPT-2 XL 配置算一遍。

**你最容易混的一点先说清楚：**

> 这张图里的 **state ≠ activation**。  
> 图里切的是 **模型状态（model states）** = 参数 + 梯度 + 优化器状态。  
> **激活（activations）是另一桶显存**，图上根本没画。  
> 「优化器状态比参数大很多」是对的；「激活是不是也算进 state、是不是永远比 optimizer 大」——要分开说，见 §2、§8。

---

## 0. 符号表（全文统一）

| 符号 | 含义 |
|------|------|
| $\Psi$ | 可训练参数个数（元素个数）。图上 $\Psi=7.5\times 10^9$。本仓库 TinyStories small 约 $\Psi\approx 2.31\times 10^7$；`reports/flops.md` 的 GPT-2 XL 配置 $\Psi=1\,640\,452\,800$。 |
| $N_d$ | 数据并行 GPU 数（degree of data parallelism）。图上 $N_d=64$。 |
| $K$ | **每个参数**对应的优化器状态字节数（mixed-precision Adam 常见记账）。图上 $K=12$。 |
| $P$ / $\Psi$ | 同一件事：参数元素个数。本仓库 `activiations.md` 里用 $P$，图上用 $\Psi$。 |
| $B,L,d,h,N,V,d_{\mathrm{ff}}$ | batch、序列长、`d_model`、头数、层数、词表、FFN 宽（与 `activiations.md` 相同） |
| 蓝条 | Parameters（参数） |
| 橙条 | Gradients（梯度） |
| 绿条 | Optimizer states（优化器状态） |

字节约定（与图一致的 **mixed precision 记账**）：

| 东西 | 每参数多少字节 | 为什么 |
|------|----------------|--------|
| FP16 参数副本 | $2$ | 前向/反传常用半精度权重 |
| FP16 梯度 | $2$ | 反传得到的梯度也按半精度记 |
| 优化器状态 | $K=12$ | 见 §3：FP32 master + Adam 的 $m$、$v$ |

因此「一份完整模型状态」按图的公式是：

$$
(2 + 2 + K)\,\Psi = 16\Psi \quad\text{（字节）}
$$

---

## 1. 训练时 GPU 上到底有几桶东西？

一次 `train` 步（你们 `cs336_basics/train.py` 那套循环）里，显存至少有四类：

$$
M_{\mathrm{peak}}
\;\approx\;
M_{\mathrm{params}}
+ M_{\mathrm{grads}}
+ M_{\mathrm{optim}}
+ M_{\mathrm{activations}}
$$

（还有临时 buffer、碎片等，这里忽略。）

| 桶 | 是什么 | 形状跟谁走 | 本仓库落点 |
|----|--------|------------|------------|
| Parameters | 可训练权重 $W$、RMSNorm 的 $g$ 等 | 只跟模型结构有关，**没有** $B,L$ | `model.parameters()` |
| Gradients | 损失对每个参数的 $\partial L/\partial p$ | 与参数 **同形状** | `p.grad`（`loss.backward()` 写入） |
| Optimizer states | AdamW 为每个 $p$ 记的历史 | 与参数 **同形状** 的 $m$、$v$ 等 | `optimizer.state[p]` |
| Activations | 前向中间张量，反传还要用 | **有** $B,L$（甚至 $L^2$） | 计算图里保留的 $Q,K,V,\mathrm{logits},\ldots$ |

**ZeRO 这张图只动前三桶**（合称 **model states**）。  
第四桶 activation **不在图里**，也不被 ZeRO-1/2/3「按参数切片」直接切掉。

---

## 2. 先回答你的困惑：activation / state / optimizer 谁大？

### 2.1 图里的 “state” 是什么？

图标题写：*split up the expensive parts (**state**)*。  
这里的 state = **model states** = 蓝 + 橙 + 绿，也就是：

$$
\text{state}
=
\text{parameters}
+
\text{gradients}
+
\text{optimizer states}
$$

**不包含 activation。**

### 2.2 为什么 optimizer state 比 parameter「大很多」？

不是「元素个数更多」——Adam 的 $m$、$v$ 和参数 **一一对应、同形状**，元素个数都是 $\Psi$。  
大，是因为 **每个参数要多存好几份同形状的数，而且常用更高精度**：

用你们已经写过的 AdamW 语言（`misc/optimizer_state_and_groups.md`）：

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

参数只按 $2$ 字节记时，优化器状态按 $K=12$ 记——**每个参数上，优化器这块是参数的 6 倍字节**。  
所以图上会说「贵的部分」往往先是绿条。

若按本仓库 `activiations.md` 的 **全 FP32** 记账（作业那套）：

| 桶 | 字节 |
|----|------|
| 参数 | $4P$ |
| 梯度 | $4P$ |
| AdamW（只有 $m$+$v$，无另计 master） | $8P$ |

此时 optimizer 仍是参数的 **2 倍**，不是 6 倍——因为参数本身也按 4 字节计了。  
**两种记账差在「参数算 2 还是 4、master 算不算进 $K$」**；图用的是 mixed 那套 $2+2+K$。

### 2.3 activation 会不会比 optimizer 大很多？

**会，而且经常会——但那是另一桶，不是图上的绿条。**

Activation 的元素个数带 $B,L$，attention 分数还有 $L^2$ 项（见 `activiations.md`）：

$$
A
= N\bigl(8BLd + 2BhL^{2} + 4BL\,d_{\mathrm{ff}}\bigr)
+ BLd
+ 2BLV
$$

$$
M_{\mathrm{activations}} = 4A \quad\text{（FP32 作业约定）}
$$

而

$$
M_{\mathrm{optim}} = 8P \quad\text{（FP32，$m$+$v$）}
$$

$P$ **不随 batch 变**；$A$ **随 $B$、$L$ 涨**。所以：

- 小模型 + 大 batch → activation 轻松压过 optimizer；
- 大模型 + 小 batch → model states（尤其 optimizer）更显眼。

#### 用本仓库 TinyStories small 算一遍

`configs/tinystories_small.yaml`：

- $V=10000,\ L=256,\ N=4,\ d=512,\ h=16$
- $d_{\mathrm{ff}}=\mathrm{compute\_d\_ff}(512)=1408$（`8/3\cdot d` 再向上取到 64 倍数）
- 实验常用 $B=64$

参数个数：

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
| **激活 $4A$（$B=64$）** | **$\approx 6.04\,\mathrm{GB}$** |

这里 activation ≈ optimizer 的 **33 倍**。  
所以：**「激活很大」是真的；但它不是 ZeRO 图里那条绿的 optimizer state。**

#### 用 GPT-2 XL 配置感受「大模型」

`reports/flops.md`：$P=1\,640\,452\,800$。全 FP32 下：

| 桶 | 约多少 |
|----|--------|
| 参数+梯度+AdamW $=$ $16P$ | $\approx 26.2\,\mathrm{GB}$ |
| 其中仅 AdamW $8P$ | $\approx 13.1\,\mathrm{GB}$ |
| 激活 $B=1,L=1024$ | $\approx 16.4\,\mathrm{GB}$ |
| 激活 $B=4$ | $\approx 65.5\,\mathrm{GB}$ |

大模型上：**model states 已经很重**；batch 一大，activation 又会反超。  
ZeRO 解决的是：**多卡数据并行时，每张卡还要不要各存一整份 model states。**

---

## 3. 普通数据并行（Baseline）为什么显存炸？

### 3.1 数据并行在干什么（直觉）

$N_d$ 张卡：

1. 每张卡有一份 **完整模型**；
2. 每张卡吃不同的数据 micro-batch；
3. `backward` 后各卡梯度不同，做 **all-reduce（平均）**，让每张卡的 `grad` 变成全局平均；
4. 每张卡用同一套平均梯度各自 `optimizer.step()`。

通信对象主要是 **梯度**；参数和优化器状态默认 **每卡一份完整拷贝**。

### 3.2 Baseline 公式（图第一行）

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

图例：$\Psi=7.5\mathrm{B},\ K=12$：

$$
16 \times 7.5\times 10^9\ \text{bytes}
= 120\times 10^9\ \text{bytes}
= 120\,\mathrm{GB}
$$

**和 $N_d$ 无关**：你加到 64 卡，每卡仍然要 120GB 的 model states——只是吞吐上去了，**单卡装不下的模型还是装不下**。

本仓库 TinyStories 在 Baseline、mixed 记账下：

$$
16P \approx 0.37\,\mathrm{GB}
$$

小到完全不是瓶颈；图上的 7.5B 才是「单卡装不下」的典型量级。

---

## 4. 核心点子：把贵的状态切开 + reduce-scatter 等价

图上写的 core idea：

> split up the expensive parts (state) and use the reduce-scatter equivalence.

两层意思：

1. **切：** 不要每卡都存完整的绿 / 橙 / 蓝；第 $i$ 张卡只负责第 $i$ 片。  
2. **通信形态：** 普通 DP 对梯度做 all-reduce。  
   all-reduce 在信息上等价于 **reduce-scatter 再 all-gather**（你们 `misc/allreduce_*.md` 里写过这条分解）。  
   ZeRO 利用的是：既然本来就要通信，不如让「每卡最终只留下自己那一片梯度/参数」，少存冗余副本。

下面三档就是：先切绿，再切橙，最后连蓝也切。

---

## 5. ZeRO-1：$P_{os}$（只切 Optimizer States）

### 5.1 图上长什么样

- 蓝（参数）、橙（梯度）：每卡仍是 **整条**；
- 绿（优化器）：切成 $N_d$ 段，第 $i$ 卡只留第 $i$ 段。

### 5.2 公式

$$
2\Psi + 2\Psi + \frac{K\Psi}{N_d}
=
2\Psi + 2\Psi + \frac{K}{N_d}\Psi
$$

图例：

$$
\bigl(4 + 12/64\bigr)\times 7.5\mathrm{B}
=
4.1875\times 7.5\mathrm{B}
\approx 31.4\,\mathrm{GB}
$$

### 5.3 人话：一步里发生了什么

仍以 AdamW 直觉：

- 每卡还是有完整 `p` 和完整 `p.grad`（all-reduce 之后大家梯度一致）；
- 但 `state[p]["m"]`、`state[p]["v"]`（以及 FP32 master）**按参数下标分区**：  
  GPU 0 只更新参数切片 $[0,\Psi/N_d)$ 的优化器状态，GPU 1 管下一段，……
- 更新完自己负责的那一段参数后，再把新参数 **广播/all-gather** 回所有卡，保证下一轮前向每卡仍有完整权重。

省的是：**绿条从 $K\Psi$ 变成 $K\Psi/N_d$。**  
蓝和橙仍在，所以从 120GB → 31.4GB，降一大截，但还没到极限。

---

## 6. ZeRO-2：$P_{os+g}$（再切 Gradients）

### 6.1 图上长什么样

- 蓝：仍整条；
- 橙、绿：都切成 $N_d$ 段。

### 6.2 公式

$$
2\Psi + \frac{(2+K)\Psi}{N_d}
$$

图例：

$$
\bigl(2 + 14/64\bigr)\times 7.5\mathrm{B}
=
2.21875\times 7.5\mathrm{B}
\approx 16.6\,\mathrm{GB}
$$

### 6.3 为什么梯度也能切？

普通 DP：all-reduce 后 **每卡都有完整平均梯度**。  
但更新时，既然 GPU $i$ 只负责第 $i$ 片参数的 Adam 更新，它其实 **只需要第 $i$ 片梯度**。

于是：

- 用 **reduce-scatter**：每卡最终只拿到自己那一片段的平均梯度（不是完整 all-reduce 结果）；
- 橙条显存从 $2\Psi$ 降到 $2\Psi/N_d$。

参数（蓝）还要整份，因为每卡还要自己跑完整前向/反传。

---

## 7. ZeRO-3：$P_{os+g+p}$（参数也切）

### 7.1 图上长什么样

蓝、橙、绿 **全是细竖条**——每卡只有 $1/N_d$ 的参数、梯度、优化器状态。

### 7.2 公式

$$
\frac{(2+2+K)\Psi}{N_d}
=
\frac{16\Psi}{N_d}
$$

图例：

$$
120\,\mathrm{GB}/64 = 1.875\,\mathrm{GB} \approx 1.9\,\mathrm{GB}
$$

### 7.3 代价：前向时要临时把参数「凑齐」

每卡不再常驻完整模型。算某一层时大致是：

1. all-gather 这一层（或这一片）需要的参数；
2. 算完前向/反传，立刻丢掉不属于自己的参数副本；
3. 梯度仍按分片 reduce-scatter；
4. 只在自己拥有的参数分片上跑 Adam。

**显存最省，通信量最大**（反复 gather 参数）。  
图上从 120GB → 1.9GB，靠的就是「冗余副本几乎砍光」。

---

## 8. 把图上四行收成一张对照表

$\Psi=7.5\mathrm{B},\ K=12,\ N_d=64$：

| 档位 | 每卡存什么（直觉） | 公式（字节） | 图上数字 |
|------|-------------------|--------------|----------|
| Baseline | 蓝橙绿全份 | $(2+2+K)\Psi$ | 120 GB |
| $P_{os}$（ZeRO-1） | 蓝橙全份，绿切 | $2\Psi+2\Psi+K\Psi/N_d$ | 31.4 GB |
| $P_{os+g}$（ZeRO-2） | 蓝全份，橙绿切 | $2\Psi+(2+K)\Psi/N_d$ | 16.6 GB |
| $P_{os+g+p}$（ZeRO-3） | 蓝橙绿全切 | $(2+2+K)\Psi/N_d$ | 1.9 GB |

同一套公式套到本仓库 GPT-2 XL（$\Psi\approx 1.64\mathrm{B}$）上，Baseline mixed 记账约 $16\Psi\approx 26.2\,\mathrm{GB}$；若 $N_d=64$ 且 ZeRO-3，model states 可落到约 $26.2/64\approx 0.41\,\mathrm{GB}$ 量级（**仍不含 activation**）。

---

## 9. 和本仓库代码的一一对应（防概念漂移）

### 9.1 参数 $\Psi$ 从哪来

结构与 `activiations.md` / `flops.md` 一致（RoPE 无训练参数；embedding 与 lm_head 不共享）：

$$
P
= 2Vd
+ N\bigl(4d^{2} + 3d\,d_{\mathrm{ff}} + 2d\bigr)
+ d
$$

TinyStories small：$P=23\,089\,664$。  
GPT-2 XL 配置：$P=1\,640\,452\,800$。

### 9.2 梯度

`loss.backward()` → 每个 `p.grad` 与 `p` 同形状。  
数据并行里，各卡先有本地 grad，再通信成全局平均——Baseline / ZeRO-1 通信后常驻完整 grad；ZeRO-2/3 常驻的是分片。

### 9.3 优化器状态

你们实现的 AdamW：`state[p]` 里至少有与 `p` 同形状的 $m$、$v$。  
这就是绿条的主体。图上的 $K=12$ 再把 FP32 master 等一并计入。

### 9.4 激活（再次强调）

`Q\in\mathbb{R}^{B\times L\times d}`、`logits\in\mathbb{R}^{B\times L\times V}` 等——**只出现在 activation 桶**。  
ZeRO 把 120GB 削成 1.9GB，削的是 model states；  
若 $B$ 很大，$M_{\mathrm{activations}}$ 照样可以把 GPU 打满，那时要靠 **更小 batch、activation checkpointing、序列并行** 等别的手段，不是这张图单独能解决的。

---

## 10. 一张「谁比谁大」决策树（回答你原先的记忆混乱）

```text
问：图上的 state 含不含 activation？
答：不含。state = params + grads + optim。

问：为什么 optimizer 比 parameter 大很多？
答：不是元素更多，是每个参数多存 m、v（常再加 FP32 master），
    按图的记账 K=12，对 FP16 参数 2 字节而言大约 6 倍。

问：那 activation 为什么有时比什么都大？
答：因为它的尺寸跟 B、L（还有 L²）走；小模型大 batch 时（如 TinyStories B=64）
    激活可以是 Adam 状态的几十倍——但这是第四桶，不是绿条。

问：ZeRO 能让我单卡训练变快 20% 吗？
答：ZeRO 主业是「多卡时少存冗余状态、让更大模型塞进卡」，
    不是加速单卡 TinyStories。你们现在的作业模型远没到要 ZeRO 的量级。
```

---

## 11. 一句话收束

- **Baseline DP：** 每卡完整蓝+橙+绿 → 图例 120GB。  
- **ZeRO-1：** 只切绿 → 31.4GB。  
- **ZeRO-2：** 切绿+橙 → 16.6GB。  
- **ZeRO-3：** 蓝橙绿全切 → 1.9GB。  
- **Activation 很大** 和 **Optimizer state 比参数贵** 都对，但是 **两件不同的事**；这张图只解决后者在「多卡重复存储」上的浪费。
