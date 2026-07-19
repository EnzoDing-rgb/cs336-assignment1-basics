# Problem (adamw_accounting)：训练时 Peak Memory

约定：

| 符号 | 含义 |
|------|------|
| $B$ | `batch_size` |
| $L$ | `context_length` |
| $V$ | `vocab_size` |
| $N$ | `num_layers` |
| $d$ | `d_model` |
| $h$ | `num_heads` |
| $d_{\mathrm{ff}}$ | FFN 隐层宽度（本题给定 $d_{\mathrm{ff}}=\frac{8}{3}d$，公式里一律保留 $d_{\mathrm{ff}}$） |
| float32 | 每个标量 **4 bytes** |

峰值显存拆成四块：

$$
M_{\mathrm{peak}}
= M_{\mathrm{params}}
+ M_{\mathrm{activations}}
+ M_{\mathrm{grads}}
+ M_{\mathrm{optim}}
$$

---

## 0. Weight vs Activation

| | **Weight / Parameter** | **Activation** |
|--|------------------------|----------------|
| 是什么 | 可训练的 $W$、$g$（RMSNorm gain）等 | 前向算出的中间张量 |
| 形状里有没有 $B$、$L$ | 没有 | 有 |
| 例子 | $W_Q\in\mathbb{R}^{d\times d}$ | $Q\in\mathbb{R}^{B\times L\times d}$ |
| 训练时还要什么 | 同形状 gradient；AdamW 再加同形状 $m$、$v$ | 反传需要保留（本题计入峰值） |

---

## 1. Parameters

结构（RoPE 无训练参数，不计入）：

```text
Embedding (V×d)
→ N × Transformer block：
     RMSNorm (d) → MHA(Q,K,V,O) → 残差
     RMSNorm (d) → SwiGLU(W1,W2,W3) → 残差
→ Final RMSNorm (d)
→ LM Head (V×d)    （与 embedding 不共享，再计一份 V×d）
```

| 部分 | 参数量（元素个数） |
|------|-------------------|
| Embedding | $Vd$ |
| 每层 Q、K、V、O | $4d^{2}$（各 $d\times d$；多头不另乘 $h$） |
| 每层 2×RMSNorm | $2d$ |
| 每层 SwiGLU $W_1,W_2,W_3$ | $3d\cdot d_{\mathrm{ff}}$ |
| Final RMSNorm | $d$ |
| LM Head | $Vd$ |

单层 block：$4d^{2} + 3d\cdot d_{\mathrm{ff}} + 2d$。

全模型参数元素数：

$$
\boxed{
P
= 2Vd
+ N\bigl(4d^{2} + 3d\cdot d_{\mathrm{ff}} + 2d\bigr)
+ d
}
$$

$$
\boxed{M_{\mathrm{params}} = 4P \ \text{bytes}}
$$

---

## 2. Activations

只统计讲义列出的组件。训练峰值：下列张量按反传需要全部保留（无 checkpointing），故各层 ×$N$ 再加出口。

读表约定：

- **输入 / 权重 / 输出** 写张量形状；**元素数** = 输出里有多少个 float（计入 $A$ 的就是这一列）。
- 权重列是「谁乘谁」里的 $W$，**权重本身不计入 activation**（已在 §1）。
- 记 $d_h = d/h$（每头维度）。多头时 $QK^{\top}$ 在每个 head 上各做一次 $L\times L$。

块入口残差流记为 $x$，形状 $B\times L\times d$（元素 $BLd$）。下面从 attn 支路算起。

### 2.1 Attention 前的 RMSNorm

逐位置对最后一维做 RMS，再乘讲义里的 gain $g\in\mathbb{R}^{d}$（$g$ 是参数，不算 activation）：
$\mathrm{RMSNorm}(a)_i = \dfrac{a_i}{\mathrm{RMS}(a)}\, g_i$。

| 步骤 | 运算 | 输入形状 | 「乘」什么 | 输出形状 | 元素数 |
|------|------|----------|------------|----------|--------|
| attn-RMSNorm | $x_{\mathrm{n}}=\mathrm{RMSNorm}(x)$ | $B\times L\times d$ | 逐维 $\times g$（广播） | $B\times L\times d$ | $BLd$ |

### 2.2 Q、K、V 投影

每个位置的 $d$ 维向量左乘 $d\times d$ 的投影矩阵（实现里三个 `Linear`；等价于一次乘完再拆成 $h$ 头）。

| 步骤 | 运算 | 输入 | 权重 | 输出形状 | 元素数 |
|------|------|------|------|----------|--------|
| $Q$ | $Q = x_{\mathrm{n}} W_Q$ | $B\times L\times d$ | $W_Q:\ d\times d$ | $B\times L\times d$ | $BLd$ |
| $K$ | $K = x_{\mathrm{n}} W_K$ | 同上 | $W_K:\ d\times d$ | $B\times L\times d$ | $BLd$ |
| $V$ | $V = x_{\mathrm{n}} W_V$ | 同上 | $W_V:\ d\times d$ | $B\times L\times d$ | $BLd$ |
| **小计** | | | | | **$3BLd$** |

拆头后看法（元素数不变）：$Q,K,V$ 也可看成 $B\times h\times L\times d_h$，因为 $h\cdot d_h=d$。

### 2.3 $QK^{\top}$、softmax、对 $V$ 加权、输出投影 $O$

先分清两个都叫「$V$」的东西：

| 名字 | 是什么 | 形状 |
|------|--------|------|
| 权重 $W_V$ | 参数，§1 已计 | $d\times d$ |
| 激活 $V$ | $V = x_{\mathrm{n}} W_V$ 的**输出** | $B\times L\times d$，拆头后 $B\times h\times L\times d_h$ |

下面公式里的 $Y = A V$，乘的是 **激活 $V$**，不是 $W_V$。  
$d_h = d/h$。元素数：$B\cdot h\cdot L\cdot d_h = BLd$（因为 $h\cdot d_h = d$）。

#### Softmax 在这里干什么

$S = QK^{\top}/\sqrt{d_h}$ 的形状是 $B\times h\times L\times L$：对每个 batch、每个 head，是一张 $L\times L$ 的分数表。  
第 $i$ 行 =「第 $i$ 个 query 位置」对全部 $L$ 个 key 位置的原始分数（可正可负，且一般**行和不为 1**）。

$\mathrm{softmax}$ **只沿着每一行的 key 维（最后一维 $L$）**做：

$$
A_{i,j} = \frac{\exp(S_{i,j})}{\sum_{j'=1}^{L}\exp(S_{i,j'})}
$$

（实现里常先减行 max 再 $\exp$，数值更稳；因果 LM 还会在 softmax 前把未来位置的分数打成 $-\infty$。）

结果：

- 形状仍是 $B\times h\times L\times L$（元素数仍 $BhL^{2}$，另存一份）；
- 每一行变成一组 **非负且和为 1** 的权重 =「这个 query 位置对各个 key 位置看多少」。

所以 $A$ 不是又一次投影，只是把分数表收成概率表。

#### $Y = A V$：谁乘谁，为什么元素数是 $BLd$

对每个 batch、每个 head，做一次矩阵乘：

$$
\underbrace{A}_{L\times L}\ 
\underbrace{V}_{L\times d_h}
=
\underbrace{Y}_{L\times d_h}
$$

完整带上 $B,h$：

$$
(B\times h\times L\times L)\ @\ (B\times h\times L\times d_h)
\;=\;
B\times h\times L\times d_h
$$

读法：

- $A$ 的第 $i$ 行：位置 $i$ 对所有位置的注意力权重；
- $V$ 的第 $j$ 行：位置 $j$ 的 **内容向量**（长度 $d_h$，不是 $d\times d$）；
- $Y$ 的第 $i$ 行 = 用 $A$ 的第 $i$ 行去 **加权平均** 各行 $V$，得到位置 $i$ 的新向量（仍长度 $d_h$）。

元素数：

$$
|Y| = B\cdot h\cdot L\cdot d_h = B\cdot L\cdot (h\cdot d_h) = B\cdot L\cdot d = BLd
$$

把 $h$ 个头在特征维拼回，得到 $B\times L\times d$，元素数还是 $BLd$。

| 步骤 | 运算 | 谁 × 谁 | 输出形状 | 元素数 |
|------|------|---------|----------|--------|
| scores | $S = QK^{\top}/\sqrt{d_h}$ | $(B\times h\times L\times d_h)\,@\,(B\times h\times d_h\times L)$ | $B\times h\times L\times L$ | $BhL^{2}$ |
| attn 权重 | $A=\mathrm{softmax}(S)$（沿最后一维） | 不对 $V$ 做矩阵乘，只改 $S$ 的数值 | $B\times h\times L\times L$ | $BhL^{2}$ |
| 加权和 | $Y = A V$（$V$=激活） | $(B\times h\times L\times L)\,@\,(B\times h\times L\times d_h)$ | $B\times h\times L\times d_h$ → 拼头 $B\times L\times d$ | $BLd$ |
| 输出投影 | $O_{\mathrm{out}} = Y W_O$ | $(B\times L\times d)\,@\,(d\times d)$ | $B\times L\times d$ | $BLd$ |

（残差 $x\leftarrow x+O_{\mathrm{out}}$ 不单列一张激活表。）

Attention 子层 activation 小计（含 QKV，不含 attn-RMSNorm）：

$$
3BLd + BhL^{2} + BhL^{2} + BLd + BLd = 5BLd + 2BhL^{2}
$$

### 2.4 FFN 前的 RMSNorm + SwiGLU

此时残差流仍是 $B\times L\times d$，记为 $x'$。

| 步骤 | 运算 | 输入 | 权重 / 逐元 | 输出形状 | 元素数 |
|------|------|------|-------------|----------|--------|
| ffn-RMSNorm | $x'_{\mathrm{n}}=\mathrm{RMSNorm}(x')$ | $B\times L\times d$ | $\times g$ | $B\times L\times d$ | $BLd$ |
| $W_1$ 升维（gate 预激活） | $u = x'_{\mathrm{n}} W_1$ | $B\times L\times d$ | $W_1:\ d\times d_{\mathrm{ff}}$ | $B\times L\times d_{\mathrm{ff}}$ | $BL\,d_{\mathrm{ff}}$ |
| SiLU | $u_{\mathrm{SiLU}} = \mathrm{SiLU}(u)=u\cdot\sigma(u)$ | $B\times L\times d_{\mathrm{ff}}$ | 逐元 | $B\times L\times d_{\mathrm{ff}}$ | $BL\,d_{\mathrm{ff}}$ |
| $W_3$ 升维（内容支路） | $v = x'_{\mathrm{n}} W_3$ | $B\times L\times d$ | $W_3:\ d\times d_{\mathrm{ff}}$ | $B\times L\times d_{\mathrm{ff}}$ | $BL\,d_{\mathrm{ff}}$ |
| 门控乘积 | $h = u_{\mathrm{SiLU}} \odot v$ | 两个 $B\times L\times d_{\mathrm{ff}}$ | 逐元相乘 | $B\times L\times d_{\mathrm{ff}}$ | $BL\,d_{\mathrm{ff}}$ |
| $W_2$ 压回 | $y = h W_2$ | $B\times L\times d_{\mathrm{ff}}$ | $W_2:\ d_{\mathrm{ff}}\times d$ | $B\times L\times d$ | $BLd$ |

FFN 支路 activation 小计：

$$
BLd + 4BL\,d_{\mathrm{ff}} + BLd = 2BLd + 4BL\,d_{\mathrm{ff}}
$$

### 2.5 单个 block 合计（把上面加总）

| 来源 | 元素数 |
|------|--------|
| attn-RMSNorm | $BLd$ |
| Q、K、V | $3BLd$ |
| $QK^{\top}$ + softmax | $2BhL^{2}$ |
| 加权和 + $O$ | $2BLd$ |
| ffn-RMSNorm | $BLd$ |
| SwiGLU 四段宽激活（$W_1$/SiLU/$W_3$/乘积） | $4BL\,d_{\mathrm{ff}}$ |
| $W_2$ 输出 | $BLd$ |
| **block 合计** | **$8BLd + 2BhL^{2} + 4BL\,d_{\mathrm{ff}}$** |

$$
A_{\mathrm{block}}
= 8BLd + 2BhL^{2} + 4BL\,d_{\mathrm{ff}}
$$

$N$ 层则 ×$N$（每层各存一套）。

### 2.6 出口：final RMSNorm、logits、cross-entropy

整网 $N$ 层之后，残差流 $z$ 形状仍为 $B\times L\times d$。

| 步骤 | 运算 | 输入 | 权重 | 输出形状 | 元素数 |
|------|------|------|------|----------|--------|
| final RMSNorm | $z_{\mathrm{n}}=\mathrm{RMSNorm}(z)$ | $B\times L\times d$ | $\times g$ | $B\times L\times d$ | $BLd$ |
| LM Head / output embedding | $\mathrm{logits}=z_{\mathrm{n}} W_{\mathrm{LM}}$ | $B\times L\times d$ | $W_{\mathrm{LM}}:\ d\times V$（元素数同 $V\times d$） | $B\times L\times V$ | $BLV$ |

Cross-entropy（把 logits 看成 $BL$ 个样本，每个词表维 $V$）：

| 步骤 | 运算 | 形状 | 元素数 | 是否计入本题 $A$ |
|------|------|------|--------|------------------|
| logits（已上） | — | $BL\times V$ | $BLV$ | 已在 output embedding 计 |
| `shifted = logits - max` | 逐行减 max | $BL\times V$ | $BLV$ | **计入**（CE 主导额外激活） |
| `max` / `logsumexp` / per-example loss | 归约 | $O(BL)$ | $O(BL)$ | 相对 $BLV$ 忽略 |

出口合计计入 $A$ 的：

$$
BLd + BLV + BLV = BLd + 2BLV
$$

### 2.7 全模型激活合计与内存

$$
\boxed{
A
= N\bigl(8BLd + 2BhL^{2} + 4BL\,d_{\mathrm{ff}}\bigr)
+ BLd
+ 2BLV
}
$$

$$
\boxed{M_{\mathrm{activations}} = 4A \ \text{bytes}}
$$

形状速查（防把 $L$ 写进权重）：

| 名字 | 典型形状 | 有没有 $L$ |
|------|----------|------------|
| 权重 $W_Q$ 等 | $d\times d$ | 无 |
| 激活 $Q$ | $B\times L\times d$ | 有 |
| 激活 $QK^{\top}$ | $B\times h\times L\times L$ | 有（而且是 $L^{2}$） |
| 激活 $W_1x$ | $B\times L\times d_{\mathrm{ff}}$ | 有 |
| logits | $B\times L\times V$ | 有 |

---

## 3. Gradients

每个可训练参数一份同形状 `.grad`：

$$
\boxed{M_{\mathrm{grads}} = 4P \ \text{bytes}}
$$

---

## 4. Optimizer state（AdamW）

$m$、$v$ 与每个参数张量 **同形状、逐元素** 对应：

$$
m \leftarrow \beta_1 m + (1-\beta_1)g,
\qquad
v \leftarrow \beta_2 v + (1-\beta_2)g^{2}
$$

| 状态 | 元素数 | 内存 |
|------|--------|------|
| 全体 $m$ | $P$ | $4P$ |
| 全体 $v$ | $P$ | $4P$ |
| 合计 | $2P$ | $8P$ |

（步数 $t$ 可忽略。）

$$
\boxed{M_{\mathrm{optim}} = 8P \ \text{bytes}}
$$

---

## 5. Deliverable（a）

| 部分 | 表达式（bytes） |
|------|-----------------|
| parameters | $4P$ |
| activations | $4A$ |
| gradients | $4P$ |
| optimizer state | $8P$ |
| **total** | $\boxed{16P + 4A}$ |

其中 $P$、$A$ 见上；$16P$ 与 $B$ 无关，$B$ 只进入 $4A$。

四块同时在场时的拆法：

```text
parameters, optimizer state   — 常驻
gradients                    — backward 写出
activations                  — 随 B、L 变大，为反传保留
```

---

## 6. 实算：GPT-2 XL 规格下的显存结构与趋势

配置与 `reports/calculation.md` 一致（仓库可实例化的 GPT-2 XL）：

| 超参 | 值 |
|------|-----|
| $V$ | $50257$ |
| $L$ | $1024$ |
| $N$ | $48$ |
| $d$ | $1600$ |
| $h$ | $25$ |
| $d_{\mathrm{ff}}$ | $4288$ |
| dtype | float32（4 bytes / 元素） |

已核对参数量：

$$
P = 1\,640\,452\,800
$$

### 6.1 一步训练里四块怎么活着

| 块 | 字节（本配置） | 生命周期 |
|----|----------------|----------|
| parameters | $4P \approx 6.11\,\mathrm{GiB}$ | 整段训练常驻 |
| optimizer state（$m,v$） | $8P \approx 12.22\,\mathrm{GiB}$ | 整段训练常驻（AdamW） |
| gradients | $4P \approx 6.11\,\mathrm{GiB}$ | `backward` 写出；`step` 用完后可清掉，给下一步腾地方 |
| activations | $4A$，随 $B$ 变 | 前向为反传留下；反传算完梯度后可释放。在「前向+反传」这段峰值里，它和权重一样占着坑 |

和你说的对齐：

- **常驻跨 step**：parameters + optimizer state（本配置合计约 **18.33 GiB**，与 $B$ 无关）。
- **gradient**：峰值里有一份和参数同大的；用完可丢，下一 step 再写。
- **activation**：为了反传要留着，所以在单步峰值里像「临时常驻」；它随 $B$、$L$ 涨，是放大 batch 时最先顶满显存的那一块。

峰值公式仍是：

$$
M_{\mathrm{peak}} = 16P + 4A
= \underbrace{24.44\,\mathrm{GiB}}_{16P}
+ 4A(B)
$$

### 6.2 激活拆开（$B=1$ 时的元素数）

$$
A\big|_{B=1}
= N(8Ld + 2hL^{2} + 4L d_{\mathrm{ff}}) + Ld + 2LV
= 4\,093\,347\,840
$$

| 来源 | 元素数（$B=1$） | 约占 $A$ |
|------|----------------:|----------:|
| $N\cdot 2BhL^{2}$（各层 $S$+$A$） | $2\,516\,582\,400$ | **61.5%** |
| $N\cdot 4BL d_{\mathrm{ff}}$（SwiGLU 宽激活） | $843\,055\,104$ | 20.6% |
| $N\cdot 8BLd$（norm / QKV / 加权和 / $O$ 等） | $629\,145\,600$ | 15.4% |
| $2BLV$（logits + CE `shifted`） | $102\,926\,336$ | 2.5% |
| final $BLd$ | $1\,638\,400$ | $<0.1\%$ |

观察：在 $L=1024$、本套记账下，**激活大头是每层两张 $L\times L$ 注意力表（×头数×层数）**，不是 embedding/logits。

$B=1$ 时激活显存：

$$
4A \approx 15.25\,\mathrm{GiB}
$$

### 6.3 随 batch 怎么变

$A$ 对 $B$ 线性，故激活显存 $= B\times 15.25\,\mathrm{GiB}$；固定块 $16P$ 不动。

| $B$ | 激活 | $16P$ | 峰值合计 | 激活占比 |
|----:|-----:|------:|---------:|---------:|
| 1 | 15.25 GiB | 24.44 GiB | **39.69 GiB** | 38% |
| 2 | 30.50 GiB | 24.44 GiB | **54.94 GiB** | 55% |
| 4 | 60.99 GiB | 24.44 GiB | **85.44 GiB** | 71% |
| 8 | 121.99 GiB | 24.44 GiB | **146.4 GiB** | 83% |

趋势：

1. **$B$ 小**（如 1）：固定的参数+优化器+梯度（$16P$）还大于激活；峰值里「模型状态」和「这一步的激活」量级接近。  
2. **$B$ 增大**：激活线性抬升，很快变成峰值主体（$B=4$ 已约七成）。  
3. 卡在 **80 GiB** 量级时，按本式 $B_{\max}\approx 3$（$80\,\mathrm{GiB}-24.44$ 再除以每 batch 15.25 GiB）；再大就要减 $L$、checkpoint、或更小精度。  
4. **gradient** 在峰值公式里按「整网一份 $4P$」计；它与 $B$ 无关。变大的是激活，不是梯度张量个数。

### 6.4 一句话

XL + AdamW + 本套激活口径：约 **18 GiB** 权重与优化器常驻；一步峰值再叠 **梯度 ~6 GiB** 与 **每 batch ~15 GiB 激活**；$B$ 从 1 提到 4，峰值从 ~40 GiB 涨到 ~85 GiB，多出来的几乎全是激活（尤其是 $L^{2}$ 注意力表）。

---

## 7. 推理时显存是什么情况

前面 §1–§6 都是 **训练一步**（前向 + 反传 + AdamW）的峰值口径。推理（只做前向、出 token）账本不一样。  
规格仍用本节 GPT-2 XL：$N=48$，$d=1600$，$h=25$，$d_k=64$，权重 FP32 约 **6.11 GiB**。

### 7.1 还会不会像训练那样占？

大体上：**少一大截。**

| 训练峰值里有的 | 推理还要不要 |
|----------------|--------------|
| parameters（权重） | 要 ≈ **6.11 GiB** |
| optimizer state（$m,v$） | **不要** |
| gradients | **不要** |
| 为反传存满的整网激活 | **不要**按训练那套留；层算完即可释放 |

推理侧会随「已经生成了多长、同时服务几路」变大的，主要是下面的 **KV cache**。

### 7.1.1 KV cache 是什么（用本模型一步步算）

先把本节要用的名字钉死（都是本 GPT-2 XL 的数）：

| 符号 | 中文 | 本例取值 | 含义 |
|------|------|----------|------|
| $N$ | 层数 | $48$ | Transformer block 叠了多少层；每层各自存一份 $K$、$V$ |
| $d$ | 模型宽度 | $1600$ | 每个位置隐状态的总通道数 |
| $h$ | 头数（num_heads） | $25$ | 多头注意力切成多少个头 |
| $d_k$ | 每头维度 | $d/h=64$ | 每个头里，$K$ 或 $V$ 在一个位置上有多长 |
| $B$ | 并发路数 | 先取 $1$ | 同时服务几路请求；每路各有一份 cache |
| $T$ | 已缓存长度 | 随生成变大 | cache 里已经写下了多少个位置（prompt+已生成） |

关系：$h\times d_k = d$，即 $25\times 64=1600$。所以「按头存 $h\times d_k$」和「按整宽存 $d$」元素个数一样。

解码时，每层注意力都要对本轮的 query 去看 **已经出现过的所有位置** 的 $K$、$V$。  
每个位置的 $K$、$V$ 算过一次就 **存下来**，下次只算新位置，再拼上缓存。这份缓存就是 **KV cache**。

本模型每一层、一路请求（$B=1$）、某一个已写入的位置，要存：

| 张量 | 形状怎么读 | 元素数 |
|------|------------|--------|
| 该层 $K$ | $h$ 个头 × 这 1 个位置 × 每头 $d_k$ 个数 → $25\times 1\times 64$ | $25\times 64=1600$ |
| 该层 $V$ | 同上 | $1600$ |
| 该层小计 | $K$ 与 $V$ 各一份 | $3200$ |

48 层都存：

$$
48 \times 3200 = 153\,600 \text{ 个 float}
$$

FP32：

$$
153\,600 \times 4 = 614\,400 \text{ bytes} = 600\,\mathrm{KiB}
$$

**每多生成 1 个 token（一路请求），KV cache 固定增加 600 KiB。**

### 7.1.2 生成过程里 cache 怎么长（$B=1$）

假设用户 prompt 已算完并写入 cache，当前长度 $T$ 表示「cache 里已有 $T$ 个位置」。

| 已缓存长度 $T$ | KV cache 元素数 $=153600\times T$ | FP32 大小 |
|---------------:|----------------------------------:|----------:|
| 1 | $153\,600$ | **600 KiB** |
| 2 | $307\,200$ | **1.17 MiB** |
| 128 | $19\,660\,800$ | **75 MiB** |
| 512 | $78\,643\,200$ | **300 MiB** |
| 1024 | $157\,286\,400$ | **600 MiB ≈ 0.59 GiB** |

公式（本配置、$B=1$）：

$$
M_{\mathrm{KV}}
= 2 \times N \times T \times d \times 4\ \text{bytes}
= 48 \times T \times 1600 \times 2 \times 4
= 614\,400 \times T\ \text{bytes}
$$

（$2$ = 每层的 $K$ 与 $V$；$N\times d = 48\times 1600$ 把所有层、所有通道算上。）

对照同一张卡上的权重：权重约 **6.11 GiB** 常驻；一路请求撑到 $T=1024$ 时，KV cache 再加约 **0.59 GiB**。

### 7.1.3 同时服务多路时（推理 batch）

每路请求各自一份 KV cache。$B$ 路、每路都到 $T=1024$：

| 并发路数 $B$ | KV cache（$T=1024$） |
|-------------:|---------------------:|
| 1 | 0.59 GiB |
| 4 | 2.34 GiB |
| 8 | 4.69 GiB |

$$
M_{\mathrm{KV}}(B,T) = B \times 614\,400 \times T\ \text{bytes}
$$

例：8 路都生成到 1024 token → KV cache ≈ **4.69 GiB**，再加权重 6.11 GiB，光「权重 + KV」已约 **10.8 GiB**（尚未计本步临时激活等）。

### 7.2 同叫 batch，训练和推理目的不同

| | 训练的 batch | 推理的 batch |
|--|--------------|--------------|
| 一条样本通常是 | 一条（或一块）训练序列，用来估梯度 | 一个（或一组）用户请求 / 生成会话 |
| 把 $B$ 做大主要图什么 | 同样时间内多看数据、梯度更稳、吞吐更高（数据并行） | **服务效率**：一张卡同时扛多路请求，提高利用率、降低平均等待 |
| 和损失的关系 | 直接进入平均损失 / 梯度 | 一般不回传损失；各请求前向（及解码）打包算 |

两边都可以说「并行」：训练并行的是 **样本量**；推理并行的常常是 **并发请求**。  
显存上，推理的 $B$ 会放大当前激活和 KV cache，仍按「服务能塞多少路」来取舍，而不是按「反传能不能留住激活」来取舍。

---

## 8. Deliverable（b）（c）

规格与 §6 相同（GPT-2 XL 形）：$V=50257$，$L=1024$，$N=48$，$d=1600$，$h=25$，$d_{\mathrm{ff}}=4288$，float32。  
记 $B=\mathrm{batch\_size}$。由 §5：$M_{\mathrm{peak}}=16P+4A$，且 $A\propto B$。

### （b）只含 $B$ 的峰值式，以及 80GB 下最大 $B$

已算：

$$
P = 1\,640\,452\,800
$$

$$
\frac{A}{B}
= N\bigl(8Ld + 2hL^{2} + 4L d_{\mathrm{ff}}\bigr) + Ld + 2LV
= 4\,093\,347\,840
$$

因此（单位：**bytes**）：

$$
\begin{aligned}
M_{\mathrm{peak}}(B)
&= 4\cdot\Bigl(\frac{A}{B}\Bigr)\cdot B + 16P \\
&= 16\,373\,391\,360\cdot B + 26\,247\,244\,800
\end{aligned}
$$

写成题目要的形状：

$$
\boxed{M_{\mathrm{peak}}(B) = aB + b}
\qquad
a = 16\,373\,391\,360,\quad
b = 26\,247\,244\,800
$$

（$a$ = 每个 batch 的激活字节；$b=16P$ = 参数+梯度+AdamW state。）

取 $80\,\mathrm{GB}=80\times 10^{9}$ bytes：

$$
aB + b \le 80\times 10^{9}
\implies
B \le \frac{80\times 10^{9}-b}{a} \approx 3.28
$$

$$
\boxed{B_{\max} = 3}
$$

（即使用 $80\times 2^{30}$ bytes 计，上界约 $3.64$，最大整数 batch 仍是 $3$。）

### （c）AdamW 一步的 FLOPs（按元素逐项加总）

约定：与作业里矩阵乘计数一样，**一次标量乘 / 加 / 减 / 除 / 开方各算 1 FLOP**。  
AdamW 只遍历参数，与 $B$ 无关。标量 $\alpha_t$、$(1-\alpha\lambda)$ 等每个 step（或每个 param tensor）算一次，相对 $P$ 可忽略，下面只计 **每个参数元素** 上的运算。

讲义一步（对每个参数元素 $\theta$，配有梯度 $g$、动量 $m$、二阶矩 $v$）：

| 步骤 | 公式 | 逐元素运算 | FLOPs |
|------|------|------------|------:|
| 解耦 weight decay | $\theta \leftarrow \theta\cdot(1-\alpha\lambda)$ | 1 次乘（$(1-\alpha\lambda)$ 为标量） | $1$ |
| 更新 $m$ | $m \leftarrow \beta_1 m + (1-\beta_1)g$ | $\beta_1 m$，$\,(1-\beta_1)g$，再加 | $3$ |
| 更新 $v$ | $v \leftarrow \beta_2 v + (1-\beta_2)g^{2}$ | $g^{2}$，$\beta_2 v$，$(1-\beta_2)g^{2}$，再加 | $4$ |
| 写回 $\theta$ | $\theta \leftarrow \theta - \alpha_t\dfrac{m}{\sqrt{v}+\varepsilon}$ | $\sqrt{v}$，$\,+\varepsilon$，除法，$\,\times\alpha_t$，再减 | $5$ |
| **每个元素合计** | | | **$13$** |

（$\alpha_t=\alpha\sqrt{1-\beta_2^{t}}/(1-\beta_1^{t})$ 每步对整组参数共用，算在 $O(1)$ 里，不进 $13$。）

因此一步 AdamW：

$$
\boxed{\mathrm{FLOPs}_{\mathrm{AdamW\ step}} = 13P}
$$

代入 GPT-2 XL 形的 $P$：

$$
13P = 13 \times 1\,640\,452\,800 = 21\,325\,886\,400 \approx 2.13\times 10^{10}
$$

$$
\boxed{13P = 21\,325\,886\,400 \ \text{FLOPs}}
$$

同规格下，一次 **前向**里矩阵乘的 FLOPs（$L=1024$）为：

$$
N\bigl(8Ld^{2} + 4L^{2}d + 6Ld\cdot d_{\mathrm{ff}}\bigr) + 2LdV
$$

代入 $N=48$，$L=1024$，$d=1600$，$d_{\mathrm{ff}}=4288$，$V=50257$：

$$
\begin{aligned}
&48\bigl(8\cdot 1024\cdot 1600^{2} + 4\cdot 1024^{2}\cdot 1600 + 6\cdot 1024\cdot 1600\cdot 4288\bigr) \\
&\quad + 2\cdot 1024\cdot 1600\cdot 50257 \\
&= 3\,516\,769\,894\,400
\approx 3.52\times 10^{12}
\end{aligned}
$$

（其中 $8Ld^{2}$ 来自 Q/K/V/O 投影，$4L^{2}d$ 来自 $QK^{\top}$ 与 $AV$，$6Ld\cdot d_{\mathrm{ff}}$ 来自 SwiGLU 三路线性，最后 $2LdV$ 来自 LM head。）

对照：

| | FLOPs | 数量级 |
|--|------:|--------|
| 一次前向（矩阵乘，$L=1024$） | $3.52\times 10^{12}$ | $\sim 10^{12}$ |
| 一步 AdamW | $2.13\times 10^{10}$ | $\sim 10^{10}$ |
| 比值（前向 / AdamW） | $3.52\times 10^{12}\,/\,2.13\times 10^{10} \approx 165$ | 约两个数量级 |

原因很具体：前向要对 **整段序列、每一层** 做大矩阵乘（还带 $L^{2}$ 的注意力），工作量随 $L$、$N$、$d$、$d_{\mathrm{ff}}$、$V$ 涨；AdamW 只对每个参数做 **13 次** 标量运算，工作量只有 $13P$，与 $B$、$L$ 无关。所以同一次训练 step 里，「更新优化器」比「跑一遍模型前向」轻大约两个数量级。

---

## 9. Batch size 进 FLOPs 公式时，$N$ 是什么？

### 9.1 符号先钉死（本文件前后文）

前向矩阵乘常用这套字母：

| 符号 | 英文 / 代码名 | 含义 | GPT-2 XL 例 |
|------|---------------|------|-------------|
| $B$ | `batch_size` | **一次前向同时送进模型的序列条数** | 题目里取 $1024$ |
| $L$ | `context_length` / seq | **每条序列的 token 长度** | $1024$ |
| $N$ | `num_layers` | **Transformer block 叠了多少层** | $48$ |
| $d$ | `d_model` | 隐状态宽度 | $1600$ |
| $h$ | `num_heads` | 注意力头数 | $25$ |
| $d_{\mathrm{ff}}$ | FFN 宽度 | SwiGLU 中间宽 | $4288$ |
| $V$ | `vocab_size` | 词表大小 | $50257$ |

公式里写在最外面的 **$N$ 是层数 `num_layers`**。  
**batch size 在本文件里写成 $B$**，两个字母各管一件事。

### 9.2 一条序列的前向公式（$B=1$）

对 **一条** 长度为 $L$ 的序列，前向里矩阵乘的 FLOPs 是：

$$
F_1
=
N\bigl(8Ld^{2} + 4L^{2}d + 6Ld\cdot d_{\mathrm{ff}}\bigr) + 2LdV
$$

这里的 $N$：每一层都做一遍 Attention + FFN，所以整式最外层乘层数 $N$。  
$L=1024$ 时：

$$
F_1 = 3\,516\,769\,894\,400 \approx 3.52\times 10^{12}
$$

### 9.3 一次训练 step、batch 为 $B$ 时

一个 step 里同时有 $B$ 条序列（每条长度 $L$）。  
矩阵乘对「序列条数」是线性的：每条各自走一遍同样的层，总算术量是 $B$ 条相加：

$$
F_{\mathrm{fwd}}(B,L)
=
B\cdot F_1
=
B\cdot\Bigl(
N\bigl(8Ld^{2} + 4L^{2}d + 6Ld\cdot d_{\mathrm{ff}}\bigr) + 2LdV
\Bigr)
$$

把 $B$ 写进形状里读：激活是 $B\times L\times d$，投影按 $(B\cdot L,\,d)\,@\,(d,d)$ 计，FLOPs $= B\cdot(2Ld^{2})$，相对单条就是乘上 $B$。

题目 $B=1024$ 时，**一个 step 的前向**是：

$$
F_{\mathrm{fwd}}(1024,1024)=1024\times F_1\approx 3.60\times 10^{15}
$$

反传按 $2\times$ 前向，该 step 模型算力：

$$
F_{\mathrm{step}}(B,L)=3\,F_{\mathrm{fwd}}(B,L)=3B\,F_1
$$

训 $N_{\mathrm{steps}}$ 步的总模型 FLOPs：

$$
F_{\mathrm{train}}
=
N_{\mathrm{steps}}\cdot 3B\cdot F_1
$$

（这就是 §10 里 MFU 时间公式用的分子。）

---

## 10. Deliverable（d）：H100 上训 GPT-2 XL 要多久

题目设定（符号与 §9 一致）：

- H100 理论峰值：$\mathrm{Peak}=495\,\mathrm{TFLOP/s}=495\times 10^{12}$ FLOP/s
- $\mathrm{MFU}=50\%$
- $N_{\mathrm{steps}}=400\,000$，$B=1024$，$L=1024$（GPT-2 XL 形）
- 反传 FLOPs $=2\times$ 前向（Kaplan / Hoffmann 惯例）

### 10.1 时间公式

$$
T_{\mathrm{wall}}
=
\frac{F_{\mathrm{train}}}{\mathrm{Peak}\times\mathrm{MFU}}
=
\frac{N_{\mathrm{steps}}\cdot F_{\mathrm{step}}(B,L)}{\mathrm{Peak}\times\mathrm{MFU}}
=
\frac{N_{\mathrm{steps}}\cdot 3B\cdot F_1}{\mathrm{Peak}\times\mathrm{MFU}}
$$

等效算力：

$$
\mathrm{Peak}\times\mathrm{MFU}
= 495\times 10^{12} \times 0.5
= 247.5\times 10^{12}\ \mathrm{FLOP/s}
$$

### 10.2 代入 $B=1024$

由 §9：$L=1024$ 时 $F_1\approx 3.52\times 10^{12}$，故

$$
F_{\mathrm{fwd}}(1024,1024)
= B\cdot F_1
= 1024 \times 3\,516\,769\,894\,400
= 3\,601\,172\,371\,865\,600
\approx 3.60\times 10^{15}
$$

$$
F_{\mathrm{step}}(1024,1024)
= 3B\,F_1
= 10\,803\,517\,115\,596\,800
\approx 1.08\times 10^{16}
$$

（AdamW 一步仅 $\sim 2\times 10^{10}$，相对 $10^{16}$ 可忽略，本题按讲义只计前向+反传。）

$$
F_{\mathrm{train}}
= N_{\mathrm{steps}}\cdot 3B\cdot F_1
= 400\,000 \times 1.08\times 10^{16}
= 4\,321\,406\,846\,238\,720\,000\,000
\approx 4.32\times 10^{21}
$$

### 10.3 墙钟时间

$$
T_{\mathrm{sec}}
=
\frac{F_{\mathrm{train}}}{\mathrm{Peak}\times\mathrm{MFU}}
=
\frac{4.32\times 10^{21}}{247.5\times 10^{12}}
\approx 1.746\times 10^{7}\ \mathrm{s}
$$

$$
T_{\mathrm{hours}}
=
\frac{T_{\mathrm{sec}}}{3600}
\approx 4850
$$

$$
\boxed{\approx 4850\ \text{hours}}
$$

（约 $202$ 天，单卡 H100、50% MFU、$B=1024$ 下。）

**算式串起来：**

$$
T_{\mathrm{h}}
=
\frac{N_{\mathrm{steps}}\cdot 3B\cdot F_1}{(\mathrm{Peak}\cdot\mathrm{MFU})\cdot 3600}
=
\frac{400\,000\cdot 3\cdot 1024\cdot 3.5167698944\times 10^{12}}{247.5\times 10^{12}\cdot 3600}
\approx 4850
$$
