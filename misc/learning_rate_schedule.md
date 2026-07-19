# Cosine Learning Rate Schedule + Gradient Clipping

对应 CS336 Assignment 1：

- `Problem (learning_rate_schedule)`（LLaMA 同款 cosine annealing）→ 本文前半（§0–§10）
- `Problem (gradient_clipping)` → 本文后半（§11–§18）

公式约定：行内 `$...$`，独立成行 `$$...$$`。

本文目标：先建立直觉，再把公式逐项读懂，最后落到实现分支。不跳步。梯度裁剪部分带玩具手算例子。

---

## 0. 学习率是什么？调度器又是什么？

优化器每一步都会用梯度去改参数。改多少，由 **学习率** 控制。

记：

| 符号 | 含义 |
|------|------|
| $t$ | 当前训练步（iteration / step），从 $0,1,2,\ldots$ 往前走 |
| $\alpha_t$ | 第 $t$ 步要用的学习率 |
| $\theta$ | 模型参数 |

粗直觉：

$$
\theta \;\leftarrow\; \theta - \alpha_t \cdot (\text{这一步的更新方向})
$$

所以：

- $\alpha_t$ 大：参数这一步被推得远
- $\alpha_t$ 小：参数这一步只挪一点点

**调度器（scheduler）** 不负责算梯度，也不负责改参数。  
它只做一件事：

> 输入当前步 $t$（以及若干超参），输出这一步该用的 $\alpha_t$。

最简单的调度是 **常数调度**：任意 $t$ 都返回同一个 $\alpha$。  
本题要写的是更常见的一种：**先 warm-up 爬升，再 cosine 退火下降，最后钉在最小值上**。

---

## 1. 为啥训练中途要改学习率？

如果整场训练都用同一个 $\alpha$，常常两边不讨好：

1. **刚开始**  
   参数几乎是随机的，损失曲面形状还不稳，梯度方向也吵。  
   - $\alpha$ 太大：一步跨太远，损失炸或剧烈抖动  
   - $\alpha$ 太小：一开始爬得极慢

2. **中期**  
   模型已经有点谱了，希望用相对大的步子，尽快把损失压下去。

3. **后期**  
   参数已经在比较好的谷底附近。  
   若仍用大学习率，容易在谷底附近来回抖，甚至抖出去。  
   所以要把 $\alpha$ 慢慢降小，做精细调整。

因此 Transformer（含 LLaMA）常用的策略是：

> **先爬到一个最大学习率（warm-up），再平滑降到一个最小学习率（cosine annealing），之后一直停在最小值上（post-annealing）。**

---

## 2. 五个超参：每个旋钮管什么

本题 cosine 调度一共吃五个量（再加当前步 $t$）：

| 符号 | 英文说法 | 含义 | 在曲线上像什么 |
|------|----------|------|----------------|
| $t$ | current iteration | 当前步 | 横轴坐标 |
| $\alpha_{\max}$ | maximum learning rate | 允许用到的最大学习率 | 曲线的「山顶」 |
| $\alpha_{\min}$ | minimum / final learning rate | 退火结束后的学习率 | 曲线的「地板」 |
| $T_w$ | warm-up iterations | warm-up 持续多少步 | 从起点爬到山顶的那段长度 |
| $T_c$ | cosine annealing final iteration | cosine 段在第几步结束 | 降到地板的那个绝对步数 |

特别强调一件容易混的事：

- $T_w$：**warm-up 的长度**（从 $t=0$ 数起，爬多少步到顶）
- $T_c$：**cosine 结束时的绝对步数**（「到第 $T_c$ 步时，余弦退火刚好结束」）
- cosine 段本身的长度是

$$
T_c - T_w
$$

不是 $T_c$。  
$T_c$ 是时间轴上的一个点，不是「再退火 $T_c$ 步」。

通常有

$$
0 \le T_w \le T_c
$$

---

## 3. 整条曲线长什么样（三段拼起来）

把横轴当成 $t$，纵轴当成 $\alpha_t$。整条调度曲线分成三段：

```text
α
│         ╭──────╮
│        ╱        ╲
│       ╱          ╲________  α_min
│      ╱
│_____╱
└─────────────────────────────→ t
      0    Tw           Tc
       warm-up      cosine     t > Tc 后一直 α_min
```

| 阶段 | 条件 | 行为 |
|------|------|------|
| Warm-up | $t < T_w$ | 从接近 $0$ **线性**涨到 $\alpha_{\max}$ |
| Cosine annealing | $T_w \le t \le T_c$ | 从 $\alpha_{\max}$ **按半个余弦弧**平滑降到 $\alpha_{\min}$ |
| Post-annealing | $t > T_c$ | **恒等于** $\alpha_{\min}$ |

下面三段各写一遍公式，并把每个符号拆开。

---

## 4. 第一段：Warm-up（$t < T_w$）

公式：

$$
\alpha_t = \frac{t}{T_w}\,\alpha_{\max}
$$

### 4.1 式子在说什么

把 $\frac{t}{T_w}$ 看成「warm-up 进度」：

- $t=0$ 时，进度为 $0$
- $t=T_w$ 时，进度为 $1$
- 中间按比例线性增加

再乘上山顶 $\alpha_{\max}$，得到当前学习率。

### 4.2 端点（用来核对实现）

| $t$ | $\alpha_t$ | 含义 |
|-----|------------|------|
| $0$ | $0$ | 起步几乎不动 |
| $T_w/2$（若 $T_w$ 为偶数） | $\alpha_{\max}/2$ | 爬到一半 |
| $T_w$ | $\alpha_{\max}$ | 刚好到山顶（注意：本题分支写的是 $t < T_w$ 才走 warm-up；到 $t=T_w$ 时通常已进入下一段，但下一段在 $t=T_w$ 处也会给出 $\alpha_{\max}$，所以衔接连续） |

### 4.3 直觉

别一上来就用 $\alpha_{\max}$ 乱撞。  
先用小学习率热身若干步，优化更稳，再放到最大步长。

---

## 5. 第二段：Cosine annealing（$T_w \le t \le T_c$）

这一段要从山顶 $\alpha_{\max}$ 平滑降到地板 $\alpha_{\min}$。

### 5.1 先定义「cosine 段内的进度」$u$

$$
u = \frac{t - T_w}{T_c - T_w}
$$

含义：

| $t$ | $u$ | 位置 |
|-----|-----|------|
| $T_w$ | $0$ | cosine 段起点（刚离开 warm-up） |
| 中点 | $1/2$ | cosine 段走到一半 |
| $T_c$ | $1$ | cosine 段终点 |

所以 $u$ 就是把 $[T_w,\,T_c]$ 线性压到 $[0,\,1]$ 的进度条。

### 5.2 再用余弦把进度变成「还剩多少落差」

讲义公式：

$$
\alpha_t
=
\alpha_{\min}
+
\frac{1}{2}
\Bigl(
1 + \cos\bigl(\pi u\bigr)
\Bigr)
\bigl(\alpha_{\max} - \alpha_{\min}\bigr)
$$

把 $u$ 代回去，完整写成：

$$
\alpha_t
=
\alpha_{\min}
+
\frac{1}{2}
\left(
1 + \cos\left(
\pi\cdot\frac{t - T_w}{T_c - T_w}
\right)
\right)
\bigl(\alpha_{\max} - \alpha_{\min}\bigr)
$$

### 5.3 逐项拆开

| 零件 | 作用 |
|------|------|
| $\alpha_{\max}-\alpha_{\min}$ | 从山顶降到地板的总落差 |
| $\cos(\pi u)$ | $u=0$ 时 $\cos 0 = 1$；$u=1$ 时 $\cos\pi = -1$ |
| $\dfrac{1}{2}\bigl(1+\cos(\pi u)\bigr)$ | 把 $\cos$ 从 $1\to -1$ 映射成系数从 $1\to 0$ |
| $\alpha_{\min} + (\text{系数})\cdot(\text{落差})$ | 系数 $=1$ 时得到 $\alpha_{\max}$；系数 $=0$ 时得到 $\alpha_{\min}$ |

为什么需要 $\dfrac{1}{2}(1+\cos(\cdot))$？

因为 $\cos$ 自己会跑到负数。我们要的是一个 **始终落在 $[0,1]$ 里** 的平滑权重：

$$
w(u) = \frac{1}{2}\bigl(1 + \cos(\pi u)\bigr)
\in [0,1]
$$

于是

$$
\alpha_t = \alpha_{\min} + w(u)\,(\alpha_{\max}-\alpha_{\min})
$$

就是在 $\alpha_{\min}$ 与 $\alpha_{\max}$ 之间做一次 **按余弦形状加权** 的插值。

### 5.4 端点核对（实现时最有用）

**起点** $t=T_w$（此时 $u=0$）：

$$
\begin{aligned}
w(0)
&=
\frac{1}{2}\bigl(1 + \cos 0\bigr)
=
\frac{1}{2}\bigl(1+1\bigr)
=
1
\\[0.5em]
\alpha_{T_w}
&=
\alpha_{\min} + 1\cdot(\alpha_{\max}-\alpha_{\min})
=
\alpha_{\max}
\end{aligned}
$$

**终点** $t=T_c$（此时 $u=1$）：

$$
\begin{aligned}
w(1)
&=
\frac{1}{2}\bigl(1 + \cos\pi\bigr)
=
\frac{1}{2}\bigl(1 + (-1)\bigr)
=
0
\\[0.5em]
\alpha_{T_c}
&=
\alpha_{\min} + 0\cdot(\alpha_{\max}-\alpha_{\min})
=
\alpha_{\min}
\end{aligned}
$$

两端对得上，说明：

- warm-up 结束时与 cosine 开始时，在 $\alpha_{\max}$ 处衔接
- cosine 结束时与 post-annealing 开始时，在 $\alpha_{\min}$ 处衔接

### 5.5 为啥用 cosine，而不是直线降？

也可以线性衰减：

$$
\alpha_t
=
\alpha_{\max}
- u\,(\alpha_{\max}-\alpha_{\min})
$$

那是一条直线。  
cosine 的不同在于：$w(u)=\tfrac12(1+\cos(\pi u))$ 不是直线——

- 刚离开山顶时，$w$ 降得慢一点
- 中间降得更快
- 接近地板时又变缓

实践里对 Transformer 很常用；LLaMA 训练也用这类 cosine schedule。  
本题按讲义实现 cosine 即可，不必自己发明别的形状。

---

## 6. 第三段：Post-annealing（$t > T_c$）

公式：

$$
\alpha_t = \alpha_{\min}
$$

含义：余弦退火日程已经走完。  
之后无论再训多少步，学习率都钉在地板 $\alpha_{\min}$，不再变化。

---

## 7. 三段合在一起（完整定义）

给定 $t,\alpha_{\max},\alpha_{\min},T_w,T_c$，学习率 $\alpha_t$ 定义为：

$$
\alpha_t
=
\begin{cases}
\dfrac{t}{T_w}\,\alpha_{\max},
&
t < T_w
\\[1.0em]
\alpha_{\min}
+
\dfrac{1}{2}
\left(
1 + \cos\left(
\pi\cdot\dfrac{t-T_w}{T_c-T_w}
\right)
\right)
(\alpha_{\max}-\alpha_{\min}),
&
T_w \le t \le T_c
\\[1.2em]
\alpha_{\min},
&
t > T_c
\end{cases}
$$

这就是实现时要对着写的三张卡片。

---

## 8. 和 constant schedule 的对比

| 调度 | $\alpha_t$ 怎么来 |
|------|-------------------|
| Constant | 任意 $t$ 都返回同一个固定 $\alpha$ |
| Cosine（本题） | 随 $t$ 变：先线性升到 $\alpha_{\max}$，再余弦降到 $\alpha_{\min}$，之后保持 $\alpha_{\min}$ |

两者接口可以一样：都是「给 $t$，吐 $\alpha_t$」。  
差别只在映射规则。

---

## 9. 实现时的分支（无逻辑跳步版）

伪代码：

```text
function get_lr(t, α_max, α_min, Tw, Tc):
    if t < Tw:
        return (t / Tw) * α_max

    else if t <= Tc:
        u = (t - Tw) / (Tc - Tw)
        return α_min + 0.5 * (1 + cos(π * u)) * (α_max - α_min)

    else:
        return α_min
```

对应关系：

1. `t < Tw` → §4 warm-up  
2. `Tw <= t <= Tc` → §5 cosine  
3. `t > Tc` → §6 post-annealing  

### 9.1 建议自测的几个点

实现后先手算 / assert 这些点（比盯整条曲线更管用）：

| 测点 | 期望 |
|------|------|
| $t=0$ | $\alpha_t=0$（若 $T_w>0$） |
| $t=T_w$ | $\alpha_t=\alpha_{\max}$ |
| $t=T_c$ | $\alpha_t=\alpha_{\min}$ |
| $t=T_c+1$（或任意更大） | $\alpha_t=\alpha_{\min}$ |
| warm-up 中点 $t=T_w/2$ | $\alpha_t=\alpha_{\max}/2$ |
| cosine 中点（$u=1/2$） | $w=\tfrac12(1+\cos(\pi/2))=\tfrac12$，故 $\alpha_t=\alpha_{\min}+\tfrac12(\alpha_{\max}-\alpha_{\min})$ |

### 9.2 边界上容易踩的坑

1. **$T_c=T_w$**  
   分母 $T_c-T_w=0$，cosine 段长度为 0。正常设定里应保证 $T_c>T_w$，或单独处理。作业测试一般会给合法区间。

2. **$T_w=0$**  
   没有 warm-up。第一段公式里会除以 $T_w$，需要按约定：直接从 $t=0$ 就进入 cosine（或视测试而定）。写代码前看 adapter / 测试怎么约束。

3. **整数除法**  
   若用 Python，进度要用浮点除法：`t / Tw`，不要写成整除 `t // Tw`。

4. **角度单位**  
   $\cos(\pi u)$ 里的 $\pi$ 是弧度。Python 里用 `math.cos(math.pi * u)` 或 `math.cos(math.pi * (t - Tw) / (Tc - Tw))`。

---

## 10. 一句话收束（学习率调度）

- 调度器 = 给步数 $t$ 吐学习率 $\alpha_t$。  
- 本题三条：  
  - $t<T_w$：线性爬到 $\alpha_{\max}$  
  - $T_w\le t\le T_c$：余弦从 $\alpha_{\max}$ 降到 $\alpha_{\min}$  
  - $t>T_c$：钉在 $\alpha_{\min}$  
- $T_c$ 是「第几步结束」，cosine 长度是 $T_c-T_w$。

下一步写代码时，直接把 §7 的分段定义翻译成函数即可。

---

# Gradient Clipping（梯度裁剪）

对应 CS336 Assignment 1：`Problem (gradient_clipping)`。

前面讲的是「这一步允许走多远」的学习率 $\alpha_t$。  
这里讲的是另一道安全阀：**反传算出来的梯度如果整体太猛，先按比例缩小，再交给优化器。**

两件事可以同时用，但管的不是同一个旋钮。下面从头说清楚。

---

## 11. 它卡在训练流水线的哪一步？

一个训练 step 通常按这个顺序走：

1. **前向**：输入 batch，算出 loss  
2. **反向**：对每个可训练参数算出梯度，写进 `param.grad`  
3. **梯度裁剪（本题）**：看全体梯度合起来是否过大；过大就原地缩小  
4. **优化器 `step()`**：用当前学习率 $\alpha_t$ 和（可能已被裁过的）梯度去改参数  

所以：

- 裁剪 **不改** loss  
- 裁剪 **不改** 学习率  
- 裁剪只改 **已经算好的梯度**  
- 而且是 **原地** 改每个参数的 `.grad`

若跳过第 3 步，优化器直接吃原始梯度；若做了第 3 步，优化器吃的是裁过之后的梯度。

---

## 12. 为啥要裁？「大梯度」到底怕什么？

优化器更新的粗形式可以想成：

$$
\theta \leftarrow \theta - \alpha_t \cdot g
$$

其中 $g$ 是梯度（或 Adam 里由梯度变来的更新方向）。

若某一步碰到：

- 特别难 / 特别怪的训练样本  
- 数值毛刺  
- 训练早期还不稳  

某些参数的梯度可能突然变得非常大。  
于是 $\alpha_t \cdot g$ 这一步会 **跨得特别远**：

- loss 可能瞬间飙高  
- 训练剧烈震荡  
- 极端时出现 `NaN` / `Inf`

梯度裁剪的目标不是「换一个更好的方向」，而是：

> **方向大致保持不变，但把这一步的总力度压到上限以内。**

---

## 13. 「全体梯度的 $\ell_2$ 范数」是什么意思？

模型有很多参数张量：embedding、每一层的 $W_Q,W_K,\ldots$、LM head 等等。  
反传之后，每个张量各有一份同形状的 `.grad`。

梯度裁剪里的 $g$，不是「只看某一个参数」，而是：

> 把 **所有参数的梯度里的每一个数** 都拿出来，想象摊平成一个超长向量，再对这个超长向量算 $\ell_2$ 范数。

记这个超长向量为 $g$，则

$$
\|g\|_2
=
\sqrt{\sum_i g_i^2}
$$

其中求和跑过 **全部参数、全部元素** 的梯度分量。

$\|g\|_2$ 是一个标量，可以把它想成：

> 「这一整步，所有参数合在一起，梯度有多猛」的一把尺子。

本题还给定一个上限：

$$
M = \text{max\_l2\_norm}
$$

以及数值稳定用的小常数（PyTorch 默认，本题也用）：

$$
\varepsilon = 10^{-6}
$$

---

## 14. 裁剪规则：就两支，没有第三支

算出 $\|g\|_2$ 之后：

### 情况 A：不够大，不用裁

若

$$
\|g\|_2 \le M
$$

则 **什么都不做**。每个参数的 `.grad` 保持原样。

### 情况 B：太大了，整体按同一比例缩小

若

$$
\|g\|_2 > M
$$

则先算一个缩放系数：

$$
\mathrm{scale}
=
\frac{M}{\|g\|_2 + \varepsilon}
$$

再对 **每一个** 参数的梯度做：

$$
g \leftarrow g \cdot \mathrm{scale}
$$

也就是：每个 `.grad` 里的每个数，都乘同一个 `scale`。

几点必须钉死：

1. **全局一起量、一起缩**（global clipping）  
   不是每个参数各自算自己的范数再各自裁。  
   是先看全体 $\|g\|_2$，再给全体乘同一个系数。

2. **方向（相对比例）基本不变**  
   原来某个分量是另一个分量的 2 倍，裁完仍是 2 倍；只是整体变短了。

3. **为什么分母是 $\|g\|_2+\varepsilon$，不是 $\|g\|_2$？**  
   防止 $\|g\|_2$ 极端接近 0 时除零。  
   正常「需要裁」的时候 $\|g\|_2$ 很大，$\varepsilon$ 几乎不影响。

4. **裁完后的范数会略小于 $M$**  
   因为

$$
\|g_{\mathrm{new}}\|_2
=
\mathrm{scale}\cdot\|g\|_2
=
\frac{M}{\|g\|_2+\varepsilon}\cdot\|g\|_2
=
M\cdot\frac{\|g\|_2}{\|g\|_2+\varepsilon}
< M
$$

讲义说 resulting norm will be **just under** $M$，指的就是这件事。

---

## 15. 玩具例子（建议跟着手算一遍）

假设模型只有 **两个参数张量**（故意做得很小，方便心算）：

$$
g^{(1)} = [3,\,4],
\qquad
g^{(2)} = [12]
$$

上限取

$$
M = 5,
\qquad
\varepsilon = 10^{-6}
$$

（真实训练里 $M$ 常见是 1 之类；这里为了数字好看选 5。）

### 15.1 先算全体 $\ell_2$ 范数

把所有分量摊在一起：$3,\,4,\,12$。

$$
\|g\|_2
=
\sqrt{3^2 + 4^2 + 12^2}
=
\sqrt{9 + 16 + 144}
=
\sqrt{169}
=
13
$$

### 15.2 和 $M$ 比较

$$
13 > 5
$$

所以要裁。

### 15.3 算缩放系数

$$
\mathrm{scale}
=
\frac{M}{\|g\|_2+\varepsilon}
=
\frac{5}{13 + 10^{-6}}
\approx
\frac{5}{13}
\approx 0.384615
$$

（$\varepsilon$ 相对 13 可以忽略，心算时当成 $5/13$ 即可。）

### 15.4 每个梯度乘同一个 scale

$$
\begin{aligned}
g^{(1)}
&\leftarrow
0.384615 \cdot [3,\,4]
\approx
[1.153846,\, 1.538462]
\\[0.5em]
g^{(2)}
&\leftarrow
0.384615 \cdot [12]
\approx
[4.615385]
\end{aligned}
$$

### 15.5 核对：裁完范数是不是略小于 $M$

$$
\begin{aligned}
\|g_{\mathrm{new}}\|_2
&\approx
\sqrt{1.153846^2 + 1.538462^2 + 4.615385^2}
\\
&\approx
\sqrt{1.331 + 2.367 + 21.302}
\\
&\approx
\sqrt{25.000}
\\
&= 5
\end{aligned}
$$

若严格带上 $\varepsilon$，会得到

$$
\|g_{\mathrm{new}}\|_2
=
5\cdot\frac{13}{13+10^{-6}}
< 5
$$

也就是 **刚好略低于 $M$**。心算忽略 $\varepsilon$ 时会看到约等于 5；实现里按公式带 $\varepsilon$ 即可。

### 15.6 再看一个「不用裁」的对照

若同样两个张量，但梯度是：

$$
g^{(1)} = [1,\,2],
\qquad
g^{(2)} = [2]
$$

则

$$
\|g\|_2
=
\sqrt{1+4+4}
=
\sqrt{9}
=
3
$$

因为 $3 \le 5$，**整步梯度原样保留**，不乘任何 scale。

---

## 16. 和学习率调度有什么不同？

| | Cosine LR schedule（上文） | Gradient clipping（本节） |
|--|--|--|
| 管什么 | 这一步的学习率 $\alpha_t$ | 这一步梯度 $g$ 会不会过大 |
| 何时变 | 主要由步数 $t$ 决定 | 每个 step 看当前 $\|g\|_2$ |
| 改谁 | 优化器里用的 $\mathrm{lr}$ | 各参数的 `.grad`（原地） |
| 典型动机 | 前期大胆、后期精细 | 防偶发爆炸梯度 |

可以同时用：

- schedule 管「长期节奏」  
- clipping 管「单步安全阀」  

顺序仍是：先 backward → 再 clip → 再（用当前 $\alpha_t$）optimizer step。

---

## 17. 实现时要对着写的流程（无跳步）

函数输入通常是：

- `parameters`：参数列表（或可迭代对象），每个是 `nn.Parameter`  
- `max_l2_norm`：就是 $M$  
- 固定 $\varepsilon = 10^{-6}$

步骤：

1. 遍历所有参数，跳过 `grad is None` 的。  
2. 把每个 `.grad` 的元素平方求和，全部加总，再开方，得到 $\|g\|_2$。  
   （实现上常写成：先 `total = sum(grad.detach().float().norm(2)**2 for ...)`，再 `total.sqrt()`；或等价地用 `torch.nn.utils.clip_grad_norm_` 同类逻辑。作业要求自己写，不要只调库函数糊弄——以测试 / 讲义为准。）  
3. 若 $\|g\|_2 \le M$，直接返回。  
4. 若 $\|g\|_2 > M$，算

$$
\mathrm{scale} = \frac{M}{\|g\|_2 + \varepsilon}
$$

5. 再遍历一遍，对每个非空 `.grad` **原地**乘上 `scale`（例如 `grad.mul_(scale)`）。

适配器 `adapters.run_gradient_clipping` 接上你的函数后，用：

```bash
uv run pytest -k test_gra
```

核对是否通过。

---

## 18. 一句话收束（梯度裁剪）

- 反传之后、`step` 之前，量全体梯度的 $\|g\|_2$。  
- $\le M$：不动。  
- $> M$：全体乘 $\dfrac{M}{\|g\|_2+\varepsilon}$，范数被压到略低于 $M$。  
- 这是全局按比例缩小，不是改方向，也不是改学习率。
