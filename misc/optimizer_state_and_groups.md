# 优化器里的 `state[p]` 与 `param_groups`：为什么要这样设计？

本文补上 `misc/sgd_optimizer.md` 里容易断裂的两块：

1. 为什么每个参数要有一份 `state[p]`（用 Adam 把动机讲透）；
2. `param_groups` 到底在分什么组。

公式用 `$...$` / `$$...$$`。

---

## 0. 先钉死两句，避免误解

- 训练时整网仍然 **一起** 前向、一起反传、同一次 `opt.step()` 里更新。
- `state[p]` **不是**「不同模块活在不同时间线」。
- `param_groups` **不是**「有的层先算、有的层后算」。

这两套机制解决的是别的问题。下面用具体形状和公式说明。

---

## 1. 纯 SGD 的麻烦是什么？

更新规则很简单：

$$
\theta \leftarrow \theta - \eta \cdot g
$$

- $\theta$：某个参数张量里的一个数（或整块张量，逐元素更新）
- $g$：对应的梯度（来自 `p.grad`）
- $\eta$：学习率（learning rate），全局共用一个数（或每个参数组共用一个数）

麻烦在于：**同一个 $\eta$ 要伺候尺度差很大的梯度。**

常见情况：

- 有的维度梯度经常很大 → 一步迈太猛，训练抖；
- 有的维度梯度又小又稀疏（例如词表里罕见词对应的 embedding 行，很多步梯度接近 0）→ 几乎走不动。

人会希望：在「全局还是有一个 $\eta$」的前提下，让 **每个参数维度自己再调一下有效步长**。  
Adam 就是冲着这件事来的。

---

## 2. Adam 多记住的两样东西：一阶动量 $m$、二阶动量 $v$

对 **每一个参数张量** $p$，Adam 在优化器里额外保存两块 **与 $p$ 形状完全相同** 的张量。

### 2.1 一阶动量 $m$（first moment）

每一步（示意，忽略 bias correction）：

$$
m \leftarrow \beta_1 m + (1-\beta_1)\, g
$$

含义：不要只看这一步的 $g$（噪声大），看最近一段时间梯度的指数滑动平均。  
作用：更新方向更稳，少被单步噪声带跑。这就是「动量 / momentum」那一层直觉。

### 2.2 二阶动量 $v$（second moment）

$$
v \leftarrow \beta_2 v + (1-\beta_2)\, g^{2}
$$

这里 $g^{2}$ 是 **逐元素平方**。

含义：某个维度历史上梯度绝对值经常很大，$v$ 就大；经常很小，$v$ 就小。

### 2.3 更新时怎么用

示意（忽略 $\epsilon$ 与 bias correction 细节）：

$$
p \leftarrow p - \eta \cdot \frac{m}{\sqrt{v}+\epsilon}
$$

- $v$ 大的维度：分母大 → **实际步长变小**（别抖死）
- $v$ 小的维度：分母小 → **实际步长变大**（别走不动）

**Adam 解决的问题（一句话）：**  
不同参数、不同维度的梯度尺度差很大时，纯 SGD 一个学习率顾不过来；Adam 用 $m$ 稳住方向，用 $v$ 给每个维度自适应步长。

这也是大模型训练里常用 Adam / AdamW 的原因之一。

---

## 3. 为什么必须是 `state[p]`，而不能全网共用一份 $m,v$？

### 3.1 具体形状例子（写死，不用省略号）

假设：

- `embedding` 形状是 `[1000, 64]`（1000 个词，每个 64 维）
- `lm_head.weight` 形状是 `[50000, 768]`（词表 50000，模型宽度 768）

对 `embedding`：

| 名字 | 形状 | 存什么 |
|------|------|--------|
| `embedding` 本身 | `[1000, 64]` | 参数 |
| `embedding.grad` | `[1000, 64]` | 本步梯度 |
| `m_emb` | `[1000, 64]` | 该矩阵每个元素自己的一阶动量 |
| `v_emb` | `[1000, 64]` | 该矩阵每个元素自己的二阶动量 |

对 `lm_head.weight`：

| 名字 | 形状 | 存什么 |
|------|------|--------|
| `lm_head.weight` | `[50000, 768]` | 参数 |
| `lm_head.weight.grad` | `[50000, 768]` | 本步梯度 |
| `m_head` | `[50000, 768]` | 另一套一阶动量 |
| `v_head` | `[50000, 768]` | 另一套二阶动量 |

两套 $m,v$：

1. **形状不同**，塞不进同一块张量；
2. **语义不同**：embedding 第 $(i,j)$ 格的历史，和 lm_head 第 $(a,b)$ 格的历史不是一回事，搅在一起自适应步长会指错对象。

所以 PyTorch 优化器写成：

```text
state[embedding] = {
  "m": 形状 [1000, 64] 的张量,
  "v": 形状 [1000, 64] 的张量,
  "step": 整数步数,
}

state[lm_head.weight] = {
  "m": 形状 [50000, 768] 的张量,
  "v": 形状 [50000, 768] 的张量,
  "step": 整数步数,
}
```

### 3.2 `state[p]` 的本质

**优化器为参数 $p$ 保存的、更新 $p$ 时必须用到的私有记忆。**

| 算法 | `state[p]` 里典型有什么 |
|------|-------------------------|
| 讲义带衰减的 SGD | 主要是一个整数 $t$（迭代次数） |
| Adam / AdamW | 与 $p$ 同形状的 $m$、$v$，以及步数等 |

讲义 SGD 把 $t$ 也放进 `state[p]`，是在演示 **同一套接口**：  
「状态挂在参数上」。对纯 SGD 来说，全网一个全局计数器也够用，各参数的 $t` 会一起加一，看起来有点重复；但这是为了和 Adam 共用抽象。

### 3.3 和「一起 forward / backward」冲不冲突？

不冲突。

1. `loss.backward()`：整网一起算，每个 `p.grad` 都写好；
2. `opt.step()`：同一次调用里遍历所有参数；
3. 更新 `embedding` 时读 `state[embedding]` 的 $m,v$；
4. 更新 `lm_head` 时读 `state[lm_head.weight]` 的 $m,v`；
5. 正常训练里各参数步数一起增加，不是 embedding 停在 $t=3$、lm_head 停在 $t=10$。

---

## 4. `param_groups` 在分什么？

### 4.1 分的是超参数配置，不是计算图

前向、反向仍然整网一起做。  
分组分的是：**更新时用哪套超参数**（学习率、weight decay 等）。

### 4.2 真实例子：embedding 用更小学习率

```python
opt = SGD(
    [
        {"params": model.token_embeddings.parameters(), "lr": 1e-4},
        {
            "params": list(model.layers.parameters()) + list(model.lm_head.parameters()),
            "lr": 1e-3,
        },
    ]
)
```

| 组 | 里面有什么 | 学习率 |
|----|------------|--------|
| 组 0 | token embedding | $10^{-4}$（更小，更新更保守） |
| 组 1 | Transformer blocks + lm_head | $10^{-3}$ |

同一次 `opt.step()`：两组都更新，但组 0 用 `group["lr"]=1e-4`，组 1 用 `1e-3`。

另一个常见分法：普通权重做 weight decay，bias / RMSNorm 的 $\gamma$ 不做——也是不同组配不同超参数。

### 4.3 `defaults` 是什么？

```python
defaults = {"lr": 1e-3}
super().__init__(params, defaults)
```

含义：某个参数组如果没单独写 `lr`，就用这份默认值。  
若你只写 `SGD(model.parameters(), lr=1e-3)`，基类会自动做成 **一个组**，全员共用这份默认超参数。

---

## 5. 两套机制怎么叠在一起？

```text
一次 opt.step()：

  param_groups[0]  (lr=1e-4)              param_groups[1]  (lr=1e-3)
       │                                       │
       └─ embedding                            ├─ layers.*
            state[embedding] = {m, v, t}       └─ lm_head
                                                    state[lm_head] = {m, v, t}
```

- **横轴 group**：同一套超参数（如同一个 `lr`）的一捆参数；
- **纵轴 state[p]**：每个参数张量自己的优化器记忆（Adam 的 $m,v$，或 SGD 的 $t$）；
- **时间**：同一次 `step` 里一起更新。

---

## 6. 对照速查

| 机制 | 解决什么问题 | 不是在解决什么 |
|------|----------------|----------------|
| `state[p]` | 每个参数需要私有历史（Adam 的 $m,v$；SGD 的 $t$） | 不是让不同层处在不同训练时间线 |
| `param_groups` | 不同参数用不同超参数（lr、weight decay） | 不是把前向/反向拆成多段独立传播 |
| `defaults` | 给参数组提供默认超参数 | 不是模型结构的一部分 |

---

## 7. 和讲义代码的衔接

讲义里的衰减 SGD：

```python
state = self.state[p]
t = state.get("t", 0)
p.data -= lr / math.sqrt(t + 1) * grad
state["t"] = t + 1
```

读法：

1. 从「这个参数的记事本」取出步数 $t$；
2. 用本组的 `lr`（来自 `param_groups`）算 $\alpha/\sqrt{t+1}$；
3. 原地更新；
4. 把 $t+1$ 写回记事本。

后面实现 AdamW 时，骨架不变：仍然 `for group in param_groups`、`for p in group["params"]`，只是 `state[p]` 里多出与 $p$ 同形状的 $m$、$v$，更新公式换成 AdamW 那一套。
