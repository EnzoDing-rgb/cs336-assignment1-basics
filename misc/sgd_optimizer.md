# PyTorch 里实现 SGD（讲义 §4.2.1）

符号与代码跟 CS336 assignment 讲义一致。  
公式用 `$...$` / `$$...$$`。

---

## 0. 我们现在要解决什么问题？

前面已经会算损失（例如交叉熵）：给定当前参数，得到一个标量 $\ell$。

训练还差一步：**根据损失对参数的梯度，把参数改小一点，让下次损失更小。**  
负责「怎么改参数」的模块，就叫 **优化器（optimizer）**。

本笔记要讲清楚：

1. 一次参数更新在数学上是什么；
2. PyTorch 为什么用 `Optimizer` 这种写法；
3. 讲义里带衰减学习率的 SGD 公式；
4. 对应代码每一行在干什么；
5. 它如何嵌进最小训练循环。

---

## 1. 从第一性看：一次更新在干什么？

把模型里所有可训练参数看成一个大向量 $\theta$（theta）。  
当前在一个小批量（batch）$B_t$ 上算出损失 $L(\theta_t; B_t)$，再反传得到梯度

$$
g_t = \nabla L(\theta_t; B_t)
$$

其中：

- $\theta_t$：第 $t$ 步开始时的参数（$t$ 从 0 起算）
- $B_t$：第 $t$ 步用的那一小批数据
- $\nabla L$：损失对参数的梯度；每个参数张量上对应一块 `p.grad`
- $g_t$：告诉你「往哪个方向走，损失上升最快」；训练时通常 **反着走**

最简单的梯度下降一步是：

$$
\theta_{t+1} = \theta_t - \eta_t \, g_t
$$

其中 $\eta_t$（eta）叫 **学习率（learning rate）**：这一步沿梯度反方向走多远。

讲义里的 SGD 变体把学习率设成随时间衰减：

$$
\theta_{t+1}
= \theta_t - \frac{\alpha}{\sqrt{t+1}} \nabla L(\theta_t; B_t)
\tag{讲义式 20}
$$

其中：

- $\alpha$（alpha）：初始学习率，代码里的 `lr`
- $t$：当前迭代编号，从 0 开始
- $\dfrac{\alpha}{\sqrt{t+1}}$：第 $t$ 步实际用的学习率；随 $t$ 增大而变小

直觉：早期步子可以大一点，后期步子变小，更新更稳。

---

## 2. 为什么不手写 `param -= lr * grad`，而要弄一个 Optimizer 类？

可以手写，但训练稍复杂就会重复很多事：

- 模型有很多参数张量，要逐个更新；
- 不同参数组可能要用不同学习率；
- Adam 等算法还要给每个参数存动量等 **状态（state）**；
- 要统一提供 `zero_grad()`、`step()` 这类接口。

PyTorch 约定：自定义优化器 = 继承 `torch.optim.Optimizer`，至少实现：

- `__init__`：登记要优化哪些参数、默认超参数是什么；
- `step`：在已有 `p.grad` 的前提下，原地更新参数。

讲义先用「带 $\sqrt{t+1}$ 衰减的 SGD」把这套 API 走通；后面 AdamW 也走同一套骨架。

---

## 3. `__init__`：优化器启动时登记什么？

典型签名：

```python
def __init__(self, params, lr=1e-3):
```

### 3.1 `params` 是什么？

`params`：要优化的参数集合。常见两种给法：

1. 一个可迭代对象，例如 `model.parameters()` 或 `[weights]`；
2. 若干 **参数组（param group）** 的字典列表，每组可以有自己的超参数（例如不同学习率）。

若你只传入一堆 `torch.nn.Parameter`，基类构造函数会自动做成 **一个** 参数组，并填上默认超参数。

### 3.2 `defaults` 是什么？

超参数要放进一个字典再交给基类，例如：

```python
defaults = {"lr": lr}
super().__init__(params, defaults)
```

之后在 `step` 里可以从每个 `group["lr"]` 读出学习率。

### 3.3 学习率为负要报错

```python
if lr < 0:
    raise ValueError(f"Invalid learning rate: {lr}")
```

负学习率意味着沿梯度同方向走，损失往往会上升，通常视为非法配置。

---

## 4. `step`：在梯度已经算好之后，如何改参数？

调用时机（见第 6 节训练循环）：

1. `loss.backward()` 已经跑完 → 每个参数的 `p.grad` 里是 $\nabla L$；
2. 再调用 `opt.step()` → 按公式更新 $p.data$。

### 4.1 `closure` 是什么？

```python
def step(self, closure: Optional[Callable] = None):
    loss = None if closure is None else closure()
```

`closure`：一个可调用对象，调用它会重新前向并返回损失。  
少数算法（如 LBFGS）需要反复重算损失；SGD / Adam 通常 **用不到**。  
讲义保留这个参数，是为了符合 `Optimizer` 的接口约定。

### 4.2 双层循环：先参数组，再参数

```python
for group in self.param_groups:
    lr = group["lr"]
    for p in group["params"]:
        ...
```

- `self.param_groups`：基类维护的参数组列表；
- 每个 `group` 里有 `"params"` 和该组超参数（如 `"lr"`）；
- 内层 `p`：一个 `torch.nn.Parameter`（可训练张量）。

### 4.3 没有梯度就跳过

```python
if p.grad is None:
    continue
```

有的参数不要求梯度，或本步没参与损失，就没有 `grad`，不能更新。

### 4.4 `state`：每个参数自己的记事本

```python
state = self.state[p]
t = state.get("t", 0)
```

- `self.state`：优化器内部字典，键是参数张量 `p`；
- `state[p]`：只属于这个参数的持久状态；
- 这里存的是迭代计数 `t`。第一次还没有时，用 `.get("t", 0)` 得到 0。

Adam 一类算法会在同一个 `state` 里存一阶矩、二阶矩等；SGD 变体至少需要 `t`。

### 4.5 真正的更新（对应式 20）

```python
grad = p.grad.data
p.data -= lr / math.sqrt(t + 1) * grad
state["t"] = t + 1
```

逐项对应：

| 代码 | 数学 |
|------|------|
| `lr` | $\alpha$ |
| `t` | 当前步编号（从 0 开始） |
| `lr / math.sqrt(t + 1)` | $\alpha / \sqrt{t+1}$ |
| `grad` | $\nabla L$ 在该参数上的块 |
| `p.data -= ...` | $\theta \leftarrow \theta - \eta_t g$（原地修改） |
| `state["t"] = t + 1` | 下一步用 $t+1$ |

「原地（in-place）」：直接改 `p.data` 里的数，不是返回一个新张量再替换引用。

---

## 5. 讲义中的完整 SGD 类（带注释阅读顺序）

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 取出该组学习率 α
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # 该参数的状态字典
                t = state.get("t", 0)  # 迭代编号；没有则从 0 开始
                grad = p.grad.data  # ∂L/∂p

                # θ ← θ - (α / √(t+1)) * grad
                p.data -= lr / math.sqrt(t + 1) * grad

                state["t"] = t + 1  # 步数加一，供下次 step 使用

        return loss
```

阅读顺序建议：

1. `__init__` 如何把 `lr` 放进 `defaults` 并 `super().__init__`；
2. `step` 如何遍历 `param_groups` → `params`；
3. `state["t"]` 如何读写；
4. 更新式如何对应讲义式 20。

---

## 6. 最小训练循环：优化器如何被调用？

讲义示例（把「损失」简化成权重平方的均值，只为演示接口）：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad()                 # ① 清梯度
    loss = (weights ** 2).mean()    # ② 前向：算标量损失
    print(loss.cpu().item())
    loss.backward()                 # ③ 反传：写入 weights.grad
    opt.step()                      # ④ 按公式更新 weights
```

### 6.1 每一步分别在干什么？

| 步骤 | 调用 | 作用 |
|------|------|------|
| ① | `opt.zero_grad()` | 把各参数的 `.grad` 清零。PyTorch 默认会 **累加** 梯度；不清零的话，本步梯度会叠加上一步的。 |
| ② | 算 `loss` | 前向得到标量。这里 `loss = mean(weights²)`，最小值在 `weights=0` 附近。 |
| ③ | `loss.backward()` | 自动微分，把 $\partial L/\partial\mathrm{weights}$ 写进 `weights.grad`。 |
| ④ | `opt.step()` | 用当前 `grad` 和状态里的 `t`，按式 20 更新 `weights.data`，并令 `t ← t+1`。 |

### 6.2 和语言模型训练的对应关系

把上面的玩具损失换成交叉熵，把 `weights` 换成 `model.parameters()`，骨架不变：

```text
取一个 batch
→ opt.zero_grad()
→ logits = model(token_ids)
→ loss = cross_entropy(...)
→ loss.backward()
→ opt.step()
```

优化器 **不负责** 算损失；它只在 `backward` 之后，根据已有梯度改参数。

### 6.3 跑这个循环时期望看到什么？

`loss = mean(weights²)`，梯度大约指向「让权重靠近 0」。  
学习率虽随 $\sqrt{t+1}$ 衰减，前许多步仍会把损失往下拉，打印出来的 `loss` 整体应逐渐变小（具体曲线取决于随机初始值和 `lr`）。

---

## 7. 知识串联（避免断裂）

用一条因果链收束：

1. **目标**：改参数使损失变小。  
2. **信息来源**：`loss.backward()` 给出每个参数的 `p.grad`。  
3. **更新规则**：讲义式 20，$\theta \leftarrow \theta - \frac{\alpha}{\sqrt{t+1}} g$。  
4. **谁执行规则**：`Optimizer.step()`；`t` 存在 `self.state[p]` 里。  
5. **谁准备下一轮**：`zero_grad()` 清掉旧梯度，避免累加污染。  
6. **参数组**：同一优化器里可为不同参数配置不同 `lr` 等超参数；本例只有一组。

---

## 8. 术语速查

| 中文 | 英文 | 含义 |
|------|------|------|
| 参数 | parameter | 可训练张量，`nn.Parameter` |
| 梯度 | gradient | 损失对参数的导数，存在 `p.grad` |
| 学习率 | learning rate | 每步沿梯度反方向走的步长系数 |
| 参数组 | param group | 共享同一套超参数的一组参数 |
| 优化器状态 | optimizer state | 每个参数上持久保存的辅助量（如步数 `t`） |
| 原地更新 | in-place update | 直接修改 `p.data` |
| 闭包 | closure | 可选回调，用于重新计算损失（SGD 通常不用） |

---

## 9. 和作业后续的关系

本文件对应讲义「先实现一个会衰减学习率的 SGD，熟悉 Optimizer API」。  
后面实现 AdamW、学习率调度时：

- 仍然继承 `torch.optim.Optimizer`；
- 仍然在 `step` 里读 `p.grad`、写 `p.data`；
- 差别主要在 **更新公式** 和 **`state` 里多存什么**。

先把本节的 `__init__` / `step` / `state["t"]` / 训练四步循环啃熟，后面不会断层。
