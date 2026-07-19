# 优化器笔记：讲义 SGD → `state[p]` / `param_groups` → AdamW

本文合并原 `sgd_optimizer.md` 与本文内容，一条线讲完：

1. 一次参数更新在数学上是什么，以及讲义带衰减学习率的 SGD；
2. PyTorch 为什么用 `Optimizer` 这种写法；`__init__` / `step` / 训练循环；
3. Adam 相对纯 SGD 多了什么、学习率怎样保留；动量线 + 自适应步长线怎么拼起来；
4. `state[p]` 是什么、里面有什么、梯度和它什么关系；
5. `param_groups` 在分什么组；
6. AdamW 实现流程（步骤清单，便于动手）。

公式用 `$...$` / `$$...$$`。  
**约定：每个符号第一次出现时立刻说明含义；后文沿用同一含义。**

---

## 符号表（先扫一眼）

| 符号 / 名字 | 英文 | 含义 |
|-------------|------|------|
| $p$ | parameter | 模型里的一块可训练参数张量，对应代码里的 `torch.nn.Parameter`（例如整张 embedding 矩阵） |
| $\theta$ | parameters（总称） | 讨论更新公式时，对「当前这块参数」的写法；和 $p$ 指同一类东西，只是公式里常用 $\theta$ |
| $g$ / $g_t$ | gradient | 损失对 $p$（或 $\theta$）的梯度，形状与参数相同；代码里就是 `p.grad`（`loss.backward()` 之后才有） |
| $L$ / $\ell$ | loss | 标量损失；小批量上常写成 $L(\theta; B_t)$ |
| $B_t$ | minibatch | 第 $t$ 步用的那一小批数据 |
| $\eta$ / $\eta_t$ | learning rate | 学习率：沿梯度反方向走多远的系数；$\eta_t$ 表示第 $t$ 步实际用的学习率 |
| $\alpha$ | alpha / `lr` | 讲义衰减 SGD 里的初始学习率（代码里的 `lr`）；AdamW 里也常用 $\alpha$ 表示标称学习率 |
| $m$ | first moment | Adam 为 $p$ 保存的一阶动量，形状与 $p$ 相同 |
| $v$ | second moment | Adam 为 $p$ 保存的二阶动量，形状与 $p$ 相同 |
| $\beta_1, \beta_2$ | betas | Adam 里更新 $m$、$v$ 时的衰减系数，是介于 0 和 1 之间的超参数 |
| $\epsilon$ / $\varepsilon$ | epsilon | 加在分母上的很小正数，用来保证除法数值安全 |
| $\lambda$ | weight decay | AdamW 里解耦 weight decay 的强度 |
| $t$ | step / iteration | 这个参数被优化器更新了多少次（SGD 讲义从 0 起算；AdamW 讲义伪代码从 $t=1$ 起算） |

三件东西分开放（这一段最重要）：

```text
p              =  参数本身（nn.Parameter）
p.grad         =  本步梯度 g（backward 写在参数对象上）
state[p]       =  优化器给这块 p 准备的「记事本」（一个 Python 字典）
state[p]["m"]  =  记事本里的一阶动量矩阵
state[p]["v"]  =  记事本里的二阶动量矩阵
state[p]["t"]  =  记事本里的步数（整数）
```

和 SGD / Adam 的对照（先看结论）：

$$
\text{纯 SGD:}\quad p \leftarrow p - \eta \cdot g
$$

$$
\text{讲义衰减 SGD:}\quad
\theta_{t+1}
= \theta_t - \frac{\alpha}{\sqrt{t+1}} \nabla L(\theta_t; B_t)
$$

$$
\text{Adam（示意）:}\quad p \leftarrow p - \eta \cdot \frac{m}{\sqrt{v}+\epsilon}
$$

学习率仍然出现在公式里。Adam 的差别是：右边从乘 $g$，变成乘 $\dfrac{m}{\sqrt{v}}$。

---

## 1. 我们现在要解决什么问题？

前面已经会算损失（例如交叉熵）：给定当前参数，得到一个标量 $\ell$。

训练还差一步：**根据损失对参数的梯度，把参数改小一点，让下次损失更小。**  
负责「怎么改参数」的模块，就叫 **优化器（optimizer）**。

训练时参数怎么被更新（正向描述）：

- 整网一起前向、一起反传；
- 同一次 `opt.step()` 里，优化器遍历各个参数 $p$ 并更新它们；
- `state[p]` 保存的是「为了下次还能接着算」而留下的历史量；
- `param_groups` 保存的是「这一组参数用哪套超参数」（例如哪个学习率）。

---

## 2. 从第一性看：一次更新在干什么？

把模型里所有可训练参数看成一个大向量 $\theta$（theta）。  
当前在一个小批量（batch）$B_t$ 上算出损失 $L(\theta_t; B_t)$，再反传得到梯度

$$
g_t = \nabla L(\theta_t; B_t)
$$

其中：

- $\theta_t$：第 $t$ 步开始时的参数（讲义衰减 SGD 里 $t$ 从 0 起算）
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

## 3. 为什么不手写 `param -= lr * grad`，而要弄一个 Optimizer 类？

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

## 4. `__init__`：优化器启动时登记什么？

典型签名：

```python
def __init__(self, params, lr=1e-3):
```

### 4.1 `params` 是什么？

`params`：要优化的参数集合。常见两种给法：

1. 一个可迭代对象，例如 `model.parameters()` 或 `[weights]`；
2. 若干 **参数组（param group）** 的字典列表，每组可以有自己的超参数（例如不同学习率）。

若你只传入一堆 `torch.nn.Parameter`，基类构造函数会自动做成 **一个** 参数组，并填上默认超参数。

### 4.2 `defaults` 是什么？

超参数要放进一个字典再交给基类，例如：

```python
defaults = {"lr": lr}
super().__init__(params, defaults)
```

含义：某个参数组如果没单独写 `lr`，就用这份默认的学习率。  
若你只写 `SGD(model.parameters(), lr=1e-3)`，基类会自动做成 **一个组**，里面所有参数 $p$ 共用这份默认超参数。  
之后在 `step` 里可以从每个 `group["lr"]` 读出学习率。

### 4.3 学习率为负要报错

```python
if lr < 0:
    raise ValueError(f"Invalid learning rate: {lr}")
```

负学习率意味着沿梯度同方向走，损失往往会上升，通常视为非法配置。

---

## 5. `step`：在梯度已经算好之后，如何改参数？

调用时机（见第 7 节训练循环）：

1. `loss.backward()` 已经跑完 → 每个参数的 `p.grad` 里是 $\nabla L$；
2. 再调用 `opt.step()` → 按公式更新 $p.data$。

### 5.1 `closure` 是什么？

```python
def step(self, closure: Optional[Callable] = None):
    loss = None if closure is None else closure()
```

`closure`：一个可调用对象，调用它会重新前向并返回损失。  
少数算法（如 LBFGS）需要反复重算损失；SGD / Adam 通常 **用不到**。  
讲义保留这个参数，是为了符合 `Optimizer` 的接口约定。

### 5.2 双层循环：先参数组，再参数

```python
for group in self.param_groups:
    lr = group["lr"]
    for p in group["params"]:
        ...
```

- `self.param_groups`：基类维护的参数组列表；
- 每个 `group` 里有 `"params"` 和该组超参数（如 `"lr"`）；
- 内层 `p`：一个 `torch.nn.Parameter`（可训练张量）。

### 5.3 没有梯度就跳过

```python
if p.grad is None:
    continue
```

有的参数不要求梯度，或本步没参与损失，就没有 `grad`，不能更新。

### 5.4 `state`：每个参数自己的记事本（先扫一眼）

```python
state = self.state[p]
t = state.get("t", 0)
```

- `self.state`：优化器内部字典，键是参数张量 `p`；
- `state[p]`：只属于这个参数的持久状态；
- 这里存的是迭代计数 `t`。第一次还没有时，用 `.get("t", 0)` 得到 0。

Adam 一类算法会在同一个 `state` 里存一阶矩、二阶矩等；SGD 变体至少需要 `t`。  
更深的「为什么每个 $p$ 一份、$g$ 放在哪」见第 10 节。

### 5.5 真正的更新（对应讲义式 20）

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

代码对齐版（读 `step` 时对照）：

```python
state = self.state[p]              # state：字典记事本
t = state.get("t", 0)             # 从记事本取步数 t
grad = p.grad.data                # 从参数对象取本步梯度 g
p.data -= lr / math.sqrt(t + 1) * grad   # p ← p - (α/√(t+1)) * g
state["t"] = t + 1                # 把新的 t 写回记事本
```

---

## 6. 讲义中的完整 SGD 类（带注释阅读顺序）

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

可运行玩具实验见 `misc/run_lr_tuning_toy.py`（比较不同 `lr` 下损失衰减 / 发散）。

---

## 7. 最小训练循环：优化器如何被调用？

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

### 7.1 每一步分别在干什么？

| 步骤 | 调用 | 作用 |
|------|------|------|
| ① | `opt.zero_grad()` | 把各参数的 `.grad` 清零。PyTorch 默认会 **累加** 梯度；不清零的话，本步梯度会叠加上一步的。 |
| ② | 算 `loss` | 前向得到标量。这里 `loss = mean(weights²)`，最小值在 `weights=0` 附近。 |
| ③ | `loss.backward()` | 自动微分，把 $\partial L/\partial\mathrm{weights}$ 写进 `weights.grad`。 |
| ④ | `opt.step()` | 用当前 `grad` 和状态里的 `t`，按式 20 更新 `weights.data`，并令 $t \leftarrow t+1$。 |

### 7.2 和语言模型训练的对应关系

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

### 7.3 跑这个循环时期望看到什么？

`loss = mean(weights²)`，梯度大约指向「让权重靠近 0」。  
学习率虽随 $\sqrt{t+1}$ 衰减，前许多步仍会把损失往下拉，打印出来的 `loss` 整体应逐渐变小（具体曲线取决于随机初始值和 `lr`）。

---

## 8. 纯 SGD 的麻烦是什么？（为什么还要 Adam）

先约定这一节里的字母：

- $p$：当前正在更新的那一块参数张量（parameter）
- $g$：损失对这块参数的梯度（gradient），也就是 `p.grad`
- $\eta$：学习率（learning rate）

更新规则（逐元素，对 $p$ 的每一个格子做同样形式的事）：

$$
p \leftarrow p - \eta \cdot g
$$

有的笔记写成 $\theta \leftarrow \theta - \eta \cdot g$，这里的 $\theta$ 和 $p$ 是同一类对象：都是「当前这块参数」。

同一个 $\eta$ 要同时服务尺度差很大的梯度 $g$：

- $g$ 里有的维度经常很大 → 一步迈得很猛，训练容易抖；
- $g$ 里有的维度又小又稀疏（例如词表里罕见词对应的 embedding 行，很多步梯度接近 0）→ 几乎走不动。

因此人们希望：在仍然保留一个全局 $\eta$ 的前提下，让每个参数维度自己再调一下有效步长，并且方向更稳。  
Adam 就是冲着这件事来的。

---

## 9. Adam 是怎么来的？

Adam 是两条已经验证有效的线索拼在一起。

### 9.1 动量线：为什么要有 $m$ 和 $\beta_1$？

纯 SGD 每步只看当前梯度 $g$，噪声大、方向容易抖。

更早的做法叫 **动量（momentum）**：

$$
m \leftarrow \beta_1 m + (1-\beta_1)\, g
$$

含义：

- $m$：最近一段时间梯度的指数滑动平均（越近的 $g$ 权重大）
- $\beta_1$：控制「多信历史、多信这一步」；常见默认是 $0.9$（历史占大头）

直觉：

- 物理上像惯性；
- 统计上像对噪声梯度做平滑。

**Adam 里的 $m$，就是把动量这一层搬进来。**  
$m$ 主要管「往哪走」更稳。

### 9.2 自适应步长线：为什么要有 $v$ 和 $g^2$？

另一条独立线索：AdaGrad / RMSProp。

问题：有的维度上 $g$ 经常很大，有的维度上 $g$ 又小又稀，同一个 $\eta$ 顾不过来。

做法：先量每个维度「梯度有多猛」，猛的就把实际步长压小。  
「有多猛」用梯度的平方来量（只关心幅度）：

$$
v \leftarrow \beta_2 v + (1-\beta_2)\, g^{2}
$$

含义：

- $g^{2}$：把 $g$ **逐元素平方**（按每个格子各自平方）
- $v$：每个维度「梯度幅度」的滑动平均
- $\beta_2$：幅度历史的衰减；常见默认是 $0.999$（比 $\beta_1$ 更「记性长」）

更新时除以 $\sqrt{v}$：幅度大的维，实际步长自动变小。

**Adam 里的 $v$，来自这条自适应学习率线索（更接近 RMSProp）。**  
「历史上太猛就除以根号压一压」——主要是 $v$ / $\sqrt{v}$ 在干。

### 9.3 Adam = 动量 + 自适应步长

把两条线合在一起：

$$
m \leftarrow \beta_1 m + (1-\beta_1)\, g
$$

$$
v \leftarrow \beta_2 v + (1-\beta_2)\, g^{2}
$$

$$
p \leftarrow p - \eta \cdot \frac{m}{\sqrt{v}+\epsilon}
$$

| 零件 | 从哪来 | 干什么 |
|------|--------|--------|
| $\eta$ | SGD | 全局学习率仍然保留 |
| $m$、$\beta_1$ | Momentum | 方向更稳 |
| $v$、$\beta_2$、$g^{2}$ | RMSProp / AdaGrad 一系 | 按维度缩放步长 |
| $\epsilon$ | 工程细节 | 保证除法数值安全 |

读最终更新式：

- 分子 $m$：带惯性的梯度方向；
- 分母 $\sqrt{v}+\epsilon$：按该维度历史梯度大小缩放步长；
- $v$ 大 → 分母大 → 实际步长变小；
- $v$ 小 → 分母小 → 实际步长变大。

一句话直觉：

- $m$ = 平滑后的方向；
- $v$ = 历史梯度有多猛；
- $m/\sqrt{v}$ = 按「猛不猛」缩放过的步子；
- 最后仍然乘上学习率 $\eta$。

2014 年的 Adam 论文，核心是：把已经有用的 **动量** 和 **按维自适应步长** 拼成一个默认好用的优化器，并补上训练初期的 bias correction（刚开始 $m$、$v$ 偏小的修正）。

$\beta_1 = 0.9$、$\beta_2 = 0.999$ 是论文推荐的默认值，后来大量实验证明够用。它们是 **经验默认**。

### 9.4 公式为什么会显得「有经验成分」？

1. 最终形式是工程上的合成；
2. 日常几乎总是用默认的 $\beta_1$、$\beta_2$；
3. $g^{2}$、$\sqrt{v}$ 的动机很朴素：只量「这个维有多猛」，再反过来压步长。

零件各自的动机很清楚：先有「加惯性」，再有「大梯度维走小步」，再拼成 Adam。

---

## 10. `state[p]` 到底是什么？

### 10.1 先说结论：`state[p]` 是一个字典，里面可以有多份张量

在代码里：

```python
state = self.state[p]   # 这是一个 dict；里面可以装多份张量和标量
```

对 Adam 来说，这个字典里常见字段是：

```text
state[p] = {
  "m":  与 p 同形状的张量,   # 一阶动量
  "v":  与 p 同形状的张量,   # 二阶动量
  "t":  整数,               # 步数
}
```

所以：

- `state[p]` 本身 = 记事本（字典）；
- `state[p]["m"]` = 记事本里的那一份 $m$ 矩阵；
- `state[p]["v"]` = 记事本里的那一份 $v$ 矩阵；
- `state[p]["t"]` = 记事本里的步数。

你的理解对：`state[p]` 取出来往往是 **一整包东西**，里面可以有多个矩阵，外加一些标量。

### 10.2 梯度 $g$ 在哪里？在 `p.grad`，在 `state[p]` 外面

梯度 **放在参数对象自己身上**：

```text
g  =  p.grad
```

`loss.backward()` 算完后，PyTorch 把本步梯度写进 `p.grad`。  
优化器的 `step()` 读的是 `p.grad`，再用它去更新 `state[p]["m"]`、`state[p]["v"]`，最后改 `p.data`。

数据流（Adam 示意）：

```text
1) loss.backward()
      写出   p.grad  (= g)

2) opt.step() 里对这个 p：
      读出   g = p.grad
      读出   m = state[p]["m"]
      读出   v = state[p]["v"]
      更新   m ← β1*m + (1-β1)*g
      更新   v ← β2*v + (1-β2)*(g*g)
      写回   state[p]["m"] = m
      写回   state[p]["v"] = v
      更新   p.data ← p.data - η * m / (√v + ε)
```

因此：

| 你想取什么 | 正确取法 |
|------------|----------|
| 参数本身 | `p` 或 `p.data` |
| 本步梯度 $g$ | `p.grad` |
| 一阶动量 $m$ | `state[p]["m"]` |
| 二阶动量 $v$ | `state[p]["v"]` |
| 步数 $t$ | `state[p]["t"]` |

标准 Adam 实现里，`state[p]` 通常存 $m$、$v$、步数；**本步梯度就留在 `p.grad` 里用一次**，用完后下一轮 `zero_grad()` 会清掉，再由下一次 `backward()` 重新写入。

### 10.3 为什么每个 $p$ 都要自己的一份记事本？

假设模型里至少有两块参数张量（两个不同的 $p$）：

- 第一个 $p$ = `embedding`，形状 `[1000, 64]`
- 第二个 $p$ = `lm_head.weight`，形状 `[50000, 768]`

对 `embedding`：

| 名字 | 形状 | 存放位置 |
|------|------|----------|
| 参数 $p$ | `[1000, 64]` | `embedding` 自身 |
| 梯度 $g$ | `[1000, 64]` | `embedding.grad` |
| 动量 $m$ | `[1000, 64]` | `state[embedding]["m"]` |
| 动量 $v$ | `[1000, 64]` | `state[embedding]["v"]` |

对 `lm_head.weight`：

| 名字 | 形状 | 存放位置 |
|------|------|----------|
| 参数 $p$ | `[50000, 768]` | `lm_head.weight` 自身 |
| 梯度 $g$ | `[50000, 768]` | `lm_head.weight.grad` |
| 动量 $m$ | `[50000, 768]` | `state[lm_head.weight]["m"]` |
| 动量 $v$ | `[50000, 768]` | `state[lm_head.weight]["v"]` |

两套 $m,v$ 分开存，因为：

1. 形状不同，各自需要自己大小的矩阵；
2. 语义不同：embedding 第 $(i,j)$ 格的历史，和 lm_head 第 $(a,b)$ 格的历史是两回事。

写成代码结构就是：

```text
state[embedding] = {
  "m": 形状 [1000, 64] 的张量,
  "v": 形状 [1000, 64] 的张量,
  "t": 整数,
}

state[lm_head.weight] = {
  "m": 形状 [50000, 768] 的张量,
  "v": 形状 [50000, 768] 的张量,
  "t": 整数,
}
```

### 10.4 `state[p]` 的本质（一句话）

**优化器为参数张量 $p$ 保存的私有记忆字典**；Adam 里这份记忆通常包括与 $p$ 同形状的 $m$、$v$，以及步数 $t$。  
本步梯度 $g$ 则挂在 `p.grad` 上，和这份字典分开。

| 算法 | `state[p]` 里典型有什么 |
|------|-------------------------|
| 讲义带衰减的 SGD | 整数 $t$ |
| Adam / AdamW | 与该 $p$ 同形状的 $m$、$v$，以及步数 $t$ |

讲义 SGD 也把 $t$ 放进 `state[p]`，是为了和 Adam 共用「状态挂在参数上」这套接口。

### 10.5 和整网一起 forward / backward 怎样配合？

1. `loss.backward()`：整网一起算，每个 $p$ 的 `p.grad`（也就是 $g$）都写好；
2. `opt.step()`：同一次调用里遍历所有 $p$；
3. 对每个 $p$：从 `p.grad` 读 $g$，从 `state[p]` 读 $m,v,t$，更新后再写回 `state[p]`，并改 `p.data`；
4. 正常训练里，各参数的步数 $t$ 一起增加。

---

## 11. `param_groups` 在分什么？

### 11.1 分的是超参数配置

前向、反向仍然整网一起做。  
分组分的是：**更新时用哪套超参数**（学习率 $\eta$、weight decay 等）。

### 11.2 真实例子：embedding 用更小学习率

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

| 组 | 里面有哪些参数 $p$ | 学习率 $\eta$ |
|----|-------------------|---------------|
| 组 0 | token embedding 里的参数 | $10^{-4}$（更小，更新更保守） |
| 组 1 | Transformer blocks + lm_head 里的参数 | $10^{-3}$ |

同一次 `opt.step()`：两组里的 $p$ 都更新；组 0 用 `group["lr"]=1e-4`，组 1 用 `1e-3`。

另一个常见分法：普通权重做 weight decay，bias / RMSNorm 的缩放向量 $\gamma$ 使用另一套 weight decay 设置——也是不同组配不同超参数。

（`defaults` 的含义见 §4.2：某组没单独写的超参数，用 `__init__` 里那份默认值。）

---

## 12. 两套机制怎么叠在一起？

```text
一次 opt.step()：

  param_groups[0]  (lr=1e-4)              param_groups[1]  (lr=1e-3)
       │                                       │
       └─ p = embedding                        ├─ p = layers.* 里的各参数
            p.grad = g                         └─ p = lm_head
            state[p] = {m, v, t}                    p.grad = g
                                                    state[p] = {m, v, t}
```

- **横轴 group**：同一套超参数（同一个学习率 $\eta$）的一捆参数 $p$；
- **纵轴**：每个 $p` 自己有 `p.grad`（本步梯度）和 `state[p]`（历史记忆）；
- **时间**：同一次 `step` 里一起更新。

---

## 13. 对照速查 / 术语

| 机制 / 中文 | 英文 | 它是什么 / 做什么 |
|-------------|------|-------------------|
| $p$ / 参数 | parameter | 一块可训练张量，`nn.Parameter` |
| $g$ / 梯度 | gradient | 损失对参数的导数，存在 `p.grad` |
| 学习率 | learning rate | 每步沿梯度反方向走的步长系数（$\eta$ / $\alpha$） |
| `state[p]` / 优化器状态 | optimizer state | 优化器给这块 $p$ 的记事本（字典） |
| `state[p]["m"]` / `["v"]` / `["t"]` | moments / step | 记事本里的动量与步数 |
| `param_groups` / 参数组 | param group | 按超参数配置把参数捆成组（例如不同 $\eta$） |
| `defaults` | defaults | 参数组未单独指定时使用的默认超参数 |
| 原地更新 | in-place update | 直接修改 `p.data` |
| 闭包 | closure | 可选回调，用于重新计算损失（SGD 通常不用） |

---

## 14. 知识串联（避免断裂）

用一条因果链收束：

1. **目标**：改参数使损失变小。  
2. **信息来源**：`loss.backward()` 给出每个参数的 `p.grad`。  
3. **更新规则（讲义 SGD）**：式 20，$\theta \leftarrow \theta - \frac{\alpha}{\sqrt{t+1}} g$。  
4. **谁执行规则**：`Optimizer.step()`；`t` 存在 `self.state[p]` 里。  
5. **谁准备下一轮**：`zero_grad()` 清掉旧梯度，避免累加污染。  
6. **参数组**：同一优化器里可为不同参数配置不同 `lr` 等超参数。  
7. **往 Adam / AdamW**：同一套 `Optimizer` 骨架；`state[p]` 里多存 $m$、$v$；更新公式换成矩形式，AdamW 再加解耦的 $\lambda\theta$ 衰减。

---

## 15. AdamW 实现流程（先走通步骤，再写代码）

前置条件已经齐了：讲义衰减 SGD 把 `Optimizer` 骨架走通了；本文把 `state[p]`、`param_groups`、$m$/$v$ 的含义也对齐了。  
**可以开始实现 AdamW。** 下面是推荐的落地顺序（按讲义 Algorithm 1）。

### 15.1 作业要交什么

| 项 | 内容 |
|----|------|
| 类 | `torch.optim.Optimizer` 的子类（例如 `AdamW`） |
| `__init__` 超参数 | 学习率 $\alpha$；$\beta=(\beta_1,\beta_2)$；$\varepsilon$；weight decay $\lambda$ |
| 状态 | 每个 $p$ 在 `self.state[p]` 里存一阶矩 $m$、二阶矩 $v$、步数 $t$ |
| 适配器 | `adapters.get_adamw_cls` 返回这个类 |
| 验收 | `uv run pytest -k test_adamw`（与参考实现或 `torch.optim.AdamW` 之一对齐即可） |

测试里常见调用写法（名字要和 PyTorch 习惯对齐）：

```text
AdamW(params, lr=α, betas=(β1, β2), eps=ε, weight_decay=λ)
```

### 15.2 符号与讲义一步在干什么

| 符号 | 含义 |
|------|------|
| $\theta$ / $p$ | 当前这块参数 |
| $g$ | 本步梯度 `p.grad` |
| $\alpha$ | 标称学习率（`group["lr"]`） |
| $\beta_1,\beta_2$ | 更新 $m$、$v$ 的衰减系数 |
| $\varepsilon$ | 分母上的小常数，保证数值稳定 |
| $\lambda$ | weight decay 强度（`group["weight_decay"]`） |
| $t$ | 该参数被更新的次数；讲义从 $t=1$ 起算 |
| $\alpha_t$ | 带 bias correction 的有效学习率 |

讲义每一步（对一块 $\theta$）的顺序：

1. 已有 $g$（假定 `backward` 已完成）。
2. 算 bias-corrected 学习率：
   $$
   \alpha_t \leftarrow \alpha \cdot \frac{\sqrt{1-\beta_2^{t}}}{1-\beta_1^{t}}
   $$
3. **解耦的 weight decay**（先拉向 0，与梯度更新分开）：
   $$
   \theta \leftarrow \theta - \alpha \lambda \theta
   $$
4. 更新矩：
   $$
   m \leftarrow \beta_1 m + (1-\beta_1) g,\qquad
   v \leftarrow \beta_2 v + (1-\beta_2) g^{2}
   $$
5. 用矩更新参数：
   $$
   \theta \leftarrow \theta - \alpha_t \cdot \frac{m}{\sqrt{v}+\varepsilon}
   $$

和「普通 Adam + L2」的差别：$\lambda$ 直接乘在 $\theta$ 上衰减，**不**混进 $g$ 里再进 $m$/$v$。

### 15.3 推荐实现步骤

1. **落文件与骨架**  
   新建优化器模块（或放在已有 optimizer 文件），写 `class AdamW(torch.optim.Optimizer)`。  
   `__init__`：校验 $\alpha,\varepsilon,\lambda \ge 0$ 且 $0\le\beta_1,\beta_2<1$；  
   `defaults = {"lr": α, "betas": (β1,β2), "eps": ε, "weight_decay": λ}`；  
   `super().__init__(params, defaults)`。

2. **写 `step` 的双层循环**（与讲义 SGD 相同）  
   ```text
   for group in self.param_groups:
       取出本组 α, (β1,β2), ε, λ
       for p in group["params"]:
           若 p.grad 为空则跳过
           ...
   ```

3. **初始化 `state[p]`（每个 $p$ 第一次进 `step` 时）**  
   - $m \leftarrow 0$（与 $p$ 同形状的零张量）  
   - $v \leftarrow 0$（同上）  
   - $t \leftarrow 0$（之后每次更新前或更新后递增到讲义的 $t=1,2,\ldots$）  
   用 `if len(state) == 0:` 或 `state.get(...)` 只初始化一次。

4. **按讲义顺序写更新**（顺序很重要；CHANGELOG 特意对齐过论文：先 weight decay，再矩更新）  
   - $t \leftarrow t+1$  
   - 读 $g = p.grad$  
   - 算 $\alpha_t$（注意 $\beta_1^{t}$、$\beta_2^{t}$ 的幂是当前 $t$）  
   - $\theta \leftarrow \theta - \alpha\lambda\theta$  
   - 更新 $m$、$v$  
   - $\theta \leftarrow \theta - \alpha_t \cdot m / (\sqrt{v}+\varepsilon)$  
   - 把新的 $m$、$v$、$t$ 写回 `state[p]`

5. **接上适配器**  
   `get_adamw_cls()` 直接 `return AdamW`（或你们模块里的类名）。

6. **跑测试并对照**  
   `uv run pytest -k test_adamw`  
   若数值差一点：优先核对 **weight decay 与矩更新的先后**、**$t$ 从 1 起算**、**$\alpha_t$ 公式里分子分母是否写反**、以及 $g^{2}$ 是否逐元素平方。

### 15.4 实现时一眼对照表

| 讲义 | 代码落点 |
|------|----------|
| $\theta$ | `p.data`（或对 $p$ 的 in-place 更新） |
| $g$ | `p.grad`（本步；不进 `state` 长期存） |
| $m,v,t$ | `self.state[p]["m"]` 等 |
| $\alpha,\beta,\varepsilon,\lambda$ | 当前 `group[...]` |
| 样本 / loss / backward | 训练循环负责；`step` 只读 `p.grad` 并改 $\theta$ |

### 15.5 和本文前面章节的衔接

- 骨架：§3–§7 的 `Optimizer` + `step` 双循环 + 训练四步。  
- 记忆与分组：§10–§12 的 `state[p]` / `param_groups`。  
- Adam 动机：§8–§9 的 $m$（方向平滑）+ $v$（按维缩放）；AdamW 再加解耦的 $\lambda\theta$ 衰减。
