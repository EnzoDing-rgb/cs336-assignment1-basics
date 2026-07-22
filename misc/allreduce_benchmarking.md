# 为什么要 benchmark `all_reduce`：课上这段代码在干什么

老师这一段的题目是：

- How to reason about collective operations  
- Sample benchmarking code  

你前面学的是 **all-reduce / reduce-scatter / all-gather 是什么**。  
这一段往前走一步：这些操作 **有多贵**，以及 **怎么用代码把时间测准**。

公式约定：行内 `$...$`，独立成行 `$$...$$`。

---

## 0. 先把上下文接上（你走神也不怕）

### 0.1 分布式训练里，卡和卡要说话

多卡数据并行时，每张卡（一个 **rank**）各自算一批数据的梯度。  
要让所有卡用同一套参数更新，通常要把各卡梯度 **求和（或平均）**，并且 **每张卡都拿到同一份结果**。

这件事就是集体通信里的：

$$
\mathrm{all\text{-}reduce}
$$

（细节见 `misc/allreduce_reducescatter_allgather.md`。）

### 0.2 为什么老师要讲 benchmark

知道 “all-reduce 会求和” 不够。训练变慢时，你要能回答：

1. 慢在 **算** 还是慢在 **通信**？  
2. all-reduce 的时间随 **消息大小**（`num_elements`）、**卡数**（`world_size`）怎么变？  
3. 测出来的时间是不是假的（没 warmup、没同步、各卡起点不齐）？

所以课上给一段 **最小可跑的测速骨架**：用 `torch.distributed` 做一次 `all_reduce`，并示范正确计时习惯。

### 0.3 几个名词（本笔记只用这些）

| 词 | 含义 |
|----|------|
| rank | 当前进程 / 当前 GPU 的编号，$0,1,\ldots,W-1$ |
| world_size $W$ | 一共多少个进程（多少张卡参与） |
| collective | 所有 rank 一起参加的通信（all-reduce 是其中一种） |
| backend | 通信实现（常见 NCCL，走 GPU 网络） |
| warmup | 正式计时前先跑几轮，避开首次初始化的额外开销 |
| synchronize | 等 GPU 上排队的工作真正做完 |
| barrier | 所有 rank 互相等待，对齐到同一时刻再继续 |

---

## 1. 这段代码的整体目标

函数签名（课上）：

```python
def all_reduce(rank: int, world_size: int, num_elements: int):
```

三件事：

1. 在本 rank 上建一个长度为 `num_elements` 的 GPU tensor  
2. 对这个 tensor 做 `dist.all_reduce(..., op=SUM)`  
3. 用正确的同步方式测这段通信花了多久  

`num_elements` 控制 **消息有多大**。  
改它，就能画出 “消息大小 → all-reduce 延迟” 的曲线，用来推理通信行为。

---

## 2. 代码逐段讲

下面按课上结构写完整骨架（含后面变灰的计时部分），每段说明 **为什么要有这一行**。

```python
import time
import torch
import torch.distributed as dist


def all_reduce(rank: int, world_size: int, num_elements: int):
    # ----------------------------------------------------------
    # 1) 初始化分布式环境
    #    告诉进程：我是几号 rank、一共几人、用什么后端通信
    # ----------------------------------------------------------
    setup(rank, world_size)

    # ----------------------------------------------------------
    # 2) 每张卡准备自己的数据
    #    cuda_if_available(rank)：把 tensor 放到该 rank 对应的 GPU 上
    #    注意：这里是随机数；真实训练里往往是梯度
    # ----------------------------------------------------------
    data = torch.randn(
        num_elements,
        device=cuda_if_available(rank),
    )

    # ----------------------------------------------------------
    # 3) Warmup：正式计时前先跑 collective
    #    第一次 all_reduce 可能包含：
    #      - NCCL 建连 / 分配通信缓冲
    #      - CUDA context、kernel 首次编译/加载
    #    这些开销很大，但不代表稳态训练时的通信成本
    #    所以 warmup 里先跑掉，后面测到的才接近“常态”
    # ----------------------------------------------------------
    # Warm up
    dist.all_reduce(
        tensor=data,
        op=dist.ReduceOp.SUM,   # 跨卡求和
        async_op=False,        # False = 同步：调用返回时，这次 all-reduce 已提交完
                               #（GPU 侧是否完全做完，还要看后面的 synchronize）
    )

    # ----------------------------------------------------------
    # 4) 等本卡 GPU 真正做完；再让所有卡对齐
    #    没有 synchronize：CPU 上的 time.time() 可能在 GPU 还没跑完时就记结束
    #    没有 barrier：有的卡已经开跑，有的卡还在前面磨蹭，测到的是“最慢卡+乱序”
    # ----------------------------------------------------------
    torch.cuda.synchronize()
    dist.barrier()

    # ----------------------------------------------------------
    # 5) 正式计时：再做一次（或多次）all-reduce
    # ----------------------------------------------------------
    # Perform all-reduce
    start_time = time.time()

    dist.all_reduce(
        tensor=data,
        op=dist.ReduceOp.SUM,
        async_op=False,
    )
    torch.cuda.synchronize()   # 再次确保 GPU 通信/kernel 结束再停表

    end_time = time.time()
    elapsed = end_time - start_time

    # 通常只在 rank0 打印，避免 W 份重复输出
    if rank == 0:
        print(
            f"world_size={world_size} "
            f"num_elements={num_elements} "
            f"elapsed_s={elapsed:.6f}"
        )
```

---

## 3. 高亮那一行到底做了什么

```python
dist.all_reduce(tensor=data, op=dist.ReduceOp.SUM, async_op=False)
```

含义：

1. 每个 rank 贡献自己的 `data`  
2. 按元素做 **SUM**（也可以是 AVG 等，这里是 SUM）  
3. 求和结果 **写回每个 rank 自己的 `data`**（in-place 常见语义）  
4. `async_op=False`：这个 Python 调用按同步接口返回（配合 `synchronize` 才能和墙钟对齐）

数学上（长度为 $L=\texttt{num\_elements}$）：

$$
\mathrm{data}^{(r)}
\leftarrow
\sum_{r'=0}^{W-1}
\mathrm{data}^{(r')}
\qquad
\text{对每个 rank } r \text{ 都成立}
$$

结束后，所有 rank 上的 `data` 内容相同。

---

## 4. 为什么 warmup / synchronize / barrier 缺一不可

把计时想成秒表。三种常见测歪方式：

| 漏了什么 | 会发生什么 |
|----------|------------|
| 漏 warmup | 第一次把建连、初始化算进通信时间，数字虚高 |
| 漏 `cuda.synchronize()` | CPU 以为结束了，GPU 还在传；测到的偏短或很飘 |
| 漏 `barrier()` | 各卡起步时刻不同，测的是“对齐很差的集体操作”，不是稳态带宽 |

老师强调 “就像以前做 benchmark 一样”，指的就是这套：**warmup → 同步 → 对齐 → 再计时**。

---

## 5. 你拿这段代码要推理什么

测完一组 `(world_size, num_elements, elapsed)` 后，可以问：

1. **消息变大**（`num_elements` 变大）时，时间是否近似线性涨？  
   - 近似线性：更像带宽受限  
   - 很小消息时几乎不掉：更像延迟（latency）受限  

2. **卡数变多**（`world_size` 变大）时，同样消息大小是否更慢？  
   - all-reduce 的通信量和算法实现有关；经验上卡多通常更贵  

3. 训练一步里，若 all-reduce 时间接近甚至超过计算时间：  
   - 缩放效率会变差（加卡不加倍快）  
   - 这时才值得谈：梯度压缩、通信计算重叠、更大 batch、不同并行策略等

所以这段代码不是“展示 API”，而是给你一把尺：**先量通信，再谈优化。**

---

## 6. 和前面三个名词怎么串

$$
\mathrm{All\text{-}reduce}
=
\mathrm{Reduce\text{-}scatter}
+
\mathrm{All\text{-}gather}
$$

- 概念课：结果长什么样（每人最终都有完整和）  
- 本课代码：这个操作在真实多卡上 **跑多久**  
- 工程上：有时库内部用 reduce-scatter + all-gather 实现 all-reduce；你 benchmark 的仍是对外的 `all_reduce` 接口

---

## 7. 最小心智模型（复习）

1. 多卡训练需要集体通信；梯度同步常用 all-reduce。  
2. 不会测，就无法判断慢在算还是慢在传。  
3. 测准三件套：warmup、`cuda.synchronize()`、`barrier()`。  
4. `num_elements` 扫一圈，才能看出延迟区 vs 带宽区。  
5. 优化通信之前，先有一张自己的 benchmark 表。

---

## 8. 带宽公式：每一项乘的是什么（用数字拆开）

课上计时之后还有一段（大意如下）：

```python
dist.barrier()

size_bytes = data.element_size() * data.numel()
sent_bytes = size_bytes * 2 * (world_size - 1)
total_duration = duration
bandwidth = sent_bytes / total_duration

print(
    f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = "
    f"{round(bandwidth / 1024**3)} GB/s",
    flush=True,
)

# Notes:
# - Effective bandwidth ~ 2 * size_bytes / total_duration
# - Independent of world_size
# - Independent of topology (ring or tree)

cleanup()
```

下面只用一组假数字，把每个因子钉死。

### 8.1 先定数字

$$
\begin{aligned}
W &= \texttt{world\_size} = 4 \\
L &= \texttt{num\_elements} = 268435456 \\
&\quad\text{（若 float32，则 } L = 256 \times 2^{20}\text{，约 2.68 亿个元素）}
\end{aligned}
$$

每个 float32 占 4 字节：

$$
\texttt{size\_bytes}
=
4 \times L
=
4 \times 268435456
=
1073741824
=
1\ \mathrm{GiB}
$$

也就是：**这一份要 all-reduce 的向量，单卡上看有 1 GiB。**

再假设你测到：

$$
\texttt{duration} = 0.1\ \mathrm{s}
$$

注意：这里的 `duration` 是 **墙钟时间**（collective 从开始到 GPU 做完的时间）。  
代码里是：

$$
\texttt{total\_duration} = \texttt{duration}
$$

**没有**写成 `world_size * duration`。  
$W$ 只出现在 **字节数** 那一侧，不乘到时间上。

### 8.2 `size_bytes`：一份数据有多大

$$
\texttt{size\_bytes}
=
\texttt{element\_size} \times \texttt{numel}
$$

例子：`element_size=4`，`numel=L`，得到 `1 GiB`。  
这是 **单卡上这份 tensor 的体积**，还没谈“通信走了几趟”。

### 8.3 为什么乘 `(world_size - 1)`

Ring all-reduce 的粗略图景（课上注释说的那种）：

- 有 $W$ 张卡排成环  
- 完整做完一次 all-reduce，大约要走 **$W-1$ 步** 量级的传递  
  （更细的实现是 reduce-scatter $W-1$ 步 + all-gather $W-1$ 步；课上用 $(W-1)$ 把“有多步”先标出来）

本例：

$$
W - 1 = 4 - 1 = 3
$$

所以公式里先出现一个因子 $3$：不是 3 张卡，而是 **大约 3 步传递**。

### 8.4 为什么再乘 `2`（send + receive）

每一步上，一张卡通常既要 **发出** 一块数据，也要 **收到** 一块数据。  
统计“这张卡经手的流量”时，发和收都算：

$$
2 = \text{send} + \text{receive}
$$

于是课上的：

$$
\texttt{sent\_bytes}
=
\texttt{size\_bytes} \times 2 \times (W - 1)
$$

代入数字：

$$
\texttt{sent\_bytes}
=
1\ \mathrm{GiB} \times 2 \times 3
=
6\ \mathrm{GiB}
$$

人话：按这个课上模型，**单卡在这次 all-reduce 里被记成经手了 6 GiB**  
（1 GiB 的数据 × 双向 × 大约 3 步）。

> 说明：更精细的 ring 公式常写成  
> $2\cdot\frac{W-1}{W}\cdot\texttt{size\_bytes}$（每步只传 $\texttt{size\_bytes}/W$ 的 chunk）。  
> 课上这版 **没有除以 $W$**，是刻意简化的“步数 × 双向 × 整包大小”模型，方便先建立数量级直觉。

### 8.5 `bandwidth = sent_bytes / total_duration`

$$
\texttt{bandwidth}
=
\frac{\texttt{sent\_bytes}}{\texttt{duration}}
=
\frac{6\ \mathrm{GiB}}{0.1\ \mathrm{s}}
=
60\ \mathrm{GiB/s}
$$

打印时再除以 $1024^{3}$，得到 “多少 GB/s” 的整数展示。

这一行量的是课上说的 **measured bandwidth（按算法步数估的总线流量 / 时间）**。

### 8.6 Notes 里的 Effective bandwidth：又一个更粗的尺子

课上注释：

$$
\mathrm{Effective\ bandwidth}
\approx
\frac{2 \times \texttt{size\_bytes}}{\texttt{duration}}
$$

代入同一组数：

$$
\frac{2 \times 1\ \mathrm{GiB}}{0.1\ \mathrm{s}}
=
20\ \mathrm{GiB/s}
$$

这里：

- 分子固定是 $2\times\texttt{size\_bytes}$（一份数据的“进+出”直觉）  
- **不再乘 $(W-1)$**  
- 所以注释写：Independent of `world_size`  
- 也不依赖你假设 ring 还是 tree：Independent of topology  

对比：

| 名字 | 公式（课上） | 本例数值 | 把什么算进“流量” |
|------|--------------|----------|------------------|
| measured（代码里的 `bandwidth`） | $\texttt{size\_bytes}\times 2\times(W-1)\ /\ \texttt{duration}$ | $60\ \mathrm{GiB/s}$ | 步数 + 双向 |
| effective（Notes） | $2\times\texttt{size\_bytes}\ /\ \texttt{duration}$ | $20\ \mathrm{GiB/s}$ | 只按“一份数据双向” |

两个都是 **字节 / 秒**；差在分子怎么定义“算多少业务量”。  
老师说“本质上就是带宽”，指的就是这类比值：用流量去除以测到的时间。

### 8.7 最容易混的三件事

1. **`duration` 不是 `world_size * duration`**  
   时间就是你测到的那一次墙钟 `duration`。  
   $W$ 出现在 `sent_bytes = size_bytes * 2 * (W - 1)` 里。

2. **乘 `2` 不是乘 `world_size`**  
   `2` = send + receive。  
   `W-1` = 大约多少步。

3. **`size_bytes` 和 `sent_bytes` 不是一回事**  
   - `size_bytes`：这份 tensor 本身多大（本例 1 GiB）  
   - `sent_bytes`：按模型估计，单卡这次集体操作经手多少字节（本例 6 GiB）

### 8.8 一行收束

$$
\begin{aligned}
\texttt{size\_bytes}
&=
\text{元素字节数} \times \text{元素个数} \\[0.5em]
\texttt{sent\_bytes}
&=
\texttt{size\_bytes}
\times
\underbrace{2}_{\text{发+收}}
\times
\underbrace{(W-1)}_{\text{大约几步}} \\[0.5em]
\texttt{measured bandwidth}
&=
\frac{\texttt{sent\_bytes}}{\texttt{duration}} \\[0.5em]
\mathrm{effective\ bandwidth}
&\approx
\frac{2\times\texttt{size\_bytes}}{\texttt{duration}}
\end{aligned}
$$

---

## 9. 真正用起来：DDP 一步里 all-reduce 梯度（课上这段训练循环）

前面 benchmark 测的是“通信有多贵”。  
课上接下来这段代码回答：**训练一步里，all-reduce 插在哪、干什么。**

字幕那句「每个 rank 都具有相同的梯度」就是这段的结论。

### 9.1 数据并行在干什么（先用人话）

假设 $W=2$ 张卡：

| | rank0 | rank1 |
|--|-------|-------|
| 这一步看到的数据 | batch $B_0$ | batch $B_1$ |
| 前向 + 反向之后 | 得到梯度 $g_0$ | 得到梯度 $g_1$ |

两份数据不同，所以 **本地梯度不同**：$g_0 \neq g_1$。  
若各自直接 `optimizer.step()`，两张卡参数会越走越歪。

数据并行想要的是：等价于（或接近）用 $B_0\cup B_1$ 这一大步的平均梯度来更新。  
于是在 step 之前做：

$$
g \leftarrow \mathrm{AVG}(g_0, g_1) = \frac{g_0 + g_1}{2}
$$

然后 **两张卡都用同一个 $g$** 去 `step()`。  
做完之后，两边参数继续保持一致（若一开始参数就一致，且之后都用同一套更新）。

这就是 DDP 相对“单卡普通训练”多出来的那一步。

### 9.2 课上代码 + 你的四卡例子写进注释

设定（标量梯度，好算）：

$$
\begin{aligned}
\mathrm{rank0.grad} &= 0.4 \\
\mathrm{rank1.grad} &= 0.8 \\
\mathrm{rank2.grad} &= 1.2 \\
\mathrm{rank3.grad} &= 1.6
\end{aligned}
$$

目标（AVG all-reduce）：

$$
g = \frac{0.4+0.8+1.2+1.6}{4} = \frac{4.0}{4} = 1.0
$$

结束后四个 rank 的 `param.grad` 都变成 `1.0`。

```python
# 每个 rank 都有一份自己的参数副本 params，也有自己的优化器。
# 梯度不需要在 rank0「新建一个总梯度 tensor」。
# backward() 之后，每张卡自己的 param.grad 里已经有本地梯度；
# all_reduce 就地改写大家各自手里的那份 param.grad。
params = [...]
optimizer = torch.optim.AdamW(params, lr=1e-3)

for step in range(num_steps):
    # ---------- 1) 前向：本卡自己的 batch ----------
    x = data
    for param in params:
        x = x @ param
        x = F.gelu(x)
    loss = x.square().mean()

    # ---------- 2) 反向：只在本卡写 param.grad ----------
    loss.backward()
    #
    # 你的例子（某个标量参数）此时长这样：
    #   rank0: param.grad = 0.4   ← 在 rank0 自己的显存
    #   rank1: param.grad = 0.8   ← 在 rank1 自己的显存
    #   rank2: param.grad = 1.2
    #   rank3: param.grad = 1.6
    # 还没有任何跨卡通信。

    # ---------- 3) 同步梯度：对外是一次 all_reduce(AVG) ----------
    # Sync gradients across workers
    # (ONLY difference between standard training and DDP)
    for param in params:
        dist.all_reduce(
            tensor=param.grad,     # 就地改这份 grad，不是先拷到 rank0
            op=dist.ReduceOp.AVG,
            async_op=False,
        )
        #
        # 调用结束后（你的例子）：
        #   rank0/1/2/3 的 param.grad 全都是 1.0
        #
        # 对外你只看见这一行。对内 NCCL 常拆成：
        #   all-reduce = reduce-scatter + all-gather
        # 下面 9.3 用 0.4/0.8/1.2/1.6 把两步拆开。

    # ---------- 4) 用已对齐的梯度更新 ----------
    optimizer.step()
    # 四张卡都用 grad=1.0 更新 → 参数继续一致

    print(f"[rank{rank}] loss={loss.item()}")
```

### 9.3 魔法拆开：reduce-scatter 然后 all-gather（不是堆到 rank0）

先回答三个问题：

1. **哪一步是啥**
   - 第一步：reduce-scatter（按块跨卡求和，每卡只留一块）
   - 第二步：all-gather（把各卡手里的块拼齐，人人拿到完整结果）

2. **是不是先把 1、2、3 都加到 0，再分出去？**
   **不是。**
   那是 “reduce 到 rank0 再 broadcast” 的另一条路。
   常见 NCCL / ring 路径是：数据按块在环上流动累加，没有“全世界先在 rank0 上建一个大梯度再发货”。

3. **tensor 怎么变？要不要在 rank0 新建 gradient？**
   - `loss.backward()` 已经在每张卡自己的 `param.grad` 里写好本地梯度
   - `all_reduce(param.grad)` 就地改这份显存
   - 不需要 `rank0_grad = torch.zeros(...)` 这种总控 tensor

下面用 SUM 先把数算清，最后再 `/4` 得到 AVG。

#### 9.3.1 通信前

| rank | 本地 `param.grad` |
|------|-------------------|
| 0 | $0.4$ |
| 1 | $0.8$ |
| 2 | $1.2$ |
| 3 | $1.6$ |

$$
S = 0.4 + 0.8 + 1.2 + 1.6 = 4.0
$$

#### 9.3.2 第一步：reduce-scatter

为了看清 “切开”，用长度为 4 的梯度向量（四个参数，或一个大 grad 切成 4 chunk）。设通信前：

$$
\begin{aligned}
\mathrm{rank0} &= [0.4,\ 0.4,\ 0.4,\ 0.4] \\
\mathrm{rank1} &= [0.8,\ 0.8,\ 0.8,\ 0.8] \\
\mathrm{rank2} &= [1.2,\ 1.2,\ 1.2,\ 1.2] \\
\mathrm{rank3} &= [1.6,\ 1.6,\ 1.6,\ 1.6]
\end{aligned}
$$

按列跨卡求和，每列都是 $4.0$。  
reduce-scatter 之后每人只留一列：

$$
\begin{aligned}
\mathrm{rank0} &\leftarrow [4.0] \\
\mathrm{rank1} &\leftarrow [4.0] \\
\mathrm{rank2} &\leftarrow [4.0] \\
\mathrm{rank3} &\leftarrow [4.0]
\end{aligned}
$$

求和发生在各段自己的归约路径上，不是先 gather 到 rank0。

#### 9.3.3 第二步：all-gather

各卡把手里那一段拼齐：

$$
\begin{aligned}
\mathrm{rank0} &\leftarrow [4.0,\ 4.0,\ 4.0,\ 4.0] \\
\mathrm{rank1} &\leftarrow [4.0,\ 4.0,\ 4.0,\ 4.0] \\
\mathrm{rank2} &\leftarrow [4.0,\ 4.0,\ 4.0,\ 4.0] \\
\mathrm{rank3} &\leftarrow [4.0,\ 4.0,\ 4.0,\ 4.0]
\end{aligned}
$$

这是 SUM all-reduce。AVG 再除以 $W=4$：

$$
[4.0,\ 4.0,\ 4.0,\ 4.0] / 4 = [1.0,\ 1.0,\ 1.0,\ 1.0]
$$

标量版同理：SUM 得 $4.0$，AVG 得 $1.0$，四张卡的 `param.grad` 都是 $1.0$。

#### 9.3.4 和代码一行的对应

```python
dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
```

| 你看见的 | 库内部常做的 |
|----------|----------------|
| 这一行 | reduce-scatter（各段全局和） |
| （仍是这一行） | all-gather（各段拼齐） |
| （仍是这一行） | 若 op=AVG，再 $/W$ |
| 输入输出 | 都是本卡的 `param.grad`，就地更新 |

示意拆解（帮助建立心理模型；不是你手写的 API）：

```python
# ----- 你写的（对外）-----
dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

# ----- 等价拆解（对内，示意）-----
# 输入：每卡 param.grad，例如标量 0.4 / 0.8 / 1.2 / 1.6
chunk = reduce_scatter_sum(param.grad)
# 向量例子：rank0 持有列0 的和=4.0，rank1 持有列1 的和=4.0，...

full = all_gather(chunk)
# 每卡都有完整 SUM：[4,4,4,4]（或标量 4.0）

param.grad.copy_(full / world_size)
# AVG：每卡 param.grad = 1.0
```

### 9.4 标量版收束

通信前：$(0.4),\ (0.8),\ (1.2),\ (1.6)$  
all-reduce(AVG) 后：$(1.0),\ (1.0),\ (1.0),\ (1.0)$  
然后四张卡都用 $1.0$ 做 `optimizer.step()`。

### 9.5 `SUM` 和 `AVG`

$$
\begin{aligned}
\mathrm{SUM}:&\quad g \leftarrow \sum_{r=0}^{W-1} g_r \\[0.4em]
\mathrm{AVG}:&\quad g \leftarrow \frac{1}{W}\sum_{r=0}^{W-1} g_r
\end{aligned}
$$

你的例子：$W=4$，SUM 得 $4.0$，AVG 得 $1.0$。

### 9.6 和真正的 DDP

课上是手写骨架：`backward` → 显式 `all_reduce(AVG)` → `step`。  
`DistributedDataParallel` 把中间那步藏进 `backward()`。

### 9.7 时间线

$$
\begin{aligned}
&1.\ \text{本卡取 batch} \\
&2.\ \text{forward} \rightarrow \text{loss} \\
&3.\ \text{backward} \rightarrow \text{本地 } param.grad\ (0.4/0.8/1.2/1.6) \\
&4.\ \mathrm{all\text{-}reduce(AVG)} \\
&\quad 4\mathrm{a}.\ \mathrm{reduce\text{-}scatter} \\
&\quad 4\mathrm{b}.\ \mathrm{all\text{-}gather} \\
&\quad 4\mathrm{c}.\ /W \rightarrow \text{各卡均为 }1.0 \\
&5.\ \mathrm{optimizer.step()} \\
&6.\ \text{下一 step}
\end{aligned}
$$

---

## 10. 下一段 PPT：参数切开 + forward 里的 `all_gather`

先回答你卡在的那个矛盾。

### 10.0 你的直觉哪里对、哪里差一点

你想的是：

> 权重放不下才切开 → 那拼起来不还是放不下吗？

对了一半：

1. **永远不会把完整权重 $W$ 拼回某一张卡。** 权重切开后，一直切开，各卡只更新自己那一片。
2. **通信、拼起来的东西是「相乘之后的结果」（激活），不是权重。**
3. 这段 PPT 里是 **左右拼接（`cat`）**，不是把结果 **加** 起来。  
   （另一种切法会「加」——那是 row-parallel；课上这页是 column-parallel 风格。）

为什么「激活拼得起来、权重却放不下」说得通？

- 权重：很多层 × 每层大矩阵，再加 optimizer 状态（Adam 大约再 ×2），常年驻留显存。
- 激活：当前这一层算出来的 `x`，形状只是 `[batch, 特征维]`。  
  拼完之后每张卡都有一份完整激活，但 **权重仍然只存 $1/4$**。省的是参数和优化器，不是「激活永远比权重小」这种绝对命题；课堂例子先把机制讲清楚。

数学上为什么「各算一段再拼」等于「整矩阵一次乘」：

$$
x\,W
=
x\,[W_0|W_1|W_2|W_3]
=
[\,xW_0\ |\ xW_1\ |\ xW_2\ |\ xW_3\,]
$$

左边是完整结果；右边是四人各算一块再 `cat`。数值一样，只是谁存哪块 $W$ 不同。

---

### 10.1 用一个小到能手算的例子钉死

设定（全程用这组数）：

| 量 | 取值 | 含义 |
|--|--|--|
| `world_size` | 4 | 4 张 GPU：rank 0,1,2,3 |
| `batch_size` | 1 | 一条样本，方便看 |
| `num_dim` | 4 | 完整特征维 = 4 |
| `local_num_dim` | 1 | 每人只负责 1 维输出，$4/4=1$ |

输入激活（四张卡 **同一份**，因为不是数据并行）：

$$
x_{\mathrm{in}} = \begin{bmatrix} 1 & 2 & 3 & 4 \end{bmatrix}
\quad\text{形状 }[1,4]
$$

完整权重（现实中 **不存在于任何一张卡上**；这里只是上帝视角）：

$$
W = \begin{bmatrix}
10 & 20 & 30 & 40 \\
11 & 21 & 31 & 41 \\
12 & 22 & 32 & 42 \\
13 & 23 & 33 & 43
\end{bmatrix}
\quad\text{形状 }[4,4]
$$

切开方式（按 **列** 切，课上的 `W0 | W1 | W2 | W3`）：

$$
\begin{aligned}
W_0 &= \begin{bmatrix}10\\11\\12\\13\end{bmatrix},\ 
W_1 = \begin{bmatrix}20\\21\\22\\23\end{bmatrix},\ 
W_2 = \begin{bmatrix}30\\31\\32\\33\end{bmatrix},\ 
W_3 = \begin{bmatrix}40\\41\\42\\43\end{bmatrix}
\end{aligned}
$$

每张卡 **只存自己那一列**，形状都是 `[4, 1]`。

若在一台机器上算完整乘（手算核对）：

$$
\begin{aligned}
(xW)_0 &= 1\cdot10 + 2\cdot11 + 3\cdot12 + 4\cdot13 = 120 \\
(xW)_1 &= 1\cdot20 + 2\cdot21 + 3\cdot22 + 4\cdot23 = 220 \\
(xW)_2 &= 1\cdot30 + 2\cdot31 + 3\cdot32 + 4\cdot33 = 320 \\
(xW)_3 &= 1\cdot40 + 2\cdot41 + 3\cdot42 + 4\cdot43 = 420
\end{aligned}
$$

完整结果：

$$
x_{\mathrm{out}} = \begin{bmatrix} 120 & 220 & 320 & 420 \end{bmatrix}
$$

下面看分布式代码如何得到同一个结果，且 **从不组装完整 $W$**。

---

### 10.2 逐行 walkthrough（盯着 rank0；其他 rank 对称）

下面注释里的数字，就是上面例子跑完后的真实内容。

```python
# ---------- 建模型：每人只拿 1/4 权重 ----------
# rank0 本地：
params[layer] = W0 = [[10],
                      [11],
                      [12],
                      [13]]          # 形状 [4, 1] = [num_dim, local_num_dim]
# rank1 只有 W1=[[20],[21],[22],[23]]
# rank2 只有 W2；rank3 只有 W3
# 没有任何一张卡持有完整 4×4 的 W

params = [
    get_init_params(num_dim, local_num_dim, rank)
    for layer in range(num_layers)
]

# ---------- 进入某一层；四卡输入相同 ----------
# 此时每卡上的 x 都是：
#   x = [[1, 2, 3, 4]]               # 形状 [1, 4] = [batch, num_dim]

# ---- 本地 matmul：只用本卡这一片权重 ----
x = x @ params[layer]
# rank0：
#   [[1,2,3,4]] @ [[10],[11],[12],[13]]  →  [[120]]
#   形状从 [1,4] 变成 [1,1] = [batch, local_num_dim]
# rank1 同步算出 [[220]]
# rank2 算出 [[320]]
# rank3 算出 [[420]]
#
# 关键：每人手里只有「完整输出的一截」。
# 权重 W 仍然是切开的，没人去拼 W。

# ---- 为 all_gather 准备 4 个接收坑 ----
activations = [
    torch.empty(batch_size, local_num_dim, device=...)
    for _ in range(world_size)
]
# 现在 activations 是长度 4 的 list，每个元素形状 [1,1]，内容未定义（empty）
# 可以想成：
#   activations = [ ?, ?, ?, ? ]

# ---- all_gather：每人把自己的那截发出去，并收下别人的 ----
dist.all_gather(tensor_list=activations, tensor=x, async_op=False)
# 输入（各卡各自的 tensor=x）：
#   rank0 贡献 [[120]]
#   rank1 贡献 [[220]]
#   rank2 贡献 [[320]]
#   rank3 贡献 [[420]]
#
# 输出（之后 四张卡上的 activations 都变成一模一样）：
#   activations[0] = [[120]]   # 来自 rank0
#   activations[1] = [[220]]   # 来自 rank1
#   activations[2] = [[320]]   # 来自 rank2
#   activations[3] = [[420]]   # 来自 rank3
#
# 注意：这里搬的是激活小块，不是 W0/W1/W2/W3。

# ---- 沿特征维拼回去 ----
x = torch.cat(activations, dim=1)
# 输入：上面四个 [1,1]
# 输出（每张卡上都是）：
#   x = [[120, 220, 320, 420]]      # 形状 [1, 4] = [batch, num_dim]
#
# 这正是「完整 x @ W」的结果。
# 下一层可以把这份完整激活再当作输入；
# 下一层的权重仍然各自只存自己那一片。
```

---

### 10.3 用一张表对照「你以为在搬什么」

| 东西 | 切了吗？ | 会不会拼回一张卡？ | 这段代码里发生了什么 |
|--|--|--|--|
| 权重 $W$ | 切成 $W_0..W_3$ | **不会** | 一直只存在各卡本地 `params[layer]` |
| 激活（matmul 结果） | 每人先算出一截 | **会**（`all_gather` + `cat`） | 拼成完整 `x`，给下一层用 |

所以：

- 「放不下」指的是 **整份模型权重（+优化器）**；
- 「拼起来」指的是 **当前层的输出激活**；
- 这段 PPT 是 **拼接**，不是把四个人的结果 **加总**。

---

### 10.4 和 §9 DDP 差在哪（一句话）

| | §9 数据并行 | §10 这段参数切开 |
|--|--|--|
| 每卡权重 | 完整一份 | 只有 $1/4$ |
| 每卡输入 batch | 通常不同 | 通常相同 |
| 通信的是什么 | `param.grad` | 局部激活 `x` |
| 操作为什么 | `all_reduce` 让梯度一致 | `all_gather` 让激活拼齐 |

---

### 10.5 反向（只要建立对应关系）

Forward：各算一截激活 → **all_gather** → 拼成完整激活。  
Backward：完整上游梯度 → **reduce-scatter** → 每人只拿 **自己那一列权重** 对应的梯度，更新本地 $W_r$。

权重始终切开；通信的始终是「跟这一层计算有关的激活/梯度片段」，不是把大 $W$ 搬回单卡。

---

## 11. Column parallelism vs Row parallelism

是的：课上这页就叫 **column parallelism**（列并行）。  
名字来自：把 $W$ 按 **列** 切成竖条。

还有对称的一种叫 **row parallelism**（行并行）：把 $W$ 按 **行** 切成横条。  
两种都不是把矩阵切成 $2\times2$ 小块。

继续用 §10 同一组数，方便对照。

$$
x=\begin{bmatrix}1&2&3&4\end{bmatrix},\quad
W=\begin{bmatrix}
10&20&30&40\\
11&21&31&41\\
12&22&32&42\\
13&23&33&43
\end{bmatrix},\quad
xW=\begin{bmatrix}120&220&320&420\end{bmatrix}
$$

---

### 11.1 Column parallelism（课上这页）

**怎么切 $W$：** 竖着切成 4 条。

$$
W=[W_0\mid W_1\mid W_2\mid W_3]
$$

每卡持有形状 `[输入维, local_输出维]`，例子里是 `[4,1]`。

**输入：** 四卡都拿完整的 $x$（形状 `[1,4]`）。

**本地计算：**

$$
\begin{aligned}
\mathrm{rank0}&:\ xW_0=[120] \\
\mathrm{rank1}&:\ xW_1=[220] \\
\mathrm{rank2}&:\ xW_2=[320] \\
\mathrm{rank3}&:\ xW_3=[420]
\end{aligned}
$$

每人输出都是「完整结果的一截」，形状 `[1,1]`。

**怎么合并：** `all_gather` + `torch.cat(..., dim=1)` → **左右拼接**。

$$
[120]\mid[220]\mid[320]\mid[420]
=
[120,\ 220,\ 320,\ 420]
$$

**一句话：** 列切 → 各算输出的不同特征维 → **拼**。

---

### 11.2 Row parallelism（对称的另一种）

**怎么切 $W$：** 横着切成 4 条。

$$
W=\begin{bmatrix}W_0\\W_1\\W_2\\W_3\end{bmatrix}
$$

每卡持有形状 `[local_输入维, 输出维]`，例子里是 `[1,4]`：

$$
\begin{aligned}
W_0&=\begin{bmatrix}10&20&30&40\end{bmatrix} \\
W_1&=\begin{bmatrix}11&21&31&41\end{bmatrix} \\
W_2&=\begin{bmatrix}12&22&32&42\end{bmatrix} \\
W_3&=\begin{bmatrix}13&23&33&43\end{bmatrix}
\end{aligned}
$$

**输入也要按同样方式切开**（每人只拿对应那一段输入维）：

$$
x=[x_0\mid x_1\mid x_2\mid x_3]
=
[1]\mid[2]\mid[3]\mid[4]
$$

**本地计算：** 每人算的是「对完整输出的一份贡献」，形状已经是 `[1,4]`：

$$
\begin{aligned}
\mathrm{rank0}&:\ x_0W_0=1\cdot[10,20,30,40]=[10,\ 20,\ 30,\ 40] \\
\mathrm{rank1}&:\ x_1W_1=2\cdot[11,21,31,41]=[22,\ 42,\ 62,\ 82] \\
\mathrm{rank2}&:\ x_2W_2=3\cdot[12,22,32,42]=[36,\ 66,\ 96,\ 126] \\
\mathrm{rank3}&:\ x_3W_3=4\cdot[13,23,33,43]=[52,\ 92,\ 132,\ 172]
\end{aligned}
$$

数学上：

$$
xW = x_0W_0 + x_1W_1 + x_2W_2 + x_3W_3
$$

**怎么合并：** `all_reduce`（SUM）→ **按位置加总**（不是 `cat`）。

$$
\begin{aligned}
&[10,20,30,40] \\
+\ &[22,42,62,82] \\
+\ &[36,66,96,126] \\
+\ &[52,92,132,172] \\
=\ &[120,\ 220,\ 320,\ 420]
\end{aligned}
$$

**一句话：** 行切 → 各算对同一输出的部分和 → **加**。

---

### 11.3 对照表

| | Column parallelism | Row parallelism |
|--|--|--|
| $W$ 怎么切 | 按 **列** 竖切 `$[W_0\|W_1\|W_2\|W_3]$` | 按 **行** 横切 `堆叠 $W_0..W_3$` |
| 每卡 $W$ 形状（本例） | `[4,1]` | `[1,4]` |
| 输入 $x$ | 通常 **完整** 一份 | 通常 **切开**，每人一段 |
| 本地输出含义 | 完整结果的 **一段特征** | 完整结果的 **一份加数** |
| Forward 合并 | `all_gather` + `cat`（拼） | `all_reduce` SUM（加） |
| 课上 PPT | 就是这个 | 对称的另一种 |

---

### 11.4 工业上常成对出现

Megatron 一类实现里，常见模式是：

1. 某一层用 **column parallel**（输出先保持切开，或 gather）  
2. 下一层用 **row parallel**（切开的激活直接对上切开的行条带，最后 `all_reduce`）

这样两层之间有时可以 **少一次** 把激活拼成完整向量，通信更省。  
课上先只展示 column 这一侧，把 `all_gather` 钉死即可。

记住口诀：

- **Column = 竖切 $W$ = 输出分段 = 拼**  
- **Row = 横切 $W$ = 输出同形部分和 = 加**
