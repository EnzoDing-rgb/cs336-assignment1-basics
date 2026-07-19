# MA：Data Loading（从长 token 序列里切训练 batch）

对应 CS336 Assignment 1：`Problem (data_loading)`。

公式约定：行内 `$...$`，独立成行 `$$...$$`。

本文目标：讲清楚 data loader 在整条训练流水线里干什么、输入输出分别是什么、为什么 $y$ 比 $x$ 错一位、随机起点怎么采、大数据如何用 mmap 假装在内存、最后怎么落到实现。不跳步。带玩具例子。

---

## 0. 先把整条训练链路摆出来

到目前为止，作业里已经有了：

| 部件 | 作用 |
|------|------|
| Tokenizer / 编码后的数据 | 文本 → 一长串 token ID |
| 模型（Transformer LM） | 输入一段 token，预测下一个 token |
| 优化器（AdamW）+ LR schedule + grad clip | 用梯度更新参数 |

还缺一块：**每次训练 step，从那一长串 token 里拿出一小捆 `(输入, 目标)` 送给模型。**

这块就叫 **data loader**（本题只要写「采样一个 batch」的函数，不必写成 PyTorch 的 `DataLoader` 类）。

一次训练 step 的数据流：

```text
磁盘上的 token 序列 x = (x_0, x_1, …, x_{n-1})
        │
        ▼
   get_batch(...)          ← 本题
        │
        ├── inputs  : 形状 (B, m) 的 token ID
        └── targets : 形状 (B, m) 的「下一个 token」ID
        │
        ▼
   模型前向 → logits → cross-entropy(logits, targets)
        │
        ▼
   backward →（可选）clip → optimizer.step
```

---

## 1. 磁盘上的数据长什么样？

讲义说：tokenized data 是 **一条** 长序列

$$
x = (x_1,\ldots,x_n)
$$

实现里更常见是 **0 下标** 的一维 numpy 整数数组：

$$
\texttt{dataset[0]},\ \texttt{dataset[1]},\ \ldots,\ \texttt{dataset[n-1]}
$$

每个元素是一个 token ID（词表里的整数编号）。

即使原文是很多篇网页 / 很多个文件，常见做法也是：

1. 每篇分别 tokenize  
2. 中间插入分隔符（如 `<|endoftext|>`）  
3. **全部拼成一条超长序列** 存成 `.npy` / memmap

本题的函数 **假定这件事已经做完**：输入就是这一条一维数组。

---

## 2. 语言模型要学的监督信号是什么？

给定前面若干 token，预测 **下一个** token。

若模型一次吃长度为 $m$ 的输入窗口：

$$
\text{inputs} = (x_i,\ x_{i+1},\ \ldots,\ x_{i+m-1})
$$

对应的目标（每个位置预测「再往后一个」）是：

$$
\text{targets} = (x_{i+1},\ x_{i+2},\ \ldots,\ x_{i+m})
$$

一一对应关系：

| 输入位置 | 模型看到的 token | 要预测的目标 |
|----------|------------------|--------------|
| 第 0 格 | $x_i$ | $x_{i+1}$ |
| 第 1 格 | $x_{i+1}$ | $x_{i+2}$ |
| … | … | … |
| 第 $m-1$ 格 | $x_{i+m-1}$ | $x_{i+m}$ |

所以：**targets 就是 inputs 整体向右错开 1 格**（再多取序列里后面那一个 token）。

这就是 adapter 文档里说的 language modeling labels。

---

## 3. Batch 是什么？符号先钉死

| 符号 | 本题名字 | 含义 |
|------|----------|------|
| $n$ | `len(dataset)` | 整条 token 序列长度 |
| $B$ | `batch_size` | 一次采多少条训练序列 |
| $m$ | `context_length` | 每条序列的长度（上下文窗口） |
| $i$ | start index | 一条序列在长数组里的起始下标 |

函数要返回两个张量，都放在指定 `device` 上：

$$
\texttt{inputs},\ \texttt{targets}
\in \mathbb{Z}^{B \times m}
$$

- 第 $b$ 行：第 $b$ 条样本  
- 第 $j$ 列：该样本里第 $j$ 个位置的 token ID  

讲义例子（$B=1$，$m=3$）：一批可以是

$$
\bigl([x_2,x_3,x_4],\ [x_3,x_4,x_5]\bigr)
$$

（讲义用 1 下标；$x_2$ 对应实现里下标 1 附近，只看「错开一位」即可。）

---

## 4. 为什么这样切数据？讲义给的三点理由

1. **几乎处处都能采**  
   任意合法起点 $i$ 都能切出一段长度为 $m$ 的训练窗，采样规则极简单。

2. **长度统一，不用 padding**  
   每条都是 $m$，batch 里不会出现「有的短有的长」。硬件更好吃满，也方便把 $B$ 做大。

3. **不必整库装进内存才能训**  
   只要能按整数下标读 `dataset[i : i+m+1]`（numpy memmap 也能），就能采样。超大数据集也能训。

下一节把第 3 点展开：文件很大时，怎么在「假装整库在内存」的前提下采样。

---

## 5. 数据太大塞不进内存时：mmap / `np.memmap`

### 5.1 问题

token 序列可能有几十 GB。若每次训练都

```text
dataset = np.load("tokens.npy")   # 一口气读进 RAM
```

机器内存可能直接爆掉。但 `get_batch` 其实每次只摸很小一段：`dataset[i : i+m+1]`。  
我们需要的是：**按需读盘上的那几页，而不是整文件进 RAM。**

### 5.2 `mmap` 在干什么（第一性）

Unix 有个系统调用叫 **`mmap`（memory map）**：

- 把磁盘上的一个文件 **映射** 到进程的虚拟地址空间  
- 你像访问普通数组那样用下标去读  
- 真正的文件内容：**碰到那个地址时，操作系统才懒加载（lazy load）对应页进物理内存**

所以你可以「假装」整份数据都在内存里，实际常驻的只是最近访问过的那几页。

### 5.3 Numpy 怎么用

Numpy 把这件事包成了 **`np.memmap`**，返回一个 **看起来像 `ndarray`、按需读盘** 的对象。

两种常见打开方式（取决于你当初怎么存的）：

**方式 A：直接 memmap 一个二进制文件**

```python
dataset = np.memmap(
    "tokens.bin",   # 或你的路径
    dtype=np.uint16,  # 必须和当初写入时一致
    mode="r",         # 只读
    shape=(n,),       # 一维 token 序列长度
)
```

**方式 B：当初用 `np.save` 存成 `.npy`，加载时开 mmap**

```python
dataset = np.load("tokens.npy", mmap_mode="r")
```

`mmap_mode='r'` 表示只读映射，不会把整阵一次性拷进 RAM。

### 5.4 训练采样时要注意什么

讲义要求：真正开训、从 dataset 里采样时，**用 memory-mapped 模式打开**（`np.memmap` 或 `np.load(..., mmap_mode='r')`）。

另外两件很容易踩坑的事：

1. **`dtype` 必须和落盘时一致**  
   若保存是 `uint16` / `int32` / `uint32`，加载时写错 dtype，读出来的整数全是错的（token ID 会花掉）。

2. **打开后先做一次 sanity check**  
   例如：
   - `dataset.shape`、`dataset.dtype` 对不对  
   - `dataset.min()` / `dataset.max()` 是否落在词表范围内（不应出现 $\ge V$ 的 ID，除非你另有特殊约定）  
   - 抽几段 `dataset[0:64]` 用 tokenizer decode 一眼看看像不像人话  

`get_batch` 函数本身通常 **不负责** 打开文件：它只接收一个已经是「1D 整数数组样」的 `dataset`（真 `ndarray` 或 `memmap` 都行）。  
**谁在训练脚本里构造 `dataset`，谁就要用 mmap 模式打开大文件。**

### 5.5 和本题 `get_batch` 的关系

| | 小数组（单元测试） | 大语料（真训练） |
|--|--|--|
| `dataset` 类型 | 普通 `np.ndarray` | `np.memmap` / `mmap_mode='r'` 加载的数组 |
| `get_batch` 里怎么切 | `dataset[i:i+m+1]` | **写法完全一样** |
| 差别在哪 | 数据已在 RAM | 下标访问时才读盘 |

所以：mmap 是 **数据怎么打开** 的事；切片逻辑仍是 §6–§8 那套，不用为 memmap 另写一套采样。

---

## 6. 合法起点 $i$ 的范围（最容易写错）

要同时切出：

$$
\text{inputs} = \texttt{dataset}[i : i+m]
\qquad
\text{targets} = \texttt{dataset}[i+1 : i+1+m]
$$

targets 的最后一个下标是 $i+m$。  
数组合法下标最大是 $n-1$，所以必须：

$$
i + m \le n - 1
\quad\Leftrightarrow\quad
i \le n - m - 1
$$

又 $i \ge 0$，因此：

$$
i \in \{0,\,1,\,\ldots,\,n-m-1\}
$$

合法起点一共有：

$$
n - m
$$

个。

### 6.1 用小数字核对

设

$$
\texttt{dataset} = [0,1,2,3,4,5,6,7,8,9],
\quad n=10,\ m=3
$$

合法 $i$：$0,1,2,\ldots,10-3-1=6$，共 $7=10-3$ 个。

| $i$ | inputs `dataset[i:i+3]` | targets `dataset[i+1:i+4]` |
|-----|-------------------------|----------------------------|
| 0 | `[0,1,2]` | `[1,2,3]` |
| 1 | `[1,2,3]` | `[2,3,4]` |
| 6 | `[6,7,8]` | `[7,8,9]` |
| 7（非法） | `[7,8,9]` | `[8,9,10]` ← 越界 |

测试里对 `dataset = np.arange(100)`、`context_length = 7` 断言：

- 可能起点个数 $= 100-7 = 93$
- 起点最大值 $= 92 = 93-1$
- 起点最小值 $= 0$

与上面公式一致。

---

## 7. 玩具例子：手搓一个 $B=2,\ m=4$ 的 batch

仍用

$$
\texttt{dataset} = [10,\,11,\,12,\,13,\,14,\,15,\,16,\,17,\,18,\,19]
$$

（数字写成 10 起，避免和「下标」混淆。）

$n=10$，$m=4$，合法起点 $i\in\{0,1,\ldots,5\}$。

假设随机抽到两个起点：$i_0=2$，$i_1=5$。

**第 0 条：**

$$
\begin{aligned}
\text{inputs}[0]
&=
[12,\,13,\,14,\,15]
\\
\text{targets}[0]
&=
[13,\,14,\,15,\,16]
\end{aligned}
$$

**第 1 条：**

$$
\begin{aligned}
\text{inputs}[1]
&=
[15,\,16,\,17,\,18]
\\
\text{targets}[1]
&=
[16,\,17,\,18,\,19]
\end{aligned}
$$

拼成 batch：

$$
\texttt{inputs}
=
\begin{bmatrix}
12 & 13 & 14 & 15 \\
15 & 16 & 17 & 18
\end{bmatrix}
,\quad
\texttt{targets}
=
\begin{bmatrix}
13 & 14 & 15 & 16 \\
16 & 17 & 18 & 19
\end{bmatrix}
$$

形状都是 $(2,\,4) = (B,\,m)$。

若 `dataset` 恰好是连续整数（测试用的 `np.arange`），则有更狠的检查：

$$
\texttt{targets} = \texttt{inputs} + 1
$$

（逐元素加一。）真实语料里 token ID 不连续，不能靠 $+1$，但「错开一位」的关系不变。

---

## 8. 函数接口（对着作业 / adapter 写）

```text
get_batch(dataset, batch_size, context_length, device)
    → (inputs, targets)
```

| 参数 | 类型 | 含义 |
|------|------|------|
| `dataset` | 1D `np.ndarray`（整数）或 memmap | 整条 token 序列 |
| `batch_size` | `int` | $B$ |
| `context_length` | `int` | $m$ |
| `device` | `str` | 如 `'cpu'`、`'cuda:0'` |

返回：

| 返回值 | 形状 | dtype 期望 | 设备 |
|--------|------|------------|------|
| `inputs` | $(B,\,m)$ | 整数（`torch.long`） | `device` |
| `targets` | $(B,\,m)$ | 同上 | `device` |

适配器：`adapters.run_get_batch`。  
测试：`uv run pytest -k test_get_batch`。

---

## 9. 实现步骤（无逻辑跳步）

### 9.1 算合法起点个数

$$
n = \texttt{len(dataset)},
\quad
m = \texttt{context\_length}
$$

合法起点下标集合大小：

$$
n - m
$$

（或写成 `len(dataset) - context_length`。）  
采样时应用 `torch.randint` / `np.random.randint`，上界是 **开区间** 的 $n-m$（即最大取到 $n-m-1$）。

### 9.2 采 $B$ 个起点

独立、均匀地采 $B$ 个整数：

$$
i_0,\ i_1,\ \ldots,\ i_{B-1}
\sim
\mathrm{Uniform}\{0,1,\ldots,n-m-1\}
$$

同一 batch 里允许起点重复（测试按「近似均匀」统计，不要求无放回）。

### 9.3 按起点切出 inputs / targets

对每个 $b=0,\ldots,B-1$：

$$
\begin{aligned}
\texttt{inputs}[b]
&\leftarrow
\texttt{dataset}[i_b : i_b + m]
\\
\texttt{targets}[b]
&\leftarrow
\texttt{dataset}[i_b + 1 : i_b + 1 + m]
\end{aligned}
$$

注意 targets 要比 inputs **多看一个 token**（窗口右端再伸一格），所以从长数组里实际摸到的是长度 $m+1$ 的片段，再拆成两段。

一种写法：

```text
for each start i in the B starts:
    window = dataset[i : i + m + 1]   # 长度 m+1
    inputs_row  = window[0 : m]
    targets_row = window[1 : m + 1]
```

### 9.4 变成 PyTorch 张量并放到 device 上

- 用 `torch.tensor(..., dtype=torch.long, device=device)`  
  或先 `torch.from_numpy` 再 `.to(device)` / `.long()`  
- 测试会检查：非法设备（如 `'cuda:99'`）应触发错误——说明 **必须真的把张量放到传入的 `device` 上**，不能写死 `'cpu'`。

### 9.5 函数返回什么？

返回 **一对张量** `(inputs, targets)`。  
这和 gradient clipping 不同：

| | `clip_gradients` | `get_batch` |
|--|--|--|
| 本质 | 原地改已有 `.grad` | **新采样、新构造** 两个张量 |
| 返回值 | `None` | `(inputs, targets)` |

---

## 10. 和前面符号的对齐（避免和 FLOPs 文档打架）

训练记账文档里常用：

| 符号 | 含义 |
|------|------|
| $B$ | batch size |
| $L$ | context length（本文的 $m$） |

本题讲义用 $m$ 表示 context length；实现参数名是 `context_length`。  
见到 $L$ 或 $m$，都指「每条序列多长」，不是层数 $N$。

---

## 11. 测试在查什么（读完再写代码）

`test_get_batch` 大致查四件事：

1. **形状**：`(batch_size, context_length)` 两个都对  
2. **错位关系**：在 `dataset = arange(...)` 时，`targets == inputs + 1`  
3. **起点合法且近似均匀**：起点落在 $\{0,\ldots,n-m-1\}$，多次采样后各起点出现次数别离谱  
4. **device 真的用了**：`device='cuda:99'` 应报错  

---

## 12. 一句话收束

- 数据是一条长 token 数组；loader 每次随机切 $B$ 段长度为 $m$ 的窗。  
- `inputs = dataset[i:i+m]`，`targets = dataset[i+1:i+1+m]`（下一 token 监督）。  
- 合法起点 $i\in\{0,\ldots,n-m-1\}$。  
- 返回两个 `(B, m)` 的 long tensor，放到指定 `device`。  
- 真训练时大数据用 `np.memmap` / `np.load(..., mmap_mode='r')` 打开，dtype 对齐，并做一次词表范围检查；`get_batch` 切片写法不变。

下一步：把 §9 翻译成函数，接上 `run_get_batch`，跑 `pytest -k test_get_batch`。
