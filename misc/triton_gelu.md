# CUDA vs Triton：对照笔记

前半：`y=2x+1` 和 GELU —— 看清写法差在哪（这两道题上，差距主要是语法层）。  
后半：**tiled matmul** —— Triton 真正省下的是 shared memory / 同步 / 调参，那才是硬价值。

公式约定：行内 `$...$`，独立成行 `$$...$$`。  
`tl` = **`triton.language`**（`import triton.language as tl`）。

---

## 1. 小例子：`y = 2x + 1`

### 1.1 数字

$$
N = 10,\quad \mathrm{BLOCK} = 4,\quad \mathrm{grid} = \lceil 10/4\rceil = 3
$$

$$
x = [0,1,2,3,4,5,6,7,8,9]
\quad\Rightarrow\quad
y = [1,3,5,7,9,11,13,15,17,19]
$$

block / pid 分工：

$$
\begin{aligned}
0 &\rightarrow 0..3 \\
1 &\rightarrow 4..7 \\
2 &\rightarrow 8..9\quad(\text{下标 }10,11\text{ 用边界判断挡住})
\end{aligned}
$$

### 1.2 Kernel 左右对照

```text
CUDA（按 thread 写）                              Triton（按 CTA / 一段向量写）
──────────────────────────────────────────────    ──────────────────────────────────────────────
__global__ void scale(                            @triton.jit
    const float* x, float* y, int n) {            def scale(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
                                                    # tl = triton.language
                                                    # tl.constexpr: 编译期常量（这里=4）

  // 我是几号 thread、落在几号 block               # 我是几号 CTA（≈ CUDA 的 blockIdx.x）
  int i = blockIdx.x * blockDim.x                 pid = tl.program_id(axis=0)
            + threadIdx.x                         # axis=0：一维 grid 的第 0 轴

  // 例 blockIdx=1, threadIdx=2 → i=6             # 例 pid=1 → 这一段下标 [4,5,6,7]
                                                  # tl.arange(0, BLOCK_SIZE) → [0,1,2,3]
                                                  # 再加 start，得到全局 offsets
                                                  offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

  if (i < n) {                                    mask = offsets < n
                                                    # mask 长度=BLOCK_SIZE；True 才碰 HBM

      float xi = x[i];            // 读 1 个数     x = tl.load(x_ptr + offsets, mask=mask)
                                                    # 一次 load 整段向量（最多 BLOCK_SIZE 个）

      float yi = 2.f * xi + 1.f;  // 算 1 个数     y = 2.0 * x + 1.0
                                                    # 向量运算：一段一起算

      y[i] = yi;                  // 写 1 个数     tl.store(y_ptr + offsets, y, mask=mask)
  }                                                 # 一次 store 整段
}
```

盯住 **block 1 / pid=1**（下标 $4..7$）：

```text
CUDA：4 个 thread 各干各的                     Triton：1 个 CTA 干一整段
  thread0: i=4 → y[4]=9                          offsets=[4,5,6,7]
  thread1: i=5 → y[5]=11                         x=[4,5,6,7]
  thread2: i=6 → y[6]=13                         y=[9,11,13,15]   ← 一行算出
  thread3: i=7 → y[7]=15                         store 四个位置
```

### 1.3 Launch 左右对照

```text
CUDA                                              Triton
──────────────────────────────────────────────    ──────────────────────────────────────────────
scale<<<3, 4>>>(x, y, 10);                        BLOCK_SIZE = 4
#        ↑  ↑                                     grid = (triton.cdiv(10, BLOCK_SIZE),)  # (3,)
#     grid blockDim                               # cdiv = ceil 除法：⌈10/4⌉=3
#     你同时指定：                                 scale[grid](x, y, 10, BLOCK_SIZE=4)
#       多少个 block、每 block 多少 thread         # 主要指定 grid；BLOCK_SIZE 描述
                                                  # 「这一 CTA 处理多宽的向量」
```

---

## 2. 大一点：GELU（壳子相同，operate 更长）

近似式：

$$
\mathrm{GELU}(x)
\approx
0.5\, x\,
\bigl(1 + \tanh\bigl(\sqrt{2/\pi}\,(x + 0.044715\, x^{3})\bigr)\bigr)
$$

$$
\sqrt{2/\pi}\approx 0.79788456,\qquad
\tanh(a)=\frac{e^{2a}-1}{e^{2a}+1}
$$

负载仍用：

$$
N=1000,\quad \mathrm{BLOCK\_SIZE}=256,\quad \mathrm{grid}=4
$$

### 2.1 Kernel 左右对照

```text
CUDA（按 thread 写）                              Triton（按 CTA 写）
──────────────────────────────────────────────    ──────────────────────────────────────────────
__global__ void gelu(                             @triton.jit
    const float* x, float* y, int n) {            def gelu(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
                                                    # tl = triton.language

  int i = blockIdx.x * blockDim.x                 pid = tl.program_id(axis=0)
            + threadIdx.x                         # 第几个 CTA

                                                  start = pid * BLOCK_SIZE
                                                  # tl.arange：构造 [0,1,...,BLOCK_SIZE)
                                                  offsets = start + tl.arange(0, BLOCK_SIZE)

  if (i < n) {                                    mask = offsets < n

      float xi = x[i];                            x = tl.load(x_ptr + offsets, mask=mask)
                                                    # 一段 x，长度最多 BLOCK_SIZE

      // 下面全程标量                              // 下面全程向量（对 x 每个元素同步做）
      float a = 0.79788456f                       a = 0.79788456 * (x + 0.044715 * x * x * x)
                * (xi + 0.044715f * xi*xi*xi);

      float e = expf(2.f * a);                    exp = tl.exp(2.0 * a)
                                                    # tl.exp：逐元素指数

      float t = (e - 1.f) / (e + 1.f);            tanh = (exp - 1.0) / (exp + 1.0)

      float yi = 0.5f * xi * (1.f + t);           y = 0.5 * x * (1.0 + tanh)

      y[i] = yi;                                  tl.store(y_ptr + offsets, y, mask=mask)
  }
}
```

GELU 相对 `2x+1`：左右结构一样，只是中间「算」那几行变长。  
差别仍然是：**左边一次一个 `i`，右边一次一段 `offsets`。**

### 2.2 Launch 左右对照

```text
CUDA                                              Triton
──────────────────────────────────────────────    ──────────────────────────────────────────────
int block = 256;                                  BLOCK_SIZE = 256
int grid  = (1000 + block - 1) / block;          grid = (triton.cdiv(1000, BLOCK_SIZE),)  # (4,)

gelu<<<grid, block>>>(x, y, 1000);                gelu[grid](x, y, 1000, BLOCK_SIZE=256)
```

### 2.3 尾巴 CTA（pid=3）在干什么

$$
\mathrm{pid}=3 \Rightarrow \mathrm{offsets}=[768..1023],\quad
\mathrm{mask}:\ 768..999=\mathrm{True},\ 1000..1023=\mathrm{False}
$$

```text
CUDA                                              Triton
──────────────────────────────────────────────    ──────────────────────────────────────────────
某 thread 的 i=900  → if 进得去，算 GELU         offsets 里 900 对应 lane：mask True，参与 load/算/store
某 thread 的 i=1010 → if (i<n) 为假，该 thread 返回  offsets 里 1010 对应 lane：mask False，该 lane 闲置
```

---

## 3. 先说实话：前面两道题差在哪

`2x+1` 和 GELU 都是 **逐元素**：每个输出只依赖同一个位置的输入。  
这种活，CUDA 写 `i`、Triton 写 `offsets`，**算力模型几乎一样**；Triton 看起来干净，多半是少打分号、少写 `threadIdx`。  
若你习惯 C，完全可以觉得 CUDA 更顺手——这很正常，**这两道题撑不起 Triton 存在的理由**。

Triton 的硬价值出现在：**一个 CTA 要协作、要反复搬 HBM→片上、还要试很多种 tile 尺寸** 的时候。下面换一道这种题。

---

## 4. 硬价值例子：一个 CTA 算 $C$ 的一块 tile（matmul）

### 4.1 题面（数字很小，只为看清责任）

$$
C = AB,\quad A\in\mathbb{R}^{16\times 32},\ B\in\mathbb{R}^{32\times 16},\ C\in\mathbb{R}^{16\times 16}
$$

约定：**整个 $C$ 由 1 个 CTA 算完**（练习尺寸；真模型会开很多 CTA 铺满大 $C$）。

每个 CTA 仍按 $K$ 维切段，每次搬一块进片上再乘加：

$$
\mathrm{BLOCK\_K}=16
\quad\Rightarrow\quad
K\text{ 方向两段：}[0,16),\ [16,32)
$$

数据流（每个 CTA）：

$$
\mathrm{HBM}(A,B)
\;\rightarrow\;
\text{片上 tile}
\;\rightarrow\;
\text{寄存器累加 }C\text{ 的 }16\times 16
\;\rightarrow\;
\mathrm{HBM}(C)
$$

### 4.2 左右对照：你手写什么 vs 编译器接什么

下面 CUDA 侧是「手写 tiled matmul 时必须露面的零件」（shared + 同步 + 下标）；  
Triton 侧是同一算法的常见写法。左右仍在**同一代码块**里。

```text
CUDA（CTA 内协作，零件都得自己写）              Triton（同一算法，写 CTA 级数据流）
──────────────────────────────────────────────    ──────────────────────────────────────────────
__global__ void matmul_tile(                      @triton.jit
    const float* A, const float* B, float* C) {   def matmul_tile(A_ptr, B_ptr, C_ptr,
                                                    M, N, K,
                                                    stride_am, stride_ak,
                                                    stride_bk, stride_bn,
                                                    stride_cm, stride_cn,
                                                    BLOCK_M: tl.constexpr,  # 16
                                                    BLOCK_N: tl.constexpr,  # 16
                                                    BLOCK_K: tl.constexpr): # 16
                                                    # tl = triton.language

  // 片上暂存：你声明、你决定布局                # 下面 tl.load 的 tile，编译器安排
  __shared__ float As[16][16];                    # 进 shared / 寄存器 / 如何向量化
  __shared__ float Bs[16][16];

  int tx = threadIdx.x;  // 0..15 当列
  int ty = threadIdx.y;  // 0..15 当行
  float acc = 0.f;        // 每 thread 一个 C 元素  # 整块 C tile 的累加器（CTA 视角）
                                                  acc = tl.zeros((BLOCK_M, BLOCK_N),
                                                                 dtype=tl.float32)

  for (int k0 = 0; k0 < 32; k0 += 16) {           for k0 in range(0, K, BLOCK_K):
                                                    # 这一 CTA 要的 A、B 下标（二维）
                                                    offs_am = tl.arange(0, BLOCK_M)[:, None]
                                                    offs_ak = k0 + tl.arange(0, BLOCK_K)[None, :]
                                                    offs_bk = k0 + tl.arange(0, BLOCK_K)[:, None]
                                                    offs_bn = tl.arange(0, BLOCK_N)[None, :]

      // 协作把 A 的 16x16 搬进 As：谁搬哪要算清  a = tl.load(A_ptr
      As[ty][tx] = A[ty * 32 + (k0 + tx)];          + offs_am * stride_am
                                                    + offs_ak * stride_ak)
      // 同理搬 B → Bs                              # 一次 load 一整块 A tile
      Bs[ty][tx] = B[(k0 + ty) * 16 + tx];        b = tl.load(B_ptr
                                                      + offs_bk * stride_bk
                                                      + offs_bn * stride_bn)

      __syncthreads();  // 等全员搬完才能乘         # 同步点：编译器插；源码里通常看不到

      // 用片上数据做内积（仍是 per-thread）       # tl.dot：块乘块；编译器映射到
      for (int kk = 0; kk < 16; ++kk)               # Tensor Core / 共享内存流水线等
          acc += As[ty][kk] * Bs[kk][tx];         acc += tl.dot(a, b)

      __syncthreads();  // 下一段 k 再覆盖 As/Bs    # 下一段循环直接再 load
  }

  C[ty * 16 + tx] = acc;                          offs_cm = tl.arange(0, BLOCK_M)[:, None]
                                                  offs_cn = tl.arange(0, BLOCK_N)[None, :]
                                                  tl.store(C_ptr
                                                    + offs_cm * stride_cm
                                                    + offs_cn * stride_cn,
                                                    acc)
}
# launch 示意：<<<1, dim3(16,16)>>>               # launch 示意：grid=(1,)；BLOCK_* 当 constexpr
```

读这一页时盯三件事：

1. **Shared memory**：CUDA 里 `As`/`Bs` + 谁写哪个格子，源码里明文出现；Triton 里你写 `tl.load` 一块 tile，**片上怎么摆**交给编译器。  
2. **`__syncthreads__`**：CUDA 里两段循环各一次；Triton 源码按「load → dot → 再 load」写，**同步点由编译器插入**。  
3. **`tl.dot`**：你写的是「这两块 tile 乘起来加进累加器」；底下怎么用 Tensor Core、怎么切 warp，是编译器的活。

逐元素 GELU **用不到**这三层；matmul / attention / softmax 归约 **全是**这三层。Triton 的价值主要在这儿：  
**把 CTA 级算法写成数据流，把「丑陋但必须正确」的片上编排交给编译器。**

### 4.3 第二块硬价值：Autotune（同一算法试很多套参数）

真要跑得快，还得试 `BLOCK_M/N/K`、`num_warps` 等。CUDA 里通常是：改宏 → 编译 → 测 → 再改。  
Triton 可以把候选配置写在装饰器里，**按 key 自动试跑并缓存赢家**：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16}, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],   # 这些形状变了再重新搜
)
@triton.jit
def matmul_kernel(...):
    ...
```

价值是：**算法写一份，tile/warp 配置当数据搜**；这和「少打两行斜杠」是完全不同的产能。

### 4.4 和 PyTorch 的衔接（顺便）

Triton kernel 直接吃 `torch.Tensor` 的 data pointer，改完立刻在训练里换自定义 kernel（融合、FlashAttention 类）。  
价值是：**研究/生产里改 GPU 核的迭代变短**；仍和「语法好不好看」分开。

---

## 5. 对照总表

| 场景 | CUDA | Triton | 谁更「有必要」 |
|------|------|--------|----------------|
| `y=2x+1` / GELU | 写 `i`，标量算 | 写 `offsets`，向量算 | 主要是口味与篇幅 |
| tiled matmul | `__shared__` + `__syncthreads__` + 手工下标 | `tl.load` tile + `tl.dot` | Triton 省的是片上编排 |
| 调 BLOCK / warps | 人肉改测 | `@triton.autotune` | Triton 把搜索产品化 |
| 接训练代码 | 另写绑定/扩展 | 同进程调 Python | 迭代更快 |

一句话：  
**逐元素例子证明不了 Triton；CTA 协作 + 自动搜参才证明。**

---

## 6. 符号速查

$$
\begin{aligned}
\texttt{tl} &\equiv \texttt{triton.language} \\
\texttt{tl.program\_id}(0) &\equiv \text{CUDA } \texttt{blockIdx.x} \\
\texttt{tl.arange}(0,B) &\equiv [0,1,\ldots,B-1] \\
\texttt{tl.load}/\texttt{tl.store} &\equiv \text{按 offsets（+mask）读写 HBM} \\
\texttt{tl.dot} &\equiv \text{tile}\times\text{tile 累加（编译器管片上实现）} \\
\texttt{tl.zeros} &\equiv \text{CTA 视角的累加器 tile} \\
\texttt{tl.exp} &\equiv \text{逐元素 } e^{(\cdot)} \\
\texttt{tl.constexpr} &\equiv \text{编译期常量} \\
\texttt{triton.cdiv}(n,B) &\equiv \lceil n/B\rceil \\
\texttt{@triton.autotune} &\equiv \text{多组 Config 试跑并缓存最优}
\end{aligned}
$$
