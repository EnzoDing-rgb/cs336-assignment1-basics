# GPU Occupancy 与编程模型笔记

内容顺序：寄存器与 occupancy → bank conflict → coalescing → 绿块图 → 用 matmul 讲清 Grid/Block/Thread（§12）。

公式约定：行内 `$...$`，独立成行 `$$...$$`。  
符号以本节术语表为准；后文一律沿用，不再改含义。

---

## 0. 术语表 / 符号表（先把字母认全）

后面会蹦出一堆 $R$、$r$、$T$、$B$、$W$。**大小写不是随便写的**：大写多半是「整块 / 总量 / 个数」，小写 $r$ 专指「每个线程吃掉的寄存器个数」。

### 0.1 英文词（硬件 / 编程概念）

| 词 | 中文 | 人话 |
|----|------|------|
| SM | Streaming Multiprocessor | GPU 里的一条产线；真正算数的单元住在这里 |
| Thread | 线程 | 最细的一个工人；每个工人跑同一段 kernel |
| Warp | 线程束 | **固定 32 个** thread 绑成一捆，一起被调度 |
| Grid | 网格 | 一次 kernel launch 的全部 block 及其排布；§12 用 matmul 说明 |
| Block | 线程块 | 一组一起调度、可共享 shared memory 的 threads |
| Register | 寄存器 | 每个 thread 私有的最快存储 |
| Shared memory | 共享内存 | 同一 block 内 threads 可共享的片上内存（本笔记后半 bank conflict 才用到） |
| Occupancy | 占用率 | 实际同时活着的 warp 数 ÷ 硬件允许的最大 warp 数 |
| Kernel | 核函数 | 你丢给 GPU 跑的那段函数 |
| Bank conflict | 存储体冲突 | 一个 warp 里多人撞同一个 shared-memory bank 的不同地址，访问被串行化 |
| Bank | 存储体 | Shared memory 的 32 个并行窗口之一（$\mathrm{B00}\ldots\mathrm{B31}$）；详见 §9 |
| Swizzling | 搅动 / 重排布局 | 改 shared memory 里行列怎么映射到地址，用来躲 bank conflict |
| HBM | High Bandwidth Memory | 片外大显存（全局内存那一层）；慢，靠 coalescing + 高 occupancy 掩盖等待 |
| Coalescing | 内存合并 | Warp 访问 HBM 时，邻近地址被合成尽量少的 128-byte 事务；详见 §10 |
| Cache line | 缓存行 | 硬件一次倾向于搬动的内存块；课上按 **128 bytes** 讲 |
| Block occupancy | Block 占用（示意） | SM–time 图上同时堆着多少个 block；和 warp occupancy 同一类事；详见 §11 |
| Wave | 波次 | 大批 block 分批上 SM；§12.5 |

字幕里的「扭曲」= warp（字面翻译）；技术中文更常说「线程束」或直接说 warp。

### 0.2 数学符号（本文全部字母）

| 符号 | 读法 / 代码里常见写法 | 是什么 | 本例数值 |
|------|------------------------|--------|----------|
| $T$ | threads per block；`num_threads_per_block` | **一个 block 里有多少个 thread** | $128$ |
| $r$ | registers per thread；`num_registers_per_thread` | **每个 thread 用掉多少个寄存器**（小写，强调「每人一份」） | $160$ |
| $R_{\mathrm{block}}$ | registers per block；`num_registers_per_block` | **一个 block 总共要多少寄存器** $= T \cdot r$ | $20480$ |
| $R_{\max}$ | max registers；`max_registers` | **一个 SM 上寄存器总量**（硬件给的） | $65536$ |
| $B$ | blocks（同时住下的）；`num_blocks` | **一个 SM 上因资源限制能同时跑几个 block** | $3$ |
| $W$ | warps（同时活着的）；`num_warps` | **上面这些 block 折合多少个 warp** $= (B \cdot T) / 32$ | $12$ |
| $W_{\max}$ | max warps；`max_warps` | **一个 SM 最多允许同时挂多少个 warp**（硬件给的） | $64$ |
| $\mathrm{occupancy}$ | occupancy | $W / W_{\max}$ | $12/64 = 18.75\%$ |
| $32$ | warp size | 一个 warp 固定含 32 个 thread（常数，不是变量） | $32$ |
| $255$ | max registers per thread | 单线程寄存器上限（硬件硬限制） | $r \le 255$ |

---

## 1. 先建立世界观（完全零基础版）

把一块 GPU 想成一座工厂。上面术语表里的 SM / Thread / Warp / Block 就是厂房、工人、班组、班次；这里只再强调三条：

1. **调度按 warp，不是按单个 thread。** 一个 warp 永远是 32 个线程。
2. **一个 SM 上能同时住多少东西，受多种资源同时限制**：寄存器、shared memory、以及「最多能挂多少 warps / blocks」这类硬上限。
3. **谁先不够用，谁就是瓶颈。** 课上这张图专门讲「寄存器先不够」的情况。

---

## 2. 硬件给你什么？

图里给出两个硬件常数（某个架构上的例子；换代 GPU 数字会变，但思路不变）：

$$
\begin{aligned}
R_{\max} &= 65536 \\
W_{\max} &= 64
\end{aligned}
$$

其中：

- $R_{\max}$：一个 SM 上寄存器总数（图里的 `max_registers`）
- $W_{\max}$：一个 SM 上最多同时存在的 warp 数（图里的 `max_warps`）

另外还有一条「单线程寄存器上限」：

$$
r \le 255
$$

其中 $r$ 是每个 thread 用掉的寄存器个数（图里的 `num_registers_per_thread`）。超过这个上限，编译器 / 硬件就不让你这么干（会 spill 到慢内存，或根本编不过；图里用 `assert` 先卡死）。

---

## 3. 你的 kernel 带来什么？

启动配置里你（或框架）会定：

$$
\begin{aligned}
T &= 128 \\
r &= 160
\end{aligned}
$$

（含义见 §0.2：$T$ = 每 block 线程数，$r$ = 每线程寄存器数。）

直觉：

- $T$ 越大，一个 block 越大，但也更「胖」；
- $r$ 越大，每个工人带的小本子越厚，同样空间里能塞下的人越少。

---

## 4. 一步一步算：寄存器先把 block 卡住

### 4.1 一个 block 要多少寄存器？

每个 thread 要 $r$ 个，block 里有 $T$ 个 thread，所以：

$$
R_{\mathrm{block}} = T \cdot r
$$

代入数字：

$$
R_{\mathrm{block}} = 128 \times 160 = 20480
$$

这就是图上的：

$$
\texttt{num\_registers\_per\_block} = 20480
$$

### 4.2 一个 SM 上能同时塞几个这样的 block？

寄存器总量除以每个 block 的开销，再向下取整（装不下就装不下，不能装 0.几 个 block）：

$$
B = \left\lfloor \frac{R_{\max}}{R_{\mathrm{block}}} \right\rfloor
$$

代入：

$$
B = \left\lfloor \frac{65536}{20480} \right\rfloor = \left\lfloor 3.2 \right\rfloor = 3
$$

这就是图上的：

$$
\texttt{num\_blocks} = 3
$$

旁边那句 **Limited by registers** 的意思是：  
**不是**「硬件最多只允许 3 个 block」这种别的规则先撞墙，而是**寄存器账先算不过去**，所以只能同时住 3 个 block。

（如果寄存器很富余，还可能被 shared memory、或 $W_{\max}$、或「每 SM 最大 block 数」卡住；那是另一条约束线。本图只走寄存器这条线。）

### 4.3 这 3 个 block 等于多少个 warp？

一个 warp = 32 个 thread，所以：

$$
W = \frac{B \cdot T}{32}
$$

代入：

$$
W = \frac{3 \times 128}{32} = \frac{384}{32} = 12
$$

这就是图上高亮的那一行，以及字幕说的「这就相当于 12 个 warp」：

$$
\texttt{num\_warps} = 12
$$

### 4.4 Occupancy 是多少？

定义：

$$
\mathrm{occupancy} = \frac{W}{W_{\max}}
$$

代入：

$$
\mathrm{occupancy} = \frac{12}{64} = 0.1875 = 18.75\%
$$

人话：**硬件理论上允许这条 SM 同时挂 64 个 warp；因为寄存器太贵，你实际只能挂 12 个，所以占用率只有约两成。**

---

## 5. 把整条公式串成一条链

把上面四步合成一条：

$$
\begin{aligned}
R_{\mathrm{block}} &= T \cdot r \\
B &= \left\lfloor \frac{R_{\max}}{T \cdot r} \right\rfloor \\
W &= \frac{B \cdot T}{32} \\
\mathrm{occupancy} &= \frac{W}{W_{\max}}
\end{aligned}
$$

本例：

$$
\begin{aligned}
T &= 128,\quad r = 160 \\
R_{\max} &= 65536,\quad W_{\max} = 64 \\[0.5em]
R_{\mathrm{block}} &= 20480 \\
B &= 3 \\
W &= 12 \\
\mathrm{occupancy} &= \frac{12}{64} = 18.75\%
\end{aligned}
$$

---

## 6. 为什么 occupancy 值得关心？

GPU 喜欢「很多 warp 轮流干活」：

- 某个 warp 在等内存（latency）时，调度器可以切到别的 ready warp 继续算；
- **同时活着的 warp 太少**，等待时就没有人可切换，SM 容易空转。

所以：

- **高 occupancy** 往往（不总是）有利于掩盖延迟；
- **低 occupancy** 不一定绝对慢，但若你看到利用率低、且 profiler 显示 register-limited，这就是一个可疑点。

代价也很直白：想提高 occupancy，常常要**少用寄存器**（改算法、拆 kernel、让编译器少占寄存器），或改 block 大小 $T$，在「每个 block 多胖」和「能塞几个 block」之间找平衡。

---

## 7. 后文导航

- Bank conflict → §9  
- Coalescing → §10  
- Block occupancy 时间图 → §11  
- 用 matmul 讲 Grid / Block / Thread 与 CUDA launch → §12  

---

## 8. 一分钟自测

不看上文，试着填空（空白写成 `?`，避免下划线干扰渲染）：

**题 1.** 已知

$$
T = 128,\qquad r = 160
$$

求

$$
R_{\mathrm{block}} = {?}
$$

**题 2.** 已知

$$
R_{\max} = 65536
$$

以及题 1 的 $R_{\mathrm{block}}$，求

$$
B = {?}
$$

**题 3.** 求对应的

$$
W = {?}
$$

以及相对

$$
W_{\max} = 64
$$

的 occupancy

$$
\mathrm{occupancy} = {?}
$$

**答案：**

$$
R_{\mathrm{block}} = 20480,\quad
B = 3,\quad
W = 12,\quad
\mathrm{occupancy} = 18.75\%
$$

---

## 9. Bank conflicts 到底怎么回事（从零讲）

前面 occupancy 问的是：「这条 SM **同时能塞进**多少 warp？」  
Bank conflict 问的是另一件事：「这些人已经住进来了，一起读 **shared memory** 时，会不会 **排队**？」

### 9.1 Shared memory 长什么样？

Shared memory = 同一 block 里 threads 共用的一块很快的片上内存。  
硬件上，它不是一整条大马路，而是切成 **32 个并行窗口**，每个窗口叫一个 **bank**：

$$
\mathrm{B00},\; \mathrm{B01},\; \mathrm{B02},\; \ldots,\; \mathrm{B31}
$$

每个 bank 一次吐 **4 bytes**（一个 32-bit word）。  
可以把它想成：32 个柜台同时营业；每个柜台每拍只能服务 **一笔** 不同地址的请求。

### 9.2 地址怎么落到某个 bank？

把 shared memory 按 4-byte word 编号：$0,1,2,3,\ldots$  
第 $i$ 个 word 落在哪个 bank，通常就是：

$$
\mathrm{bank}(i) = i \bmod 32
$$

所以连续的 word 会轮流铺在 32 个 bank 上：

$$
\begin{aligned}
i &= 0 \rightarrow \mathrm{B00} \\
i &= 1 \rightarrow \mathrm{B01} \\
i &= 2 \rightarrow \mathrm{B02} \\
&\vdots \\
i &= 31 \rightarrow \mathrm{B31} \\
i &= 32 \rightarrow \mathrm{B00} \quad\text{（又转一圈）}
\end{aligned}
$$

一行 32 个 float（每个 4 bytes）刚好铺满一整圈 bank——这很常见。

### 9.3 什么叫 conflict？什么叫没事？

一个 warp 有 32 个 thread，它们往往在**同一拍**一起访问 shared memory。规则是：

1. **同一拍里，同一个 bank 只能服务一个「不同地址」。**
2. 若多个 thread 访问的是**完全同一个地址**，硬件可以 **broadcast**（广播一份给大家）——这不算冲突。
3. 若多个 thread 访问的是**同一个 bank、但不同地址**，这些访问必须 **串行**（一个接一个）——慢出来的那段就叫 **bank conflict**。

$n$ 个不同地址撞在同一个 bank 上，就叫 **$n$-way bank conflict**；最坏可以到 **32-way**（一个 warp 里 32 个人全撞同一个 bank 的不同格子）。

### 9.4 经典惨案：按「列」读矩阵

假设矩阵按 **行优先** 放进 shared memory，每一行刚好 32 个 word，铺满 $\mathrm{B00}\ldots\mathrm{B31}$。

- **按行读**（32 个 thread 各拿一行里相邻的一格）：每人落到不同 bank → **无冲突**，一拍搞定。
- **按列读**（32 个 thread 各拿「第 0 列」）：  
  第 0 行第 0 列、第 1 行第 0 列、第 2 行第 0 列……  
  因为每行起点都对齐到 $\mathrm{B00}$，第 0 列的 word 编号往往都是

$$
0,\; 32,\; 64,\; 96,\; \ldots
$$

于是

$$
\mathrm{bank}(0)=\mathrm{bank}(32)=\mathrm{bank}(64)=\cdots=\mathrm{B00}
$$

32 个 thread 全撞 $\mathrm{B00}$ 的不同地址 → **32-way bank conflict**，本来一拍能做完的事，变成串了大约 32 拍。

### 9.5 为什么课上说 matmul 里「经常躲不掉」？

矩阵乘 $A @ B$ 时，你常常要：

- 沿 $A$ 的某一行走（还好），同时
- 沿 $B$ 的某一列走（容易撞 bank）

两边访问模式拧在一起，shared memory 里就很容易出现「有人读行、有人读列」的冲突。所以课上说：做 matmul 时 bank conflict **often unavoidable**——不是数学必错，而是朴素布局下很常见。

### 9.6 常见解法：Swizzling（搅一搅存放位置）

思路：数据逻辑上还是矩阵，但**物理下标换个映射**，让「按列读」时也落到不同 bank。

课上提的典型招数：用行列做异或，例如把存放位置写成

$$
\mathrm{index}' = \mathrm{row} \oplus \mathrm{col}
$$

（具体公式因实现而异；核心是 **打乱 row/col → bank 的对应关系**。）

效果：原本会齐刷刷撞 $\mathrm{B00}$ 的那一列，被「搅」到不同 bank 上，冲突减轻甚至消失。  
这叫 **swizzling**：改 shared memory 里的摆放，算法数学仍是同一个 matmul。

### 9.7 和后面的关系

Shared memory → bank conflict（本节）。  
HBM → coalescing（§10）。  
Grid / Block / Thread 怎么接上这三层存储器 → §12 的 matmul。

### 9.8 收束

列访问 + 行优先布局 → 经典 32-way；matmul 里常见。  
缓解：shared memory swizzling（例如行列异或重排）。

---

## 10. Memory Coalescing：HBM 上的「拼车」

Bank conflict 发生在 **shared memory（片上）**。  
**Memory coalescing** 发生在 **HBM / 全局显存（片外、更远更慢）**。两件事别混。

### 10.1 先建立直觉

HBM 像一座很远的大仓库。送货不是按「一个 thread 一趟车」，而是按比较大的包裹来回搬。

课上的关键数字：

$$
\text{一笔事务} \approx 128\ \mathrm{bytes}
$$

这对应常见的 **cache line** 宽度：硬件一次倾向于搬一整条 128-byte 的线，而不是你要 4 个字节就只搬 4 个字节。

一个 warp 有 32 个 thread。若每人读一个 4-byte 的数（例如一个 `float`）：

$$
32 \times 4\ \mathrm{bytes} = 128\ \mathrm{bytes}
$$

刚好等于一条 cache line。  
所以：**最好情况**是这 32 个人要的数据，正好落在**同一条 128-byte 线**里——硬件一趟搬完，人人有份。这就是 **full coalescing（完全合并）**。

### 10.2 幻灯片上的 `M00 M01 ... M63` 在说什么？

把全局内存按 4-byte word 排开，标成：

$$
\mathrm{M00},\; \mathrm{M01},\; \mathrm{M02},\; \ldots,\; \mathrm{M31},\; \mathrm{M32},\; \ldots,\; \mathrm{M63},\; \ldots
$$

可以想成两段各 32 个 word 的「车厢」：

$$
\begin{aligned}
\text{第一段 128 bytes：}&\quad \mathrm{M00}\ldots\mathrm{M31} \\
\text{第二段 128 bytes：}&\quad \mathrm{M32}\ldots\mathrm{M63}
\end{aligned}
$$

- Warp 里 32 个 thread 分别读 $\mathrm{M00},\mathrm{M01},\ldots,\mathrm{M31}$（连续、对齐）  
  → 落在同一条 128-byte 线 → **full coalescing** → 理想情况大约 **1 笔事务**。
- 若他们读得东一块西一块（地址很散，跨很多条 cache line）  
  → 硬件得开很多趟车 → **不 coalesced** → 慢，而且浪费带宽（每趟车可能只真正用到几个字节）。

### 10.3 「合并」到底合并了什么？

人话版：

1. Warp 发出 32 个「我要这个地址」的请求；  
2. 硬件看这些地址能不能装进尽量少的 128-byte 事务里；  
3. 能装进同一条线 → 合并成一笔；装不进 → 拆成多笔，串行或并行地去 HBM 取。

所以 coalescing 不是你在代码里写的函数，而是 **硬件对「一个 warp 的一次集体访存」做的打包**。你能做的是：让同一个 warp 的 thread **尽量访问邻近、对齐的地址**。

### 10.4 好例子 vs 坏例子

**好（容易 coalesced）：**  
处理一行向量 / 一行矩阵时，thread 0 读元素 0，thread 1 读元素 1，……，thread 31 读元素 31。地址连续。

**坏（很难 coalesced）：**  
按列跨很大 stride 读（例如每行很长，thread $t$ 去读第 $t$ 行的第 0 列）。32 个地址彼此隔得很远，可能落在 32 条不同的 cache line 上 → 最多接近 32 笔事务。

注意：这和 shared memory 的「按列 → bank conflict」**长得很像，但是另一层存储器**：

| | Shared memory | HBM |
|--|---------------|-----|
| 问题名 | Bank conflict | 不 coalesced |
| 硬件单元 | 32 个 bank | 128-byte 事务 / cache line |
| 典型惨案 | 同 bank 不同地址 | 地址太散，跨很多条 line |
| 修复思路 | swizzling 等 | 改访问顺序、布局、让 warp 读连续段 |

### 10.5 和 occupancy 的关系（先埋一笔）

HBM 很慢。即使你 coalescing 做得不错，访存仍有延迟。  
GPU 靠 **很多 warp 轮流干活** 把这段等待盖住：A warp 在等内存时，B warp 可以算。

若 occupancy 太低（同时活着的 warp 太少），等 HBM 时可能 **没人可换**——这时就算 coalescing 还可以，SM 仍容易闲着。  
下一节的绿块图，画的就是「SM 上同时摊着多少 block」。

---

## 11. Block Occupancy：幻灯片底部那张绿块图

### 11.1 图在画什么？

纵轴是 **SM**（一条产线），横轴是 **time**（时间往右流）。  
浅绿色小矩形 = 某个 **thread block** 住在这条 SM 上、正在跑（或至少占着资源）的一段时间。

你会看到：

- 某一时刻，绿块在纵向上叠了好几层 → 表示这条 SM **同时**住着多个 block；  
- 过一会儿叠得矮了 → 同时住着的 block 变少了；  
- 时间轴向右延伸 → 旧 block 做完退房，新 block 再搬进来。

这就是 **block occupancy 的时间动画**：不是静态公式，而是「资源允许的话，调度器会尽量把 SM 填满；填不满时，图上就叠得矮」。

### 11.2 和前面算出的 $0.1875$ 怎么对上？

还是课上那个寄存器受限的例子：

$$
\begin{aligned}
R_{\mathrm{block}} &= 20480 \\
B &= 3 \\
W &= 12 \\
\mathrm{occupancy} &= \frac{12}{64} = 0.1875 = 18.75\%
\end{aligned}
$$

含义：

- 硬件寄存器只够这条 SM **同时塞 3 个**这样的 block（$B=3$）；  
- 折合只有 12 个 warp 在活，而天花板是 64 → occupancy 只有约两成；  
- 反映在绿块图上：纵向上很难堆很高——不是调度器懒，是 **寄存器账不允许** 再堆。

图上若某一段看起来叠了 4 层、另一段只有 2 层，不必死抠层数是否等于 3：那是示意「有时更满、有时更空」。要点是：

$$
\text{同时能堆多高} \le \text{资源算出来的 } B
$$

寄存器、shared memory、$W_{\max}$ 等谁先撞墙，谁就决定这张图的「天花板高度」。

### 11.3 「Block occupancy」和「Warp occupancy」是不是一回事？

几乎在说同一类现象，只是计数单位不同：

| 说法 | 数的是什么 | 本例 |
|------|------------|------|
| Block 侧 | 同时住在 SM 上的 block 数 $B$ | $B=3$ |
| Warp 侧 | 同时活着的 warp 数 $W$，再除以 $W_{\max}$ | $W=12$，occupancy $=0.1875$ |

课上蓝框里的 `occupancy = 0.1875` 用的是 **warp 定义**（更标准）。  
绿块图用 **block** 画，是因为调度/退房的自然单位常常是 block，画起来更直观。

换算关系前面写过：

$$
W = \frac{B \cdot T}{32}
$$

本例 $T=128$，所以 $B=3$ 时 $W=12$。

### 11.4 为什么图会随时间变「高」变「矮」？

常见原因（直觉即可）：

1. **资源上限**：全程都高不上去 → 像本例，register-limited。  
2. **尾波（tail effect）**：总算快结束时，剩下的 block 不够填满所有 SM，图上变矮——不是算法突然变差，是活干完了。  
3. **不同 kernel / 不同配置**：换了 block 大小或寄存器用量，同一张 SM–time 图的「堆高」会变。

### 11.5 三件事怎么串成一条线？

按「数据从哪来 → 片上怎么挤 → 能不能掩盖等待」：

1. **HBM coalescing（§10）**  
   Warp 去远处取数时，尽量拼成 128-byte 的少趟车。  
2. **Shared memory bank conflict（§9）**  
   数搬进片上后，读的时候别在同一个 bank 排队。  
3. **Occupancy / 绿块图（§4–§6，§11）**  
   SM 上同时多住一点 block/warp，等 HBM 时才有人可换班。

本例寄存器太贵 → 绿块堆不高 → occupancy $=0.1875$ → 掩盖延迟的余量偏少。

---

## 12. 编程模型：用一次真实 matmul 走通

整节只用这一道题：

$$
C = A B,\qquad A,B,C \in \mathbb{R}^{1024 \times 1024}
$$

元素是 `float`（4 bytes）。$A$、$B$、$C$ 三份矩阵事先放在 **HBM**（全局显存）里。

### 12.1 怎么切活？

经典切法（tiled matmul）：

- 把 $C$ 切成 $16 \times 16$ 的小块（tile）。  
- **一个 thread** 负责 $C$ 里 **1 个** 输出元素。  
- **一个 thread block** 是 $16 \times 16 = 256$ 个 thread，一起算出 **一块** $16 \times 16$ 的 $C$。  
- **整个 grid** 要铺满 $C$，需要的 block 个数是

$$
\frac{1024}{16} \times \frac{1024}{16} = 64 \times 64 = 4096
$$

所以：

$$
\begin{aligned}
\text{Grid 的形状} &= (64,\ 64) \quad\text{（一共 4096 个 block）} \\
\text{每个 Block 的形状} &= (16,\ 16) \quad\text{（一共 256 个 thread）} \\
\text{Thread} &= 1\ \text{个输出元素}
\end{aligned}
$$

一次 kernel launch **只有一个 grid**。  
「Grid 多大」=「这次要开多少个 block、怎么排」。  
你的直觉对：一个 block 干不完整张 $C$，所以用 grid 把 4096 个 block 派出去。

### 12.2 CUDA 里怎么 launch？

用户侧（CUDA C++）长这样：

```cuda
dim3 block(16, 16);   // blockDim: 每个 block 256 threads
dim3 grid(64, 64);    // gridDim:  4096 blocks，铺满 1024×1024 的 C

matmul<<<grid, block>>>(A, B, C, 1024);
//              ↑      ↑
//           gridDim  blockDim
```

`matmul` 是你写好的 `__global__` kernel。  
`<<<grid, block>>>` 告诉驱动：按这个 grid / block 尺寸开火。  
`A, B, C` 是指向 HBM 的指针；`1024` 是矩阵边长。

Kernel 里每个 thread 用编号认领自己的输出坐标，例如：

```text
row = blockIdx.y * 16 + threadIdx.y
col = blockIdx.x * 16 + threadIdx.x
```

然后这个 thread 负责算 $C[\mathrm{row},\mathrm{col}]$。

硬件把每个 block 里的 256 个 thread 再捆成 warp：

$$
256 / 32 = 8\ \text{warps / block}
$$

Warp 是调度单位；launch 时你写的是 grid 和 block。

### 12.3 一个 block 里，数据怎么流？

盯住 **某一个** block，它要算 $C$ 上坐标 $(64, 32)$ 那一块 $16\times16$ tile（数字只是举例）。

**第 0 步 — 人就位**  
这个 block 被调度到某条 SM 上。256 个 thread 的私有累加器（`float sum = 0`）在 **寄存器** 里。

**第 1 步 — 从 HBM 搬 tile 进 shared memory**  
$K=1024$，按宽度 16 一段段扫。每一段：

- 大家合力把 $A$ 的一块 $16\times16$ 从 HBM 读进 **shared memory**；  
- 再把 $B$ 的一块 $16\times16$ 从 HBM 读进 **shared memory**；  
- 读 HBM 时，同一 warp 的地址尽量连续 → **coalescing**（§10）。

**第 2 步 — 在片上算**  
每个 thread 用 shared memory 里的一行 $A$、一列 $B$，往自己的寄存器 `sum` 里累加。  
同 block 的人读同一块 shared memory → 这里出现 **bank conflict** 的话题（§9）。

**第 3 步 — 下一段 K**  
重复「HBM → shared memory → 寄存器累加」，直到 $K$ 扫完（$1024/16=64$ 段）。

**第 4 步 — 写回 HBM**  
每个 thread 把寄存器里的 `sum` 写到 HBM 里的 $C[\mathrm{row},\mathrm{col}]$。

整张图就一句话：

$$
\mathrm{HBM}(A,B)
\;\rightarrow\;
\mathrm{Shared\ memory}(\text{当前 tile})
\;\rightarrow\;
\mathrm{Registers}(\texttt{sum})
\;\rightarrow\;
\mathrm{HBM}(C)
$$

### 12.4 三层名字各自管什么？

沿着上面这条数据流看：

- **Thread + 寄存器**：每人守住一个 $C$ 元素的 `sum`，算得最快的那层。  
- **Block + shared memory**：256 人共用当前 $A$、$B$ tile；一块 tile 只从 HBM 搬一次，大家反复用。  
- **Grid + HBM**：4096 个 block 合起来覆盖整张 $C$；$A$、$B$、$C$ 本体始终在 HBM，block 只是轮流把需要的小块搬上来。

所以常说的对应是：

- Grid 这一层的工作对象 = 整份 $A,B,C$（在 HBM）  
- Block 这一层的工作对象 = 当前 tile（在 shared memory）  
- Thread 这一层的工作对象 = 自己的 `sum`（在寄存器）

### 12.5 4096 个 block 上硬件时发生什么？

GPU 只有有限条 SM。4096 个 block 会 **分波次（waves）** 上车：一批 block 跑完腾出资源，下一批再上。

例如某卡有 148 条 SM。若某时刻平均每条 SM 同时住得下若干 block，wave 仍然按「还有多少 block 没做完」往前推。最后一波若只剩很少 block，部分 SM 空闲，这叫 **尾波（tail）**；和「单条 SM 里寄存器不够、同时只能住 3 个 block」（前面 occupancy $=0.1875$）是两本账：一本是 **整卡还有没有活**，一本是 **一条 SM 里住得下多少**。

### 12.6 收束

对 $1024\times1024$ matmul、tile $=16$：

$$
\texttt{matmul<<<dim3(64,64),\ dim3(16,16)>>>(A, B, C, 1024)}
$$

- 1 个 grid = $64\times64=4096$ 个 block  
- 1 个 block = $16\times16=256$ 个 thread $=8$ 个 warp  
- 1 个 thread = 1 个 $C$ 元素  
- 矩阵在 HBM；tile 在 shared memory；`sum` 在寄存器  

这就是 Grid / Block / Thread 这套模型存在的理由：用三层把「整张输出怎么铺满、一群人怎么共享一块 tile、一个人怎么守住一个元素」说清楚，并且刚好对上 HBM / shared memory / registers 三层存储器。

