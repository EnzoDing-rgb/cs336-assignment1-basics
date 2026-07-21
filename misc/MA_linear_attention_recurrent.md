# 线性注意力的递推形式（Recurrent form of linear attention）

---

## 0. 符号表（先看这里）

### 0.1 两种「T」——必须分开

写下来长得很像，**是两件完全不同的事**：

| 写法 | 读法 | 是什么 | 例子 |
|------|------|--------|------|
| $T$（斜体大写，单独出现） | 「大 Tee」/ 序列长度 | **整数**：序列有多少个位置 | 本笔记 $T=3$（三个 token 位置） |
| $^{\top}$（贴在矩阵/向量右上角） | 「转置」transpose | **运算**：行列互换 | $K^{\top}$ = 把 $K$ 转置；若 $K$ 是 $3\times 2$，则 $K^{\top}$ 是 $2\times 3$ |

对照：

$$
\begin{aligned}
T &= 3 &&\text{（长度，一个数）}\\
K^{\top} &&&\text{（对矩阵 }K\text{ 做转置，得到另一块矩阵）}
\end{aligned}
$$

- **telling / 方块**：都不是。  
- **时间**：小写 $t=1,2,\ldots,T$ 是位置下标（第几个 token）；大写 $T$ 是「一共有多少个位置」。口语里常说「时间步」，数学上就是位置编号 $t$，总长 $T$。  
- PPT 上若把转置写成 $K^{T}$（普通正体 $T$），那是老写法，**仍表示转置**；本文统一写成 $K^{\top}$，避免和长度 $T$ 撞车。

### 0.2 全文符号

| 符号 | 是什么 |
|------|--------|
| $T$ | 序列长度（整数）。本笔记 $T=3$ |
| $t$ | 位置下标，$t=1,2,\ldots,T$ |
| $d$ | 每个 $q/k/v$ 的维数。本笔记 $d=2$ |
| $q_t$ | 位置 $t$ 的 query，列向量 $d\times 1$：用来「去取信息」 |
| $k_t$ | 位置 $t$ 的 key，列向量 $d\times 1$：用来「被匹配」 |
| $v_t$ | 位置 $t$ 的 value，列向量 $d\times 1$：要贡献出去的内容 |
| $Q$ | $T\times d$ 矩阵，第 $t$ 行是 $q_t^{\top}$ |
| $K$ | $T\times d$ 矩阵，第 $t$ 行是 $k_t^{\top}$ |
| $V$ | $T\times d$ 矩阵，第 $t$ 行是 $v_t^{\top}$ |
| $^{\top}$ | 转置（transpose），不是长度 |
| $Q K^{\top}$ | 先把 $K$ **转置**再左乘 $Q$，得到 $T\times T$ 分数矩阵；条目 $A_{t,s}=q_t^{\top} k_s$ |
| $K^{\top} V$ | $d\times d$ 聚合矩阵；等于 $\sum_{t=1}^{T} k_t v_t^{\top}$ |
| $S_t$ | 前缀状态 $\sum_{s\le t} k_s v_s^{\top}$（递推里带着往前走的那块 $d\times d$） |
| $y_t$ | 位置 $t$ 的输出；因果下 $y_t^{\top}=q_t^{\top} S_t$ |
| $Y$ | $T\times d$ 输出矩阵，第 $t$ 行是 $y_t^{\top}$ |

---

## 课上式子

$$
\begin{aligned}
\text{标准写法：}& \quad \text{输出} \propto Q K^{\top} V \\
\text{线性注意力重排：}& \quad Q K^{\top} V = Q (K^{\top} V) \\
\text{递推：}& \quad S_{t} = S_{t-1} + k_{t} v_{t}^{\top},\quad
y_{t}^{\top} = q_{t}^{\top} S_{t}
\end{aligned}
$$

下面用同一组小数把两条路算一遍，核对结果相同。

---

## 1. 本笔记用的具体数字

一次前向：长度 $T=3$，维数 $d=2$。

$$
\begin{aligned}
t=1:&\quad q_1=\begin{bmatrix}1\\0\end{bmatrix},\;
k_1=\begin{bmatrix}1\\0\end{bmatrix},\;
v_1=\begin{bmatrix}2\\0\end{bmatrix} \\
t=2:&\quad q_2=\begin{bmatrix}0\\1\end{bmatrix},\;
k_2=\begin{bmatrix}0\\1\end{bmatrix},\;
v_2=\begin{bmatrix}0\\3\end{bmatrix} \\
t=3:&\quad q_3=\begin{bmatrix}1\\1\end{bmatrix},\;
k_3=\begin{bmatrix}1\\1\end{bmatrix},\;
v_3=\begin{bmatrix}1\\1\end{bmatrix}
\end{aligned}
$$

对应矩阵：

$$
Q=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
V=\begin{bmatrix}2&0\\0&3\\1&1\end{bmatrix}
$$

---

## 2. 标准注意力里 $Q K^{\top} V$ 在算什么

先算分数矩阵：

$$
A = Q K^{\top}\qquad\text{形状 }T\times T
$$

例子：

$$
K^{\top}=\begin{bmatrix}1&0&1\\0&1&1\end{bmatrix},\quad
A=Q K^{\top}=\begin{bmatrix}1&0&1\\0&1&1\\1&1&2\end{bmatrix}
$$

含义：$A_{t,s} = q_t^{\top} k_s$，即位置 $t$ 对位置 $s$ 的匹配分数。

再乘 value：

$$
Y = A V = (Q K^{\top}) V\qquad\text{形状 }T\times d
$$

例子：

$$
Y=\begin{bmatrix}1&0&1\\0&1&1\\1&1&2\end{bmatrix}
\begin{bmatrix}2&0\\0&3\\1&1\end{bmatrix}
=\begin{bmatrix}3&1\\1&4\\4&5\end{bmatrix}
$$

所以：

$$
y_1^{\top}=[3,\;1],\quad
y_2^{\top}=[1,\;4],\quad
y_3^{\top}=[4,\;5]
$$

代价：$Q K^{\top}$ 产生 $T\times T$ 的矩阵。$T$ 变大时，这块随 $T^{2}$ 变大。

（标准 Softmax 注意力还会在 $A$ 上做逐行 Softmax；线性注意力把「相似度」换成可拆开的特征映射，核心代数仍是「先形成某种 $Q$–$K$ 交互，再乘 $V$」。本节先盯住矩阵乘顺序本身。）

---

## 3. 线性注意力的重排：$Q(K^{\top} V)$

矩阵乘法满足结合律：

$$
(Q K^{\top}) V = Q (K^{\top} V)
$$

左边：先做 $Q K^{\top}$（$T\times T$），再乘 $V$。  
右边：先做 $K^{\top} V$（$d\times d$），再左乘 $Q$。

继续用同一组数算右边。

### 3.1 先算状态矩阵 $S = K^{\top} V$

$$
K^{\top}\in\mathbb{R}^{d\times T},\quad
V\in\mathbb{R}^{T\times d},\quad
S=K^{\top} V\in\mathbb{R}^{d\times d}
$$

例子：

$$
K^{\top} V
=\begin{bmatrix}1&0&1\\0&1&1\end{bmatrix}
\begin{bmatrix}2&0\\0&3\\1&1\end{bmatrix}
=\begin{bmatrix}3&1\\1&4\end{bmatrix}
$$

逐步展开一行：

$$
\begin{aligned}
S_{11}&=1\cdot 2+0\cdot 0+1\cdot 1=3,\\
S_{12}&=1\cdot 0+0\cdot 3+1\cdot 1=1,\\
S_{21}&=0\cdot 2+1\cdot 0+1\cdot 1=1,\\
S_{22}&=0\cdot 0+1\cdot 3+1\cdot 1=4.
\end{aligned}
$$

### 3.2 再算 $Y = Q S$

$$
Y = Q (K^{\top} V) = Q S
$$

例子：

$$
Y=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}
\begin{bmatrix}3&1\\1&4\end{bmatrix}
=\begin{bmatrix}3&1\\1&4\\4&5\end{bmatrix}
$$

与第 2 节的 $Y$ 逐元素相同。

### 3.3 两种顺序差在哪里

| 顺序 | 中间大矩阵 | 中间形状 |
|------|------------|----------|
| $(Q K^{\top})V$ | $Q K^{\top}$ | $T\times T$ |
| $Q(K^{\top} V)$ | $K^{\top} V$ | $d\times d$ |

序列很长、$d$ 远小于 $T$ 时，$d\times d$ 的中间结果更省。  
这就是课上说的：线性注意力走 $Q(K^{\top} V)$ 这条代数路径。

---

## 4. 把 $K^{\top} V$ 拆成「一个位置加一项」

$K^{\top} V$ 按位置展开。第 $t$ 个位置贡献一项外积：

$$
k_t v_t^{\top}\in\mathbb{R}^{d\times d}
$$

$t=1$：

$$
k_1 v_1^{\top}
=\begin{bmatrix}1\\0\end{bmatrix}[2,\;0]
=\begin{bmatrix}2&0\\0&0\end{bmatrix}
$$

$t=2$：

$$
k_2 v_2^{\top}
=\begin{bmatrix}0\\1\end{bmatrix}[0,\;3]
=\begin{bmatrix}0&0\\0&3\end{bmatrix}
$$

$t=3$：

$$
k_3 v_3^{\top}
=\begin{bmatrix}1\\1\end{bmatrix}[1,\;1]
=\begin{bmatrix}1&1\\1&1\end{bmatrix}
$$

三项相加：

$$
\begin{bmatrix}2&0\\0&0\end{bmatrix}
+
\begin{bmatrix}0&0\\0&3\end{bmatrix}
+
\begin{bmatrix}1&1\\1&1\end{bmatrix}
=
\begin{bmatrix}3&1\\1&4\end{bmatrix}
$$

这正是前面的 $S = K^{\top} V$。

因此：

$$
K^{\top} V = \sum_{t=1}^{T} k_t v_t^{\top}
$$

---

## 5. 递推形式：一个状态，从左往右更新

推理与因果训练里，位置 $t$ 只聚合 $s \le t$ 的 key/value。定义前缀状态：

$$
S_t = \sum_{s=1}^{t} k_s v_s^{\top}
$$

由求和得到递推：

$$
S_0 = 0_{d\times d},\qquad
S_t = S_{t-1} + k_t v_t^{\top}
$$

位置 $t$ 的输出（行向量写法，与课上一致）：

$$
y_t^{\top} = q_t^{\top} S_t
$$

（$q_t$ 为列向量时，$q_t^{\top} S_t$ 是 $1\times d$ 的行；它就是输出矩阵 $Y$ 的第 $t$ 行。）

### 5.1 逐步手算

**$t=1$**

$$
S_1 = k_1 v_1^{\top}
=\begin{bmatrix}2&0\\0&0\end{bmatrix},\quad
y_1^{\top}=q_1^{\top} S_1=[1,\;0]
\begin{bmatrix}2&0\\0&0\end{bmatrix}
=[2,\;0]
$$

**$t=2$**

$$
S_2 = S_1 + k_2 v_2^{\top}
=\begin{bmatrix}2&0\\0&0\end{bmatrix}
+
\begin{bmatrix}0&0\\0&3\end{bmatrix}
=\begin{bmatrix}2&0\\0&3\end{bmatrix}
$$

$$
y_2^{\top}=q_2^{\top} S_2=[0,\;1]
\begin{bmatrix}2&0\\0&3\end{bmatrix}
=[0,\;3]
$$

**$t=3$**

$$
S_3 = S_2 + k_3 v_3^{\top}
=\begin{bmatrix}2&0\\0&3\end{bmatrix}
+
\begin{bmatrix}1&1\\1&1\end{bmatrix}
=\begin{bmatrix}3&1\\1&4\end{bmatrix}
$$

$$
y_3^{\top}=q_3^{\top} S_3=[1,\;1]
\begin{bmatrix}3&1\\1&4\end{bmatrix}
=[4,\;5]
$$

完整前缀结果：

$$
y_1^{\top}=[2,\;0],\quad
y_2^{\top}=[0,\;3],\quad
y_3^{\top}=[4,\;5],\quad
S_3=K^{\top} V=\begin{bmatrix}3&1\\1&4\end{bmatrix}
$$

### 5.2 与第 3 节并行公式的关系

第 3 节的 $Y = Q(K^{\top} V)$ 让 **每个** $q_t$ 都乘 **同一个** 最终 $S_T = K^{\top} V$，得到：

$$
Y=\begin{bmatrix}3&1\\1&4\\4&5\end{bmatrix}
$$

那是 **每个位置都看见整段** 的版本。  
第 5 节递推是 **每个位置只看见到自己为止的前缀** 的版本（因果）。  
训练若要与递推推理一致，并行实现里对「未来位置」加掩码，使位置 $t$ 只用 $S_t$，两边的 $y_t$ 相同。

---

## 6. 为什么说这「像 RNN」（你只需记住状态更新）

RNN 背景：你可以稍后再学。这里只用到一件事：

```text
每来一个新位置 t：
  1. 用旧状态 S_{t-1} 和新的 (k_t, v_t) 算出新状态 S_t
  2. 用 S_t 和 q_t 算出当前输出 y_t
  3. 把 S_t 留给下一步
```

状态 $S_t$ 是固定形状 $d\times d$ 的矩阵，与序列已经多长无关。  
走完 $T$ 步，仍只保存一个 $d\times d$ 的 $S_T$。

逐步过程：

$$
\begin{aligned}
S_0 &= 0 \\
t=1:&\quad S_1 \leftarrow S_0 + k_1 v_1^{\top},\quad
\text{输出} \leftarrow q_1^{\top} S_1 \\
t=2:&\quad S_2 \leftarrow S_1 + k_2 v_2^{\top},\quad
\text{输出} \leftarrow q_2^{\top} S_2 \\
t=3:&\quad S_3 \leftarrow S_2 + k_3 v_3^{\top},\quad
\text{输出} \leftarrow q_3^{\top} S_3
\end{aligned}
$$

这就是课上的：

$$
S_t = S_{t-1} + k_t v_t^{\top},\qquad
y_t^{\top} = q_t^{\top} S_t
$$

---

## 7. 对偶：训练走并行，推理走串行

同一套数学（因果线性注意力）有两种算法外形。

### 7.1 并行外形（训练常用）

一次拿出整段 $Q,K,V$，按块做矩阵乘（实现上用掩码保证因果）：

$$
\text{对每个 }t:\quad
y_t^{\top} = q_t^{\top}\Big(\sum_{s\le t} k_s v_s^{\top}\Big)
$$

GPU 上适合整段一起算。代数上仍围绕 $d\times d$ 的聚合，以及按位置施加的因果掩码。

### 7.2 串行外形（推理常用）

按时间一步一步来：

$$
\begin{aligned}
&S \leftarrow 0_{d\times d} \\
&\text{for } t=1..T: \\
&\quad S \leftarrow S + k_t v_t^{\top} \\
&\quad y_t^{\top} \leftarrow q_t^{\top} S
\end{aligned}
$$

生成第 $t$ 个 token 时，做一次 $d\times d$ 的加法更新，再做一次 $q_t$ 与 $S$ 的乘。  
状态始终是一个 $d\times d$ 的 $S$。

### 7.3 「对偶」四个字在说什么

```text
同一种 y_t 的定义
    ├─ 写成整段矩阵公式  → 便于训练时并行算
    └─ 写成 S_t 递推      → 便于推理时逐步算
```

两边算出的 $y_t$ 相同（同一因果约定下）。  
课上那句 duality 指的就是：**一个公式，两种算法外形。**

---

## 8. 一条线串起来

1. 注意力输出由 $Q$、$K$、$V$ 经矩阵乘得到。  
2. $(Q K^{\top})V$ 与 $Q(K^{\top} V)$ 在数字上可以相同（结合律）。  
3. $K^{\top} V$ 是 $d\times d$，并且等于每个位置外积 $k_t v_t^{\top}$ 的和。  
4. 把这个和写成前缀：$S_t = S_{t-1} + k_t v_t^{\top}$。  
5. 因果输出：$y_t^{\top} = q_t^{\top} S_t$。  
6. 同一定义：训练可用整段并行实现；推理可用 $S_t$ 一步一步更新。

课上那三行，对应的就是第 4–5 步；「像 RNN」指的是第 4 步那种 **固定大小状态、逐步更新** 的外形。
