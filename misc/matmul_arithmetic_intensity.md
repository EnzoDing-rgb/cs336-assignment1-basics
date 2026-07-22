# Matmul 算术强度：naive 到理想

课上对比：naive matmul 每次算都去 HBM 取数；理想情况把用到的 A、B 读进 shared memory 再算。  
关键量是算术强度 AI（arithmetic intensity）。

公式约定：

- 行内用单美元符号包起来
- 独立公式单独成行，用双美元符号包起来
- 不用 `\#`、不在 `\mathrm{...}` 里塞空格、表格单元格里尽量只放短公式

全程用一般矩形尺寸。

---

## 0. 尺寸约定

$$
C = A B
$$

$$
A \in \mathbb{R}^{M \times K}
$$

$$
B \in \mathbb{R}^{K \times N}
$$

$$
C \in \mathbb{R}^{M \times N}
$$

单个元素：

$$
C_{mn} = \sum_{k=0}^{K-1} A_{mk} B_{kn}
$$

符号含义：

- $M$：输出行数，也是 $A$ 的行数
- $N$：输出列数，也是 $B$ 的列数
- $K$：缩并维；$A$ 的列数等于 $B$ 的行数

下文缩写：

- $F$：浮点运算次数（FLOPs，乘和加都算）
- $E$：HBM 读/写的元素个数

---

## 1. 算术强度 AI

$$
\mathrm{AI} = \frac{F}{E}
$$

- $F$：算了多少次
- $E$：从 HBM 搬了多少个元素

人话：每从 HBM 搬一个数，片上能干多少次乘加。  
$\mathrm{AI}$ 越大，越不那么被内存带宽卡住。

---

## 2. $F = 2 M N K$ 从哪来

对一个输出 $C_{mn}$：

- 乘：$K$ 次（每个 $k$ 做一次 $A_{mk} \cdot B_{kn}$）
- 加：大约 $K$ 次（把 $K$ 个乘积累加）

所以单个 $C_{mn}$ 大约 $2K$ 次运算。

一共有 $M N$ 个输出：

$$
F = (M N) \cdot (2 K) = 2 M N K
$$

只数乘法时分子会写成 $M N K$；课上乘加都算，用 $2 M N K$。  
两者大阶都是 $O(M N K)$。

---

## 3. Naive：$AI = O(1)$

骨架：每个三元组 $(m,n,k)$ 都从 HBM 读 $A_{mk}$ 和 $B_{kn}$，再乘加。

循环体执行 $M N K$ 次，每次大约读 2 个输入元素：

$$
E_{\mathrm{read}} = O(M N K)
$$

$$
E_{\mathrm{write}} = M N
$$

主导项是读，所以总流量

$$
E_{\mathrm{naive}} = O(M N K)
$$

于是

$$
\mathrm{AI}_{\mathrm{naive}} = \frac{2 M N K}{O(M N K)} = O(1)
$$

$M,N,K$ 变大时，算变多，HBM 读也差不多同比例变多，强度不涨。

---

## 4. 理想情况：整块 A、B 进片上

假设 shared memory 一次放得下整个 $A$ 和整个 $B$：

1. 读完 $A$：搬 $M K$ 个元素
2. 读完 $B$：搬 $K N$ 个元素
3. 片上完成全部运算：$F = 2 M N K$
4. 写回 $C$：搬 $M N$ 个元素

总 HBM 流量：

$$
E_{\mathrm{ideal}} = M K + K N + M N
$$

算术强度：

$$
\mathrm{AI}_{\mathrm{ideal}} = \frac{2 M N K}{M K + K N + M N}
$$

### 4.1 方阵特例（课上写 $O(N)$ 指这个）

若 $M = N = K$，三个尺寸都等于同一个 $N$：

$$
\mathrm{AI}_{\mathrm{ideal}} = \frac{2 N^{3}}{N^{2} + N^{2} + N^{2}} = \frac{2 N^{3}}{3 N^{2}} = \frac{2}{3} N
$$

所以

$$
\mathrm{AI}_{\mathrm{ideal}} = O(N)
$$

这是特例。一般矩形请用

$$
\mathrm{AI}_{\mathrm{ideal}} = \frac{2 M N K}{M K + K N + M N}
$$

### 4.2 一般情形的直觉

分母是三块面积 $M K$、$K N$、$M N$ 之和，分子是 $2 M N K$。  
当 $M,N,K$ 同量级变大时，分子多一个线性因子，所以 $\mathrm{AI}$ 随问题尺寸上涨；方阵时就是 $O(N)$。

---

## 5. 对照

| 方案 | $F$ | $E$ | $\mathrm{AI}$ |
|------|-----|-----|---------------|
| Naive | $2 M N K$ | $O(M N K)$ | $O(1)$ |
| 理想 | $2 M N K$ | $M K + K N + M N$ | $\frac{2 M N K}{M K + K N + M N}$ |
| 理想且 $M=N=K$ | $2 N^{3}$ | $3 N^{2}$ | $\frac{2}{3} N = O(N)$ |

同一套 $C = A B$。差别只在：复用 $A_{mk}$、$B_{kn}$ 时，还要不要再去 HBM。

---

## 6. 和 tiling 的关系

现实里 $A$、$B$ 往往塞不进 shared memory，所以做 tiling：  
每次只搬能塞下的 $A$ 子块、$B$ 子块进片上，在块内尽量复用，让实际 $\mathrm{AI}$ 往理想公式靠，而不是停在 naive 的 $O(1)$。

---

## 7. 数字验算

取

$$
M = 2, \quad K = 3, \quad N = 2
$$

运算次数：

$$
F = 2 \cdot 2 \cdot 2 \cdot 3 = 24
$$

理想 HBM 流量：

$$
E_{\mathrm{ideal}} = 2\cdot 3 + 3\cdot 2 + 2\cdot 2 = 6 + 6 + 4 = 16
$$

算术强度：

$$
\mathrm{AI}_{\mathrm{ideal}} = \frac{24}{16} = 1.5
$$

直接代入一般公式：

$$
\frac{2\cdot 2\cdot 2\cdot 3}{2\cdot 3 + 3\cdot 2 + 2\cdot 2} = \frac{24}{16} = 1.5
$$

两边一致。
