# All-reduce / Reduce-scatter / All-gather

三个都是 **多卡（多 rank）之间的集体通信**。  
先记一句：

$$
\mathrm{All\text{-}reduce}
=
\mathrm{Reduce\text{-}scatter}
+
\mathrm{All\text{-}gather}
$$

公式约定：行内 `$...$`，独立成行 `$$...$$`。

---

## 0. 场景

假设 4 张卡：$\mathrm{rank}=0,1,2,3$。  
每张卡上有一个长度为 4 的向量（课上例子）：

$$
\begin{aligned}
\mathrm{rank0} &= [0,\ 1,\ 2,\ 3] \\
\mathrm{rank1} &= [1,\ 2,\ 3,\ 4] \\
\mathrm{rank2} &= [2,\ 3,\ 4,\ 5] \\
\mathrm{rank3} &= [3,\ 4,\ 5,\ 6]
\end{aligned}
$$

训练里常见：每张卡算完自己那份数据的梯度，需要和别的卡合在一起。

---

## 1. Reduce-scatter：先「按位置求和」，再「每人只留一段」

**Reduce** = 同一位置跨卡求和。  
**Scatter** = 求和结果切开，每张卡只拿走属于自己的那一段。

对课上例子，按 **列下标** 跨 4 张卡求和：

$$
\begin{aligned}
\text{下标 }0&:\ 0+1+2+3 = 6 \\
\text{下标 }1&:\ 1+2+3+4 = 10 \\
\text{下标 }2&:\ 2+3+4+5 = 14 \\
\text{下标 }3&:\ 3+4+5+6 = 18
\end{aligned}
$$

然后切开分给四张卡：

$$
\begin{aligned}
\mathrm{rank0} &\leftarrow [6] \\
\mathrm{rank1} &\leftarrow [10] \\
\mathrm{rank2} &\leftarrow [14] \\
\mathrm{rank3} &\leftarrow [18]
\end{aligned}
$$

课上 use case：backward 之后要把各 data shard 的梯度加总，但 **存储仍然切开**——每人只持有全局和的一部分。

---

## 2. All-gather：每人把手上那一段广播给所有人

**Gather** = 把散落在各卡上的片段收齐。  
**All** = 每张卡最后都拿到完整拼好的结果。

接上一步 reduce-scatter 的输出：

$$
\begin{aligned}
\mathrm{rank0\ 有}\ &[6] \\
\mathrm{rank1\ 有}\ &[10] \\
\mathrm{rank2\ 有}\ &[14] \\
\mathrm{rank3\ 有}\ &[18]
\end{aligned}
$$

All-gather 之后，**每张卡**都变成：

$$
[6,\ 10,\ 14,\ 18]
$$

---

## 3. All-reduce：每人最终都拿到「完整的全局和」

**All-reduce** = 跨卡 reduce（通常是 sum），并且 **每张卡都得到同一份完整结果**。

对课上同一组输入，直接 all-reduce 后：

$$
\begin{aligned}
\mathrm{rank0} &= [6,\ 10,\ 14,\ 18] \\
\mathrm{rank1} &= [6,\ 10,\ 14,\ 18] \\
\mathrm{rank2} &= [6,\ 10,\ 14,\ 18] \\
\mathrm{rank3} &= [6,\ 10,\ 14,\ 18]
\end{aligned}
$$

实现上常常拆成两步，所以有恒等式：

$$
\mathrm{All\text{-}reduce}
=
\mathrm{Reduce\text{-}scatter}
+
\mathrm{All\text{-}gather}
$$

先按块求和并切开（reduce-scatter），再把各块拼回每张卡（all-gather）。

---

## 4. 三张对照

| 名字 | 结束时每张卡有什么 | 一句话 |
|------|-------------------|--------|
| Reduce-scatter | 全局和的 **一段** | 求和 + 切开 |
| All-gather | 别人的片段也拿到，拼成 **完整向量** | 大家交换碎片，拼齐全貌 |
| All-reduce | 完整的 **全局和**（人人一份） | 求和，且人人都有全量结果 |

---

## 5. 和数据并行梯度的关系

数据并行常见需求：各卡梯度求和后，**每张卡都要用同一份更新后的梯度**。  
那就是 all-reduce。

若框架先 reduce-scatter 再 all-gather，数学结果与一次 all-reduce 相同；通信实现可以更省或更好流水线，课上强调的是这个分解。
