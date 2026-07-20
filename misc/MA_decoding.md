# MA：Decoding（温度采样 + Top-p / Nucleus）

对应 CS336 Assignment 1：`Problem (decoding)`（讲义约 §6 Decoding / Decoder tricks）。

公式约定：行内 `$...$`，独立成行 `$$...$$`。不跳步。带玩具例子。

---

## 0. 这题在整条链路里干什么？

训练好的 Transformer LM 会：给定前文，吐出 **词表上每个 token 的分数（logits）**。  
**Decoding** = 用这些分数反复「选下一个 token、接到后面、再送进模型」，直到停。

```text
prompt 文本
   │ encode
   ▼
token 序列 x_1 … x_t
   │
   ▼ 循环：
   TransformerLM(x) → 只要最后位置的 logits v
   →（可选）温度缩放 → softmax → 分布 q
   →（可选）top-p 截断再归一化
   → 从分布里采样一个 id
   → 接到序列末尾；若是 <|endoftext|> 或达到 max_tokens 则停
   │
   ▼ decode
生成文本
```

和训练的差别：训练用真实下一词算 CE；解码用模型自己吐出的词当下一轮输入（自回归）。

---

## 1. 一步里模型到底给出什么？

记当前已有前缀

$$
x_{1:t} = (x_1,\ldots,x_t)
$$

模型前向（形状直觉）：

$$
\texttt{TransformerLM}(x_{1:t})
\in \mathbb{R}^{t \times V}
\qquad (V=\texttt{vocab\_size})
$$

（若带 batch 维，则是 $B\times t\times V$；解码常取 $B=1$。）

**下一步只关心最后一个位置**的那一行：

$$
v = \texttt{TransformerLM}(x_{1:t})_{t}
\in \mathbb{R}^{V}
\qquad\text{（讲义式 (22)）}
$$

$v_i$ = 「下一个词是词表第 $i$ 号」的原始分数（logit），还不是概率。

标准下一词分布（讲义式 (21)）：

$$
P(x_{t+1}=i \mid x_{1:t})
=
\frac{\exp(v_i)}{\sum_{j=1}^{V}\exp(v_j)}
=
\mathrm{softmax}(v)_i
$$

然后从该分布 **采样**（或取 argmax）得到 $x_{t+1}$，拼成 $x_{1:t+1}$，再跑一轮。

停止条件（作业要求）：

| 条件 | 含义 |
|------|------|
| 采到 `<\|endoftext\|>` | 模型认为这段结束（该 special token 的 id） |
| 达到用户设的 max tokens | 强制截断，防止无限生成 |

---

## 2. Decoder trick ①：Temperature（温度）

### 2.1 公式

温度 $\tau>0$，先把 logits 除以 $\tau$ 再 softmax（讲义式 (23)）：

$$
\mathrm{softmax}(v,\tau)_i
=
\frac{\exp(v_i/\tau)}{\sum_{j=1}^{V}\exp(v_j/\tau)}
$$

记得到的分布为 $q$（下面 top-p 也用这个 $q$）。

### 2.2 在干什么？（两边对比，都算完）

温度 **不改哪个 logit 最大**，只改概率有多「尖」或「平」。

玩具词表只有 3 个词，logits：

$$
v = (3.0,\ 1.0,\ 0.0)
$$

**低温度** $\tau=0.5$（更尖，更贪心）：

| token | $v_i/\tau$ | $\exp(v_i/\tau)$ | $q_i$ |
|-------|------------|------------------|-------|
| A | $6.0$ | $403.4$ | $\approx0.88$ |
| B | $2.0$ | $7.39$ | $\approx0.016$ |
| C | $0.0$ | $1.0$ | $\approx0.002$ |
| （归一化后 A 约 **0.98** 量级，B/C 极小——表内用相对直觉；精确算：$\sum e^{v/\tau}\approx411.8$，故 $q_A\approx0.980$，$q_B\approx0.018$，$q_C\approx0.002$） |

**高温度** $\tau=2.0$（更平，更随机）：

| token | $v_i/\tau$ | $\exp(v_i/\tau)$ | $q_i$ |
|-------|------------|------------------|-------|
| A | $1.5$ | $4.48$ | $\approx0.70$ |
| B | $0.5$ | $1.65$ | $\approx0.26$ |
| C | $0.0$ | $1.00$ | $\approx0.16$ |
| 精确：$\sum\approx7.13$ → $q_A\approx0.628$，$q_B\approx0.231$，$q_C\approx0.140$ |

两边都算完了：

| | $\tau=0.5$（低） | $\tau=2.0$（高） |
|--|------------------|------------------|
| $q_A$（原本最大） | $\approx0.98$ | $\approx0.63$ |
| $q_B$ | $\approx0.018$ | $\approx0.23$ |
| $q_C$ | $\approx0.002$ | $\approx0.14$ |
| 行为 | 几乎总抽 A（接近 greedy） | A 仍最可能，但 B/C 常出现 |

极限（讲义）：

- $\tau\to 0^{+}$：质量集中在 $\arg\max_i v_i$ 上 → 行为像 **greedy**  
- $\tau=1$：普通 softmax  
- $\tau>1$：分布变平，多样性升，胡话风险也升  

实现时注意：$\tau$ 太接近 0 会数值爆炸，常用一个很小的下限，或 $\tau=0$ 时直接走 `argmax`。

---

## 3. Decoder trick ②：Nucleus / Top-p 采样

小模型、温度采样有时仍会抽出「长尾」里的怪词。  
**Top-p（nucleus）**：先按概率从高到低累加，只保留「刚好凑满概率质量 $p$」的最小词集合，其余概率置 0，再在这个集合里重新归一化后采样。

### 3.1 定义（讲义）

$q$：已经过（温度）softmax 的分布，长度 $V$。  
超参 $p\in(0,1]$（例如 $0.9$）。

$V^{(p)}$ = **最小的**下标集合，使得

$$
\sum_{j\in V^{(p)}} q_j \ge p
$$

且集合里都是概率最大的那些词（按 $q$ 从大到小贪心累加得到）。

新的采样分布（讲义式 (24)）：

$$
P(x_{t+1}=i\mid q)
=
\begin{cases}
\dfrac{q_i}{\sum_{j\in V^{(p)}} q_j}
&
\text{若 } i\in V^{(p)}
\\[0.8em]
0
&
\text{否则}
\end{cases}
$$

### 3.2 怎么算 $V^{(p)}$？（算法，不跳步）

1. 把 $q$ **按从大到小排序**，记下排序后的概率和对应的词 id  
2. 做前缀和（cumulative sum）  
3. 找到 **第一个** 让前缀和 $\ge p$ 的位置 $r$  
4. $V^{(p)}$ = 排序后的前 $r$ 个词（含第 $r$ 个）  
5. 只保留这些位置的 $q_i$，其余变 0，再除以它们的和，得到新分布  
6. 按新分布采样

「最小集合」⇔ 按概率从高到低加，加到刚好够 $p$ 就停；不要乱加低概率词。

### 3.3 玩具例子（$p$ 小 vs $p$ 大，两边都算完）

词表 5 个词，温度 softmax 后：

$$
\begin{align*}
q(\texttt{the})&=0.40 \\
q(\texttt{a})&=0.25 \\
q(\texttt{cat})&=0.15 \\
q(\texttt{dog})&=0.12 \\
q(\texttt{xyz})&=0.08
\end{align*}
$$

已按从大到小排好。前缀和：

| 累加到 | 前缀和 |
|--------|--------|
| the | $0.40$ |
| + a | $0.65$ |
| + cat | $0.80$ |
| + dog | $0.92$ |
| + xyz | $1.00$ |

**$p=0.5$（核更小）**

- $0.40 < 0.5$，继续  
- $0.65 \ge 0.5$ → 停  
- $V^{(0.5)}=\{\texttt{the},\texttt{a}\}$  
- 质量和 $=0.65$  
- 新分布：$P(\texttt{the})=0.40/0.65\approx0.615$，$P(\texttt{a})=0.25/0.65\approx0.385$，其余 **0**

**$p=0.9$（核更大）**

- 累加到 dog：$0.92\ge0.9$ → 停  
- $V^{(0.9)}=\{\texttt{the},\texttt{a},\texttt{cat},\texttt{dog}\}$（**还没轮到** `xyz`）  
- 质量和 $=0.92$  
- 新分布：四者各除以 $0.92$，`xyz` 仍为 **0**

| | $p=0.5$ | $p=0.9$ |
|--|---------|---------|
| 集合大小 | 2 个词 | 4 个词 |
| `xyz`（长尾） | 概率 0 | 概率 0 |
| `cat` | 概率 0 | 仍可被采到 |
| 行为 | 更保守，几乎只在 the/a | 更开放，但仍砍掉最烂的尾巴 |

两边都算完了：$p$ 不是「温度」，而是 **「保留多少概率质量」**；$p$ 越大核越大。

### 3.4 和温度的关系（别混）

| | Temperature $\tau$ | Top-p $p$ |
|--|---------------------|-----------|
| 作用对象 | 先改 logits 再 softmax，动整表形状 | 在已有 $q$ 上砍长尾再归一化 |
| 典型组合 | 先 $\tau$，得到 $q$，再 top-p | 作业两者都要支持 |
| $\tau\to0$ | 接近 greedy | 若再 top-p，核里往往只剩 1 个词 |
| $p=1$ | — | $V^{(p)}=$ 全体词表，top-p 等于没截断 |

---

## 4. 作业 Deliverable 要你实现什么？

写一个 **从语言模型解码** 的函数，建议支持：

1. **Prompt 续写**：用户给前缀 $x_{1:t}$，继续采样，直到 `<|endoftext|>`  
2. **max tokens**：最多新生成多少个 token  
3. **temperature**：对 logits 做 $v/\tau$ 再 softmax  
4. **top-p**：按上面 nucleus 截断后再采样（Holtzman et al., 2020）

实现时常见顺序（每一步）：

```text
logits = model(tokens)[ : , -1, : ]   # 只要最后位置，形状 (V,) 或 (1, V)
logits = logits / temperature         # τ
q = softmax(logits)
q = nucleus_filter(q, p=top_p)        # 若 top_p < 1
next_id = sample(q)
append; if next_id == eos or len >= max: break
```

细节坑：

- 解码时序列会变长，注意别超过模型的 `context_length`（超了要截断左侧或报错，作业/adapter 会约定）  
- `temperature`、`top_p` 的边界（0、1）要有明确行为  
- EOS id 来自 tokenizer 的 `<|endoftext|>`，不是魔法字符串直接比

---

## 5. 和「增量解码强度」那节的关系（可选对照）

训练/prefill：一次吃整段 prompt。  
这里的循环：每步只真正「新算」与新 token 相关的部分（高效实现会用 KV cache；作业最小实现可以每步整段重算前向，结果对、更慢）。

讲义这页关心的是 **采样分布怎么改**（温度、top-p），不是 GQA 的访存账；两套知识别焊死成一块。

---

## 6. 一句话收束

- 每步：取最后位置 logits $v$ →（$/\tau$）softmax 得 $q$ →（top-p 截断）→ 采样 → 接到后面。  
- 低 $\tau$ 更贪心；高 $\tau$ 更随机。  
- Top-p：保留累计概率刚满 $p$ 的最小高频词集合，砍掉长尾再归一化。  
- 停在 EOS 或 max tokens。

下一步写代码时，把 §3.2 的排序累加和 §4 的循环翻译成函数即可。
