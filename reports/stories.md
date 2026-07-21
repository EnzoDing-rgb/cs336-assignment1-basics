# TinyStories 文本生成（temperature × top-p）

## 0. 实验怎么做的

流程：

1. 把训练好的权重加载进 GPU  
2. 固定同一句 prompt，用 `decode()` 自回归续写  
3. 只改两个解码超参：`temperature T` 和 `top-p p`，比较故事流畅度  

| 项 | 取值 |
|----|------|
| 网格 | `T ∈ {0, 0.5, 1.0, 1.3}` × `p ∈ {0.5, 0.8, 0.95}` → **12 组** |
| 每组次数 | 各生成 **1** 次 |
| seed | 每组开始前重置为 **42** |
| 固定项 | checkpoint、prompt、`max_tokens=300`、seed |
| 自变量 | 仅 `(T, p)` |

`T=0` 时每步取 argmax（greedy），概率质量集中在一个 token 上，因此 `p=0.5/0.8/0.95` 三组全文相同。

---

## 1. 设置明细

| 项 | 取值 |
|----|------|
| checkpoint | `artifacts/checkpoints/tinystories_bs128/20260721_0343/ckpt_iter5000.pt`（`B=128` 最佳 run，final valid ≈ 1.411） |
| 解码实现 | `cs336_basics.decode.decode` |
| 脚本 | `scripts/sweep_stories.py` |
| 原文 dump | [`artifacts/stories/`](../artifacts/stories/)（每组一个 `.txt`，另有 `.json` / `summary.csv`） |
| prompt | `Once upon a time, there was a boy named Enzo.` |
| `T` | `0`, `0.5`, `1.0`, `1.3` |
| `p` | `0.5`, `0.8`, `0.95` |
| max_tokens | 300（遇 `<|endoftext|>` 提前结束） |
| seed | 42（每组重置） |

---

## 2. 十二组结果（主观流畅度 1–10）

打分标准（人工读全文）：  
**1–3** 胡话；**4–5** 能认出是故事但崩坏多；**6–7** 可读、有瑕疵；**8–9** 完整连贯、像 TinyStories；**10** 几乎无槽点（本网格最高给到 8）。

| # | `T` | `p` | 约新 token | 分 /10 | 一句话理由 | 文件 |
|--:|----:|----:|----------:|-------:|------------|------|
| 1 | 0 | 0.5 | 179 | **6** | greedy；有主线，但「鸟用嘴叼鸟」等指代乱 | `T0_p0.5.txt` |
| 2 | 0 | 0.8 | 179 | **6** | 与 #1 全文相同（`T=0` 时改 `p` 仍得同一文） | `T0_p0.8.txt` |
| 3 | 0 | 0.95 | 179 | **6** | 同上 | `T0_p0.95.txt` |
| 4 | 0.5 | 0.5 | 109 | **3** | 短；结尾「狗抓住狗并吃掉」逻辑崩 | `T0.5_p0.5.txt` |
| 5 | 0.5 | 0.8 | 186 | **7** | 公园滑梯可读；有 “bag of bag” 小怪 | `T0.5_p0.8.txt` |
| 6 | **0.5** | **0.95** | **204** | **8** | **最佳：分享玩具，结构完整、寓意清楚** | `T0.5_p0.95.txt` |
| 7 | 1.0 | 0.5 | 198 | **8** | 公园滑梯流畅；略堆「意外事件」 | `T1_p0.5.txt` |
| 8 | 1.0 | 0.8 | 151 | **6** | 大体可读；Jane→Kitty 角色名漂移 | `T1_p0.8.txt` |
| 9 | 1.0 | 0.95 | 157 | **5** | Enzo/Tom 串名，寓意别扭 | `T1_p0.95.txt` |
| 10 | 1.3 | 0.5 | 191 | **4** | 人名崩（Enzo→Ens/Rado/Radzi） | `T1.3_p0.5.txt` |
| 11 | 1.3 | 0.8 | 86 | **2** | 短且近胡话 | `T1.3_p0.8.txt` |
| 12 | 1.3 | 0.95 | 248 | **1** | 最长，假词/乱码居多 | `T1.3_p0.95.txt` |

按 `T` 平均分：`T=0` → 6；`T=0.5` → 6.0（#4 拉低；#5+#6 均分 7.5）；`T=1.0` → 6.3；`T=1.3` → 2.3。

---

## 3. 主交付样本（推荐：#6）

**设定：** `T=0.5`，`p=0.95`，seed=42 → `artifacts/stories/T0.5_p0.95.txt`（8/10）

```
Once upon a time, there was a boy named Enzo. Enzo was very excited because he was going to a new school. He wanted to learn new things and make new friends.
One day, Enzo met a girl named Lily. Lily was very nice and liked to play with Enzo. They had a lot of fun together. But Lily was a bit ignorant. She did not know that Enzo was not good at playing games.
One day, Enzo and Lily had a quarrel. They both wanted to play with the same toy. Enzo said, "I want to play with the toy first!" Lily said, "No, I want to play with the toy first!" They did not know what to do.
Enzo had an idea. He said, "Let's share the toy and play together." Lily agreed. They played with the toy and had a lot of fun. In the end, they both learned that sharing is good and makes everyone happy.
<|endoftext|>
```

**对照（#7，同为 8/10）：** `T=1.0`, `p=0.5`，公园滑梯故事，语气更口语。全文见 `artifacts/stories/T1_p0.5.txt`。

---

## 4. 影响质量的因素（≥2）

### 4.1 Temperature `T`

采样前把 logits 除以 `T` 再 softmax：

- **`T < 1`（如 0.5）：** 分布更尖 → 更常抽到高置信词 → 句式更稳、更像训练集里的儿童故事。  
- **`T = 1`：** 按模型原始分布抽样。  
- **`T > 1`（如 1.3）：** 分布被压平，长尾词概率升高 → 更容易抽到罕见拼写、错名、假词、不通顺搭配。本网格 `T=1.3` 三组得分 4 / 2 / 1，而 `T∈{0.5,1.0}` 最好到 8；对照 `T1.3_p0.95.txt` 可直接看见胡话。  
- **`T = 0`：** 每步 argmax（greedy）；#1–#3 全文相同。

### 4.2 Top-p（nucleus）

按概率从高到低累加，只保留「刚好凑满概率质量 `p`」的那一小撮词，再在核内重归一化抽样：

- **小 `p`（0.5）：** 候选极少 → 更保守；尖核若已锁进坏剧情，会一路错下去（#4）。  
- **大 `p`（0.95）：** 候选更多 → 中等 `T` 时故事往往更丰满（#6）；高 `T` 时胡话空间也更大（#12）。  
- **`T=0`：** 分布已是 one-hot，改 `p` 仍得到同一串 token。

（prompt、checkpoint 也会定调；本实验固定二者，只扫解码超参。）

---

## 5. 一句话

加载 `B=128` 最佳权重，固定 Enzo prompt，对 12 个 `(T,p)` 各生成 1 次：中等温度（约 0.5–1）+ 较宽 nucleus（约 0.8–0.95）最好读；`T>1` 因分布过平、长尾被抬高而明显掉分。推荐样例：`T=0.5`, `p=0.95`（8/10）。
