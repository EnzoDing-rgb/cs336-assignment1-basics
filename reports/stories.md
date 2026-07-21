# TinyStories 文本生成（temperature × top-p）

## 0. 实验怎么做的

我们先把训练好的模型权重加载到 GPU，再用同一句英文开头，调用 `decode()` 往后续写。整次实验只改两个解码超参——温度 `T` 和 nucleus 阈值 `p`——用来比较续写故事好不好读。

| 项 | 取值 |
|----|------|
| 网格 | `T` 取 `0 / 0.5 / 1.0 / 1.3`，`p` 取 `0.5 / 0.8 / 0.95`，一共 **12** 组 |
| 每组次数 | 每个 `(T, p)` 组合只生成 **一次** |
| seed | 每一组开始前都把随机种子重置为 **42**，方便复现 |
| 固定项 | checkpoint、prompt、`max_tokens=300`、seed 全程不变 |
| 自变量 | 只有 `(T, p)` 在变 |

当 `T=0` 时，每一步都直接选概率最大的那个词（greedy）。这时概率已经全部压在一个 token 上，再改 `p` 也不会换候选，所以 `p=0.5`、`0.8`、`0.95` 三组输出的全文完全一样。

---

## 1. 设置明细

| 项 | 取值 |
|----|------|
| checkpoint | `artifacts/checkpoints/tinystories_bs128/20260721_0343/ckpt_iter5000.pt`（batch-size 实验里 `B=128` 的最佳 run，final valid ≈ 1.411） |
| 解码实现 | `cs336_basics.decode.decode` |
| 脚本 | `scripts/sweep_stories.py` |
| 原文 dump | [`artifacts/stories/`](../artifacts/stories/)（每组一个 `.txt`，另有 `.json` 和 `summary.csv`） |
| prompt | `Once upon a time, there was a boy named Enzo.` |
| `T` | `0`, `0.5`, `1.0`, `1.3` |
| `p` | `0.5`, `0.8`, `0.95` |
| max_tokens | 最多新写 300 个 token；若先写出 `<|endoftext|>` 则提前结束 |
| seed | 42（每组重置） |

---

## 2. 十二组结果（主观流畅度 1–10）

分数是人工通读全文后打的，标准如下：

- **1–3 分：** 基本读不成故事，胡话或假词很多  
- **4–5 分：** 能看出在讲故事，但人物、情节经常对不上  
- **6–7 分：** 大体可读，偶有别扭或小错误  
- **8–9 分：** 结构完整、读起来像 TinyStories 里的儿童故事  
- **10 分：** 几乎挑不出问题（本网格最高只打到 8 分）

| # | `T` | `p` | 约新 token | 分 /10 | 打分理由 | 文件 |
|--:|----:|----:|----------:|-------:|----------|------|
| 1 | 0 | 0.5 | 179 | **6** | 有「男孩爬树、小鸟帮忙」的主线，句子也通顺，但后半段出现「鸟用嘴叼起鸟」这类指代错乱 | `T0_p0.5.txt` |
| 2 | 0 | 0.8 | 179 | **6** | 与第 1 组全文相同（`T=0` 时改 `p` 不会改变输出） | `T0_p0.8.txt` |
| 3 | 0 | 0.95 | 179 | **6** | 与第 1 组全文相同 | `T0_p0.95.txt` |
| 4 | 0.5 | 0.5 | 109 | **3** | 篇幅偏短，结尾写成「狗抓住狗并把它吃掉」，情节逻辑已经崩掉 | `T0.5_p0.5.txt` |
| 5 | 0.5 | 0.8 | 186 | **7** | 写男孩去公园玩滑梯、朋友帮忙，整体能读；中间有 “bag of bag” 这种重复用词 | `T0.5_p0.8.txt` |
| 6 | **0.5** | **0.95** | **204** | **8** | **本网格最好：Enzo 和 Lily 争玩具再学会分享，开端、冲突、收束都清楚** | `T0.5_p0.95.txt` |
| 7 | 1.0 | 0.5 | 198 | **8** | 同样是去公园玩滑梯的完整小故事，语气更口语；后半段「风吹倒滑梯又出现新滑梯」略显堆砌 | `T1_p0.5.txt` |
| 8 | 1.0 | 0.8 | 151 | **6** | 大意能读懂，但角色名字中途从 Jane 漂成 Kitty，前后对不上 | `T1_p0.8.txt` |
| 9 | 1.0 | 0.95 | 157 | **5** | Enzo 和 Tom 两个名字缠在一起，结尾的寓意也说得别扭 | `T1_p0.95.txt` |
| 10 | 1.3 | 0.5 | 191 | **4** | 还能看出在找贝壳，但主人公名字从 Enzo 漂到 Ens、Rado、Radzi | `T1.3_p0.5.txt` |
| 11 | 1.3 | 0.8 | 86 | **2** | 很短，句子已经接近胡话，几乎读不出完整情节 | `T1.3_p0.8.txt` |
| 12 | 1.3 | 0.95 | 248 | **1** | 篇幅最长，但假词、乱码和断裂句子很多，基本没法当故事读 | `T1.3_p0.95.txt` |

按温度汇总平均分：`T=0` 为 6 分；`T=0.5` 为 6.0 分（被第 4 组拉低，第 5、6 组平均 7.5）；`T=1.0` 为 6.3 分；`T=1.3` 为 2.3 分。

---

## 3. 主交付样本（推荐第 6 组）

解码设定：`T=0.5`，`p=0.95`，seed=42。原文在 `artifacts/stories/T0.5_p0.95.txt`，主观分 8/10。

```
Once upon a time, there was a boy named Enzo. Enzo was very excited because he was going to a new school. He wanted to learn new things and make new friends.
One day, Enzo met a girl named Lily. Lily was very nice and liked to play with Enzo. They had a lot of fun together. But Lily was a bit ignorant. She did not know that Enzo was not good at playing games.
One day, Enzo and Lily had a quarrel. They both wanted to play with the same toy. Enzo said, "I want to play with the toy first!" Lily said, "No, I want to play with the toy first!" They did not know what to do.
Enzo had an idea. He said, "Let's share the toy and play together." Lily agreed. They played with the toy and had a lot of fun. In the end, they both learned that sharing is good and makes everyone happy.
<|endoftext|>
```

第 7 组（`T=1.0`, `p=0.5`）同样打到 8 分，讲的是 Enzo 和妈妈去公园玩滑梯，语气更口语，全文见 `artifacts/stories/T1_p0.5.txt`。

---

## 4. 影响质量的两个因素

### 4.1 Temperature `T`

采样前会把 logits 除以 `T`，再做 softmax：

- **`T < 1`（例如 0.5）：** 概率分布更尖，模型更常选出自己最有把握的词，句子往往更稳，也更像训练集里的儿童故事口吻。  
- **`T = 1`：** 直接按模型算出的原始分布抽样。  
- **`T > 1`（例如 1.3）：** 概率分布被压得更平，原本很低的长尾词也被抬高，于是更容易抽到罕见拼写、错名、假词和不通顺的搭配。本网格里 `T=1.3` 三组只拿到 4、2、1 分，而 `T` 在 `0.5` 或 `1.0` 时最好能到 8 分；打开 `T1.3_p0.95.txt` 就能直接看到胡话。  
- **`T = 0`：** 每一步都取 argmax，第 1 到第 3 组因此全文相同。

### 4.2 Top-p（nucleus）

先把词按概率从高到低排好，从前面积累，只留下「刚好凑满概率质量 `p`」的那一小撮，再在这个核里重新归一化后抽样：

- **较小的 `p`（例如 0.5）：** 候选词很少，输出更保守；如果这小撮里已经锁进了坏剧情，后面也会一路顺着错下去（见第 4 组）。  
- **较大的 `p`（例如 0.95）：** 候选更多。温度适中时，故事往往更丰满（见第 6 组）；温度已经很高时，胡话也有更大空间（见第 12 组）。  
- **`T=0`：** 分布已经是 one-hot，改 `p` 仍然得到同一串 token。

prompt 和 checkpoint 也会影响文风；本实验把这两项固定住，只扫解码超参。

---

## 5. 小结

我们加载 `B=128` 的最佳权重，固定 Enzo 这句开头，对 12 个不同的 `(T, p)` 各生成一次。读感最好的区间大约是中等温度（`T` 在 0.5 到 1）配合较宽的 nucleus（`p` 在 0.8 到 0.95）。`T>1` 会把分布压平、抬高长尾，流畅度明显下降。交作业时推荐用第 6 组：`T=0.5`，`p=0.95`（8/10）。
