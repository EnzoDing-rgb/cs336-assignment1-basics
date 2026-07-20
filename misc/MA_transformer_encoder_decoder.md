# Encoder / Decoder / Embedding：原版 Transformer 与作业里的 LM

本文自洽：读完即可对照课上那张「original transformer」图，不必再翻 PPT。

---

## A. 原版是什么（2017 Attention is All You Need，机器翻译）

### A.1 任务

把一种语言的句子变成另一种语言的句子。  
例如：英文 `Hello` → 法文 `Bonjour`。

模型同时处理 **两段文本**：

1. **源语言**（source）：要翻译的原文  
2. **目标语言**（target）：已经写出的译文（训练时用正确答案；推理时用模型已生成的词）

### A.2 整体结构（两根柱子）

```text
源语言 token ids
        │
        ▼
  Input Embedding          ← 源语言：token id → 向量
        │
        ⊕ 正弦余弦位置编码
        │
        ▼
  Encoder 块 × N           ← 只读源语言；Self-Attention + FFN
        │
        │  源句表示（每层可作 K/V）
        │
        │                    目标语言 token ids（shifted right）
        │                              │
        │                              ▼
        │                    Output Embedding   ← 目标语言：token id → 向量
        │                              │         （名字像「输出」，实际是 Decoder 的输入）
        │                              ⊕ 正弦余弦位置编码
        │                              │
        │                              ▼
        └──────────►  Decoder 块 × N
                      ├─ Masked Self-Attention   （只看已生成的译文）
                      ├─ Cross-Attention         （Q 来自 Decoder，K/V 来自 Encoder）
                      └─ FFN
                                   │
                                   ▼
                              Linear（词表投影）
                                   │
                                   ▼
                              Softmax
                                   │
                                   ▼
                         下一个目标语言 token 的概率
```

### A.3 原版里每个名字指什么

| 名字 | 它是什么 | 输入 | 输出 |
|------|----------|------|------|
| **Input Embedding** | 源语言查表层 | 源语言 token id | `d_model` 维向量 |
| **Encoder** | 左侧重复 N 次的块 | 源语言向量序列 | 源句上下文表示，供 Decoder 的 Cross-Attention 使用 |
| **Output Embedding** | 目标语言查表层 | 目标语言 token id | `d_model` 维向量 |
| **Decoder** | 右侧重复 N 次的块 | 目标语言向量 + Encoder 表示 | 用于预测下一目标词的隐藏状态 |
| **Masked Self-Attention** | Decoder 内部注意力 | 当前译文位置 | 每个位置只聚合「已经写出的」译文位置 |
| **Cross-Attention** | Decoder 内部注意力 | Decoder 侧 Q；Encoder 侧 K、V | 每个译文位置聚合源句信息 |
| **Linear + Softmax** | 输出头 | Decoder 顶层隐藏状态 | 词表上的概率分布 |
| **Shifted right** | 训练时的输入构造 | 完整正确译文 | 右移一位后的序列，作为 Decoder 的 token 输入（teacher forcing） |

### A.4 原版数据流（一个时间步在概念上发生的事）

训练英→法时，某一步大致是：

```text
1. 英文整句 → Input Embedding → Encoder × N → 得到源句表示 H_src
2. 法文正确答案右移一位 → Output Embedding → 进入 Decoder
3. Decoder 内：Masked Self-Attn 看已写出的法文；Cross-Attn 读 H_src
4. Linear + Softmax → 预测「下一个法文 token」
5. 与正确答案比 CE loss，反传
```

原版有 **两张（或共享的）embedding 表的两条入口**：源语言一条，目标语言一条。  
原版 Decoder 依赖 **Cross-Attention** 读 Encoder。

---

## B. 现在是什么（Decoder-only LM：GPT 式 / 本作业 TinyStories）

### B.1 任务

在同一段文本上预测下一个 token。  
例如：`Once upon a time` → 下一个词的概率。

模型只处理 **一段文本**（上下文与要续写的内容同一条序列）。

### B.2 整体结构（一根柱子）

与仓库 `cs336_basics/model/transformer.py` 中 `TransformerLM` 一致：

```text
token ids
    │
    ▼
token_embeddings          ← 唯一的 Embedding：token id → 向量
    │
    │  （位置：RoPE 做在注意力里）
    ▼
TransformerBlock × N      ← 整网主体
  Pre-Norm
  Causal Self-Attention   ← 每个位置只看序列中自己及左侧
  SwiGLU FFN
    │
    ▼
Final RMSNorm
    │
    ▼
lm_head (Linear)          ← 隐藏状态 → 词表维
    │
    ▼
logits →（训练 CE / 推理 softmax）→ 下一个 token
```

### B.3 作业模型里每个名字指什么

| 名字（代码 / 课上说法） | 它是什么 | 输入 | 输出 |
|-------------------------|----------|------|------|
| **`token_embeddings`** | 唯一查表层 | 上下文 token id | `d_model` 维向量 |
| **`TransformerBlock` × N** | 重复的主体块 | 隐状态序列 | 更新后的隐状态序列 |
| **Causal Self-Attention** | 块内注意力 | 当前序列 | 每个位置聚合「自己及左边」的信息 |
| **`lm_head`** | 输出线性层 | 顶层隐状态 | 词表维 logits |
| **课上称「Decoder」** | 上述整段主体（Embedding 之上的块堆） | 单段文本表示 | 下一 token 的表示 / logits |

本作业这条路径上：

- 有一张 embedding 表：`token_embeddings`  
- 有因果自注意力，在 `TransformerBlock` 里  
- 有输出头：`lm_head`  

本作业这条路径上的信息流：单段文本 → 嵌入 → N 个块 → 词表 logits。

### B.4 作业数据流

```text
1. 文本 → tokenizer.encode → token ids
2. token ids → token_embeddings → 向量序列
3. 经过 N 个 TransformerBlock（causal self-attn + FFN）
4. RMSNorm → lm_head → logits
5. 训练：与「右移一位的下一 token」算 cross-entropy
6. 推理：对 logits 做采样 / argmax，把新 token 接回序列，重复
```

---

## C. 原版名词 → 作业里落在哪

按「角色」对齐（同一行 = 同一类工作）：

| 角色 | 原版（翻译 Enc–Dec） | 作业（Decoder-only LM） |
|------|----------------------|-------------------------|
| 读「另一段源句」并写成记忆 | Encoder 柱 | （本任务不需要第二段源句） |
| 从源句记忆里取信息 | Cross-Attention | （本任务用上下文自注意力覆盖） |
| 源语言 token → 向量 | Input Embedding | （本任务只有一段文本） |
| 正在写的那串 token → 向量 | Output Embedding | **`token_embeddings`** |
| 带掩码的自注意力续写 | Decoder 里 Masked Self-Attn | **`TransformerBlock` 里 Causal Self-Attn** |
| 隐状态 → 词表 | Linear + Softmax | **`lm_head` + CE/softmax** |
| 课上口头「整个 Decoder」 | 右柱 Decoder | **Embedding 之上的整网主体** |

因此：

- 课上图里的 **Input Embedding / Output Embedding** = 源语言入口 / 目标语言入口（两条进网通道）。  
- 作业里的 **`token_embeddings`** = 那条「正在写的序列」的进网通道，角色与原版 **Output Embedding** 相同。  
- 课上说作业是 **Decoder-only**：整网做的是原版 **右柱那种续写工作**；原版 **左柱 Encoder + Cross-Attention** 在这条语言建模配方里拆掉了，换成「上下文都在同一条序列里用 causal self-attn 看」。

---

## D. 代码里出现 `encode` / `encoder` 时指什么

这些名字与上图 **左柱 Encoder 模块** 是不同对象：

| 代码位置 | 含义 |
|----------|------|
| `Tokenizer.encode` / `encode_iterable` | 字符串 → token id 列表（预处理） |
| `gpt2_bytes_to_unicode()` 存进变量 `encoder` | byte → 显示用字符的字典（BPE 可视化） |
| `TransformerLM` | Embedding + Block×N + LM Head（上节 B 的架构） |

读课上「Decoder-only」时，对照的是 **B 节那张单柱图**；  
读 `encode` 时，对照的是 **把文本变成 id** 这一步。

---

## E. 一张总表（可背）

| 维度 | 原版 Enc–Dec | 作业 Decoder-only |
|------|--------------|-------------------|
| 文本段数 | 2（源 + 目标） | 1（同一上下文） |
| 进网 embedding 入口 | 源端 Input Emb. + 目标端 Output Emb. | 仅 `token_embeddings` |
| 核心块 | Encoder×N + Decoder×N | `TransformerBlock`×N |
| 注意力种类 | Encoder self-attn；Decoder masked self-attn + cross-attn | 仅 causal self-attn |
| 典型任务 | 机器翻译 | 语言建模 / 续写 |

**收束一句：**  
原版 = 「左边读懂原文 + 右边对着原文写译文」；  
作业 = 「只在一条文本上往后写下一个 token」，唯一的词嵌入表承担原版目标端入口的工作，主体块承担原版 Decoder 续写的工作。
