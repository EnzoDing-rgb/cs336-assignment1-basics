# Encoder / Decoder / Embedding：原版 Transformer vs 你们的 LM

> 目标：一张图分清「2017 机器翻译 Transformer」和「作业里的 decoder-only LM」。  
> 顺便解释：为什么课上说「现代 LM 没有 Encoder」，而你代码里又能搜到 `encode` / `encoder`。

---

## 0. 先消掉命名坑

| 你听到 / 搜到的词 | 实际指什么 | 是不是架构图上的 Encoder？ |
|-------------------|------------|---------------------------|
| **Encoder（架构）** | 原版左柱：只读源语言、不生成 | ✅ 本文说的这个 |
| **Decoder（架构）** | 原版右柱 / 或整个 GPT 主体 | 见下文 |
| `tokenizer.encode()` | 文本 → token id | ❌ 只是「编码成整数」 |
| `gpt2_bytes_to_unicode` 里的 `encoder` | byte→可见字符的映射表 | ❌ 和 Transformer 无关 |
| 你们 `TransformerLM` | Embedding + N×Block + LM Head | ❌ **没有**独立 Encoder 模块 |

你们仓库里的语言模型组装是：

```text
token ids → token_embeddings → N × TransformerBlock → RMSNorm → lm_head → logits
```

见 `cs336_basics/model/transformer.py`：`TransformerLM` 注释写的就是  
「Embedding → N×Block → Final RMSNorm → LM Head」——**整网都是自回归侧，没有第二根「Encoder 柱」。**

所以：**「现代主流没有 Encoder」= 没有「读源句、再给 Decoder 做 cross-attn」的那半边**；  
不是说代码里不能出现单词 `encode`。

---

## 1. 原版（2017）：Encoder–Decoder（机器翻译）

### 1.1 在干什么

两段不同文本：

```text
源语言（英文） ──Encoder──► 记忆（每层表示）
已生成译文（法文）──Decoder──► 下一个法文 token（还看 Encoder 记忆）
```

### 1.2 架构图（对应课上那张）

```text
                    ┌─────────────────────────────────────┐
                    │         OUTPUT PROBABILITIES         │
                    └──────────────────▲──────────────────┘
                                       │ Softmax
                    ┌──────────────────┴──────────────────┐
                    │              Linear (LM head)         │
                    └──────────────────▲──────────────────┘
                                       │
              ┌────────────────────────┴────────────────────────┐
              │              Decoder × N                         │
              │  ┌────────────────────────────────────────────┐ │
              │  │ Feed Forward → Add & Norm                    │ │
              │  │ Cross-Attn(Q=dec, K/V=enc) → Add & Norm      │ │  ← 看左边 Encoder
              │  │ Masked Self-Attn → Add & Norm                │ │  ← 只看已生成译文
              │  └────────────────────────────────────────────┘ │
              └────────────────────────▲────────────────────────┘
                                       │
                         Positional Encoding ⊕
                                       │
                    ┌──────────────────┴──────────────────┐
                    │     Output Embedding（名字很坑）       │
                    │  其实是：目标语言 token → 向量          │
                    └──────────────────▲──────────────────┘
                                       │
                    Outputs (shifted right)  法文 teacher forcing


   ┌────────────────────────┐
   │      Encoder × N       │
   │  FFN → Add & Norm      │
   │  Self-Attn → Add & Norm│   ← 只看源语言，不生成
   └────────────▲───────────┘
                │
    Positional Encoding ⊕
                │
   ┌────────────┴───────────┐
   │    Input Embedding      │
   │  源语言 token → 向量     │
   └────────────▲───────────┘
                │
             Inputs  英文
```

### 1.3 模块是什么（对照表）

| 名字 | 是什么 | 吃什么 | 产出什么 |
|------|--------|--------|----------|
| **Encoder** | 左柱堆叠块 | 源语言向量序列 | 源句的上下文表示（给 Decoder 的 K/V） |
| **Decoder** | 右柱堆叠块 | 已生成目标词 + Encoder 输出 | 下一步目标词的隐藏状态 |
| **Input Embedding** | 源端查表 | 源语言 token id | `d_model` 向量 |
| **Output Embedding** | **目标端查表**（易误解） | 目标语言 token id（训练：右移正确答案；推理：已生成词） | `d_model` 向量 |
| **Shifted right** | Teacher forcing | 完整正确答案右移 1 位 | Decoder 的输入序列 |

要点：

- **Output Embedding ≠ 给最终 Softmax 输出再 embed 一次。**  
  它是 Decoder 的**输入**嵌入。
- Softmax 前面的 **Linear** 才是「隐藏状态 → 词表 logits」。

---

## 2. 现代主流：Decoder-Only LM（GPT / 你们作业）

### 2.1 在干什么

只有一段文本，预测下一个 token：

```text
"Once upon a time" → 模型 → 下一个 token 的概率
```

没有「另一门语言的 Encoder」。

### 2.2 架构图（和你们代码对齐）

```text
                 logits (B, seq, vocab)
                          ▲
                     LM Head (Linear)
                          ▲
                   Final RMSNorm
                          ▲
              ┌───────────┴───────────┐
              │  TransformerBlock × N  │   ← 课上说的「整个就是 Decoder」
              │  Pre-Norm + Self-Attn  │      （带 causal mask）
              │  + SwiGLU FFN          │
              │  没有 Cross-Attn       │
              └───────────▲───────────┘
                          │
                 (+ RoPE 在注意力里)
                          │
              ┌───────────┴───────────┐
              │  token_embeddings      │  ← 唯一的 Embedding
              │  (= 原版 Output Emb.   │     功能角色)
              └───────────▲───────────┘
                          │
                    token ids
```

对应代码（概念路径）：

```text
TransformerLM.forward(token_ids)
  → self.token_embeddings(token_ids)     # 唯一 embedding
  → for block in self.layers: block(x) # N × TransformerBlock
  → self.ln_final(x)
  → self.lm_head(x)                    # → vocab logits
```

### 2.3 和原版名词怎么对应

| 原版（翻译） | 你们 LM | 说明 |
|--------------|---------|------|
| Encoder | **无** | 没有源句分支，也没有 cross-attn |
| Decoder | **整网主体**（`TransformerBlock`×N） | 只有 causal self-attn |
| Input Embedding | **无**（没有单独源语言表） | — |
| Output Embedding | **`token_embeddings`** | 唯一入口；角色对齐原版「Decoder 端输入嵌入」 |
| Linear + Softmax | **`lm_head` +（训练时 CE / 推理时 softmax）** | 输出概率，不是 embedding |

---

## 3. 为什么说「你们没有 Encoder」却又觉得奇怪？

常见三种「假 Encoder」：

```text
1) 架构 Encoder     → 作业 LM：没有
2) tokenizer.encode → 文本变 id：有，但这是预处理，不是网络左柱
3) byte encoder 字典 → BPE 显示用：有，更不是网络左柱
```

**课上那句「Decoder-only 没有 Encoder」**  
= 没有上图左柱 + 没有 Decoder 里的 **Cross-Attention**。  

你们的 `TransformerBlock` 里是 **Masked / causal Multi-Head Self-Attention**，  
信息流只有「当前序列看自己的过去」，没有「去看另一段 Encoder 记忆」。

---

## 4. 一张总对照（背这个就够）

```text
┌──────────────────┬─────────────────────────┬──────────────────────────┐
│                  │ 2017 Enc–Dec（翻译）      │ Decoder-only LM（作业）   │
├──────────────────┼─────────────────────────┼──────────────────────────┤
│ 文本段数          │ 2（源 + 目标）            │ 1（同一段上下文）         │
│ Encoder 柱        │ 有                       │ 无                        │
│ Cross-Attn        │ 有                       │ 无                        │
│ Embedding 表      │ 通常 2 个（或共享）        │ 1 个 token_embeddings     │
│ 生成方式          │ Decoder 自回归写译文      │ 整网自回归写续写           │
└──────────────────┴─────────────────────────┴──────────────────────────┘
```

**一句话：**  
原版图上的 Input / Output Embedding = **两种语言各自进网的入口**；  
你们只有一种「语言建模输入」，所以只剩一个 embedding——它对应原版右边那个被叫臭了的 Output Embedding，而不是少做了半个模型。
