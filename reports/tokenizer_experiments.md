# Tokenizer 实验报告（单一入口）

覆盖作业：

- **Problem (train_bpe_expts_owt)**：OWT 上训练 BPE、最长 token、与 TinyStories 对比  
- **Problem (tokenizer_experiments)**：(a)–(d) 压缩率、错配 tokenizer、吞吐、encode 落盘  

仓库根目录：

```text
/root/.dev/ml-sys/cs336/assignment1-basics
```

---

## 0. 一键复现（先跑这个）

复制**一整行**到终端（`&&` 连接，无 multiline）：

```bash
cd /root/.dev/ml-sys/cs336/assignment1-basics && uv run python scripts/tokenizer_experiments.py
```

终端会打印 (a)(b)(c)(d) 的 JSON + 摘要。本文件 **附录 C** 归档了 2026-07-20 一次完整原始输出（原 `tokenizer_experiments_last_run.txt`，已并入此处）。

可选：只看 BPE 产物是否在（不跑 Python）：

```bash
ls -lh /root/.dev/ml-sys/cs336/assignment1-basics/artifacts/owt_bpe/ && ls -lh /root/.dev/ml-sys/cs336/assignment1-basics/artifacts/tinystories_bpe/ && ls -lh /root/.dev/ml-sys/cs336/assignment1-basics/artifacts/owt_tokens/ && ls -lh /root/.dev/ml-sys/cs336/assignment1-basics/artifacts/tinystories_tokens/
```

---

# Part I — train_bpe_expts_owt（BPE 训练产物）

## I.0 怎么训练 BPE（重跑命令）

模块入口：`python -m cs336_basics.tokenization.train_bpe`（读 `data/*.txt`，写到 `artifacts/*_bpe/`）。

TinyStories（默认 vocab 10k，`--workers` 默认 8）：

```bash
cd /root/.dev/ml-sys/cs336/assignment1-basics && uv run python -m cs336_basics.tokenization.train_bpe --dataset tinystories
```

OWT（32k 词表；建议 tee 日志，墙钟很长）：

```bash
cd /root/.dev/ml-sys/cs336/assignment1-basics && uv run python -m cs336_basics.tokenization.train_bpe --dataset owt --vocab-size 32000 --workers 8 2>&1 | tee /root/.dev/ml-sys/cs336/assignment1-basics/artifacts/owt_bpe_train.log
```

产物：`artifacts/<dataset>_bpe/{vocab.json, merges.txt, profile_report.json}`。

---

## I.1 训练做完了吗？

| 数据集 | vocab | 产物目录 |
|--------|-------|----------|
| OWT | 32,000 | `/root/.dev/ml-sys/cs336/assignment1-basics/artifacts/owt_bpe/` |
| TinyStories | 10,000 | `/root/.dev/ml-sys/cs336/assignment1-basics/artifacts/tinystories_bpe/` |

每个目录三个文件：`vocab.json`、`merges.txt`、`profile_report.json`。

OWT 训练日志（很长，看末尾汇总）：

`/root/.dev/ml-sys/cs336/assignment1-basics/artifacts/owt_bpe_train.log`

---

## I.2 在 Cursor 里按顺序看什么？

### A. `profile_report.json`（成绩单）

| 打开 | 看什么 |
|------|--------|
| `/root/.dev/ml-sys/cs336/assignment1-basics/artifacts/owt_bpe/profile_report.json` | `longest_token.id=25822`, `length_bytes=64` |
| `/root/.dev/ml-sys/cs336/assignment1-basics/artifacts/tinystories_bpe/profile_report.json` | `longest_token.id=7160`, `display=Ġaccomplishment` |

### B. `vocab.json`（最长 token 真身）

在 Cursor 里 Ctrl+F：

- OWT：打开 `.../artifacts/owt_bpe/vocab.json`，搜 `"25822"`  
- TinyStories：打开 `.../artifacts/tinystories_bpe/vocab.json`，搜 `"7160"`

### C. `merges.txt`（merge 顺序）

打开 `.../artifacts/owt_bpe/merges.txt`，前几行类似：

```text
Ġ t
Ġ a
h e
```

OWT 共 **31,743** 行；TinyStories **9,743** 行。

### D. 命令验证（单行）

```bash
cat /root/.dev/ml-sys/cs336/assignment1-basics/artifacts/owt_bpe/profile_report.json && echo "---" && cat /root/.dev/ml-sys/cs336/assignment1-basics/artifacts/tinystories_bpe/profile_report.json
```

---

## I.3 train_bpe_expts 交作业用语

**(a) 最长 token：** id **25822**，**64 bytes**，`vocab.json` 里为一串重复 `ÃĥÃĤ…`（底层为 `b'\xc3\x83\xc3\x82'`×16）。**合理**：byte-level BPE 按共现合并，OWT 网页噪声里高频 mojibake 会被 merge 成长 token。

**(b) TS vs OWT：** TinyStories 最长 token 可读（`Ġaccomplishment`，15 bytes）；OWT 词表更大、语料更噪，最长 token 为 64-byte 乱码样 bytes；OWT 适合开放域，TinyStories 更贴叙事英语。

资源：OWT 峰值 RAM ~31.7 GiB（<100 GiB ✓）；墙钟 ~38.2 h（超过 handout 12 h 参考，但已跑完）。

---

# Part II — tokenizer_experiments (a)–(d)

> **本文档定位**：学习笔记，不是「交作业一句话」。每一问都写清：**在算什么 → 为什么这样算 → 我跑出来的数 → 你怎么自己复现**。  
> **数据来源**：2026-07-20 在本机重新跑了 §0 一键命令；`.npy` 落盘文件存在（OWT train 约 5.1 GiB，2026-07-19 生成）。下文数字与 **附录 C** 原始输出一致。

---

## II.0 先搞懂三个词：byte、token、bytes/token

### byte 是什么？

这里的 **byte** 指 **UTF-8 编码后的原始字节数**，不是「字符个数」。

- 磁盘上的 `data/*.txt` 是 **UTF-8 文本**。
- Python 里用 `len(doc.encode("utf-8"))` 计数：把字符串按 UTF-8 规则变成 `bytes`，再数有几个字节。
- 英文 ASCII 通常 1 字符 = 1 byte；中文、emoji 往往 1 字符 = 3–4 bytes。

**例子**（TinyStories 第 1 篇文档开头）：

```text
字符数 len(doc)     = 732
UTF-8 字节数        = 740   ← 报告里说的「字节」都是这个
```

多出来的 8 bytes 来自换行、标点等在 UTF-8 里占的字节（和「肉眼看到的字数」不必相等）。

### token 是什么？

**token** 不是物理单位，而是 **tokenizer 切分文本后得到的整数 id**。

流程：

```text
原始文本 (str)
  → Tokenizer.encode(text)
  → [2001, 3384, 696, ...]   # 每个整数是一个 token 的 id
  → 语言模型吃的是这些 id，不是 raw bytes
```

- 一个 token 对应词表里的一行（可能是 `" the"`、`"ing"`、一个汉字、甚至乱码 bytes 拼成的片段）。
- **token 个数** = `len(tokenizer.encode(text))`，即 id 列表的长度。
- 同一篇文本，用不同 tokenizer（不同 BPE 词表）会得到 **不同的 token 数**——这正是 (b) 要看的。

### bytes/token 是什么？为什么叫「压缩率」？

作业里的指标：

```text
bytes/token = (样本 UTF-8 总字节数) / (encode 得到的 token 总数)
```

**含义**：平均每个 token「代表」多少字节的原文。

- **越大** → 每个 token 承载更多信息 → 同样长的文本需要 **更少的 token** → 对 LM 更省（序列更短）。
- **越小** → 切得更碎 → 同样文本要 **更多 token**。

它 **不是**「每个 token 在磁盘上占几个 byte」——磁盘上存的是 **token id**（见 §II.4 的 uint16）；这里的 byte 指的是 **被编码的原文** 的字节数。

**手算一例（TinyStories 10 篇合计）**：

```text
总 UTF-8 字节  = 7,435
总 token 数    = 1,808
bytes/token    = 7435 / 1808 = 4.1123…
```

### 样本怎么取？

`scripts/tokenizer_experiments.py` 与 handout 一致：

1. 打开 `data/TinyStoriesV2-GPT4-train.txt` 或 `data/owt_train.txt`（**从头流式读**，不把整个 OWT 载入内存）。
2. 用特殊串 `<|endoftext|>` 当 **文档分隔符**（和训练 BPE 时一致）。
3. 取 **前 10 篇非空文档** 作为样本。
4. (a) 用 **各自域上训练的 tokenizer**；(b) 固定 OWT 10 篇，换 tokenizer 对比。

---

## II.1 问题 (a)：各自 tokenizer 的压缩率

### 题目在问什么？

在 **TinyStories 训练集** 上取 10 篇文档，用 **TinyStories BPE (10k)** 编码，算 bytes/token。  
在 **OWT 训练集** 上取 10 篇文档，用 **OWT BPE (32k)** 编码，算 bytes/token。  
比较：**匹配域的 tokenizer** 能把文本压成多长的 token 序列。

### 计算步骤（逐步）

对每一个数据集 `D` ∈ {TinyStories, OWT}：

1. `docs = sample_documents(data/D_train.txt, n=10)`
2. `total_bytes = sum(len(doc.encode("utf-8")) for doc in docs)`
3. `ids = []`；对每个 `doc`：`ids.extend(tokenizer_D.encode(doc))`
4. `num_tokens = len(ids)`
5. `bytes_per_token = total_bytes / num_tokens`

### 本次运行结果（2026-07-20）

| 样本 | 用的 Tokenizer | 总 UTF-8 字节 | 总 token 数 | **bytes/token** |
|------|----------------|---------------|-------------|-----------------|
| TinyStories 前 10 篇 | `tinystories_bpe` (10k) | 7,435 | 1,808 | **4.112** |
| OWT 前 10 篇 | `owt_bpe` (32k) | 31,487 | 6,712 | **4.691** |

**逐篇明细（TinyStories）**——可见每篇大约 3.9–4.4 bytes/token，合计 4.112：

| 篇 | UTF-8 bytes | tokens | bytes/token |
|----|-------------|--------|-------------|
| 0 | 740 | 175 | 4.229 |
| 1 | 663 | 164 | 4.043 |
| … | … | … | … |
| 9 | 872 | 207 | 4.213 |
| **合计** | **7,435** | **1,808** | **4.112** |

**逐篇明细（OWT）**——合计 4.691：

| 篇 | UTF-8 bytes | tokens | bytes/token |
|----|-------------|--------|-------------|
| 0 | 4,598 | 1,031 | 4.460 |
| 1 | 2,449 | 494 | 4.957 |
| … | … | … | … |
| 9 | 2,349 | 476 | 4.935 |
| **合计** | **31,487** | **6,712** | **4.691** |

### 怎么理解？

- OWT 样本 **bytes/token 更高**（4.69 > 4.11）：在各自训练域上，32k 词表在 OWT 文体上平均每个 token 覆盖更多字节（序列更短）。
- 注意：这是 **10 篇小样本**，不是全库统计；OWT 第 8 篇特别长（6654 bytes），会拉高总 token 数，但 ratio 仍稳定在大约 4.5–5.0。

### 你自己怎么复现？

**一键（推荐）**：§0 命令，看输出里 `"a": { ... }` 和末尾两行 `(a)`。

**手动拆开学**（在 repo 根目录）：

```bash
cd /root/.dev/ml-sys/cs336/assignment1-basics && uv run python -c "
from pathlib import Path
from scripts.tokenizer_experiments import load_tok, sample_documents, compression_ratio_bytes_per_token
ROOT = Path('.')
ts_docs = sample_documents(ROOT/'data'/'TinyStoriesV2-GPT4-train.txt', 10)
owt_docs = sample_documents(ROOT/'data'/'owt_train.txt', 10)
r, b, t = compression_ratio_bytes_per_token(ts_docs, load_tok('tinystories_bpe'))
print('TS:', b, 'bytes', t, 'tokens', 'ratio', round(r, 4))
r, b, t = compression_ratio_bytes_per_token(owt_docs, load_tok('owt_bpe'))
print('OWT:', b, 'bytes', t, 'tokens', 'ratio', round(r, 4))
"
```

**看单篇原文 + token 数**：

```bash
cd /root/.dev/ml-sys/cs336/assignment1-basics && uv run python -c "
from pathlib import Path
from scripts.tokenizer_experiments import load_tok, sample_documents
doc = sample_documents(Path('data/TinyStoriesV2-GPT4-train.txt'), 1)[0]
tok = load_tok('tinystories_bpe')
print('utf8 bytes:', len(doc.encode('utf-8')))
print('tokens:', len(tok.encode(doc)))
print('preview:', repr(doc[:200]))
"
```

### 交作业可写（1–2 句）

在各自训练域的前 10 篇文档上，TinyStories 10k tokenizer 约 **4.11 bytes/token**，OWT 32k tokenizer 约 **4.69 bytes/token**；匹配域的大词表在 OWT 文本上略更省 token。

---

## II.2 问题 (b)：用 TinyStories tokenizer 编 OWT 样本

### 题目在问什么？

**固定同一份 OWT 文本**（仍是前 10 篇，31,487 UTF-8 bytes），换用 **TinyStories 10k tokenizer** 编码。  
和 (a) 里「OWT 文本 + OWT tokenizer」对比：**错配词表** 会让 token 变多还是变少？

### 关键不变量

| 量 | 值 | 说明 |
|----|-----|------|
| 文本 | OWT train 前 10 篇 | 与 (a) 完全相同 |
| UTF-8 字节 | **31,487** | 换 tokenizer **不会改变**原文字节数 |
| 变的量 | token 数、bytes/token | 由词表 / merge 规则决定 |

### 计算步骤

1. `owt_docs` = 同上 10 篇 OWT 文档  
2. `total_bytes = 31487`（不变）  
3. `n_matched = len(encode(owt_docs, owt_bpe))` → **6,712**  
4. `n_mismatch = len(encode(owt_docs, tinystories_bpe))` → **9,873**  
5. `bytes/token_matched = 31487 / 6712 = 4.691`  
6. `bytes/token_mismatch = 31487 / 9873 = 3.189`  
7. `token 倍数 = 9873 / 6712 = 1.47×`

### 本次运行结果

| Tokenizer | token 数 | bytes/token |
|-----------|----------|-------------|
| OWT 32k（匹配） | 6,712 | **4.691** |
| TS 10k（错配） | 9,873 | **3.189** |

错配时 token 数约 **1.47×**；bytes/token 从 4.69 降到 3.19。

**前 5 篇逐篇对比**（同一篇 doc，字节相同，token 数不同）：

| 篇 | UTF-8 bytes | OWT tok | TS tok | TS/OWT |
|----|-------------|---------|--------|--------|
| 0 | 4,598 | 1,031 | 1,361 | 1.32× |
| 1 | 2,449 | 494 | 757 | 1.53× |
| 2 | 2,027 | 437 | 645 | 1.48× |
| 3 | 3,174 | 703 | 879 | 1.25× |
| 4 | 4,674 | 928 | 1,439 | 1.55× |

### 微观例子：同一段原文，两种切法

OWT 第 0 篇 **前 80 个字符**（80 UTF-8 bytes）：

```text
What wouldn't you do to save someone you love?

When They Come Calling is a mode
```

| Tokenizer | token 数 | 前几个 id 解码 |
|-----------|----------|----------------|
| OWT 32k | **21** | `What` ` wouldn` `'t` ` you` ` do` ` to` … |
| TS 10k | **23** | `What` ` wouldn` `'t` ` you` ` do` ` to` …（更碎，id 不同） |

TS 词表没在 OWT 网页/新闻体上训练，缺少 OWT 常见 merge → 同样字节数要更多 token 才能拼回去。`decode` 仍能得到原文，但 **LM 训练/推理** 要处理更长序列，更慢、更费显存。

### 你自己怎么复现？

```bash
cd /root/.dev/ml-sys/cs336/assignment1-basics && uv run python -c "
from pathlib import Path
from scripts.tokenizer_experiments import load_tok, sample_documents, compression_ratio_bytes_per_token
owt_docs = sample_documents(Path('data/owt_train.txt'), 10)
r1, b, t1 = compression_ratio_bytes_per_token(owt_docs, load_tok('owt_bpe'))
r2, _, t2 = compression_ratio_bytes_per_token(owt_docs, load_tok('tinystories_bpe'))
print('bytes', b, 'owt_tok', t1, 'ts_tok', t2, 'ratio', round(t2/t1, 3))
print('bytes/token matched', round(r1,3), 'mismatch', round(r2,3))
"
```

### 交作业可写（1–2 句）

用 TinyStories tokenizer 编 OWT 前 10 篇会得到约 **1.47× 更多 token**（**3.19 vs 4.69 bytes/token**），因 10k 词表缺少 OWT 高频子词 merge，切分更碎。

---

## II.3 问题 (c)：吞吐与 Pile 825 GiB 粗算

### 题目在问什么？

1. 本机 `tokenizer.encode()` **有多快**？（bytes/秒，不是 tokens/秒）  
2. 若要把 **The Pile 全量 825 GiB** 文本 tokenize 一遍，**大概多少小时**？（粗算）

### 测量方法（`scripts/tokenizer_experiments.py`）

1. 从 TinyStories train **读前 2,000,000 个字符** 作为固定 bench 文本（约 **1.91 MiB** UTF-8）。  
2. 对每个 tokenizer：先 **warmup** 一次 `encode`；再循环 `encode` 直到累计 **≥ 2 秒** wall time。  
3. `throughput = (bench_utf8_bytes × 重复次数) / 耗时`  
4. Pile 外推：  
   `pile_bytes = 825 × 1024³`  
   `hours = pile_bytes / throughput / 3600`

**注意**：这是 **单进程 Python**、**单机本次测量**；CPU 负载、缓存、并行度都会让数字波动 ±10% 很正常。

### 本次运行结果（2026-07-20）

| Tokenizer | 吞吐 | Pile 825 GiB 粗算 |
|-----------|------|-------------------|
| TS 10k | **1.28 MiB/s**（1,337,022 bytes/s） | **184.0 h** |
| OWT 32k | **1.26 MiB/s**（1,320,034 bytes/s） | **186.4 h** |

两者速度接近：32k 词表更大，但 merge 更高效，在本测试片段上几乎打平。

### 手算 Pile 时间（以 TS tokenizer 为例）

```text
pile_bytes     = 825 × 1024³ = 885,837,004,800  (约 8.86×10¹¹ bytes)
throughput     = 1,337,022 bytes/s
seconds        = 885,837,004,800 / 1,337,022 ≈ 662,545 s
hours          = 662,545 / 3600 ≈ 184.0 h
```

即：按当前单机单进程速度，tokenize 825 GiB **大约 8 天**；多进程分片、C/Rust 实现会快很多，作业只要求 **order-of-magnitude 估计**。

### 你自己怎么复现？

**一键**：§0 命令，看 `"c": { ... }` 和 `(c)` 两行。

**只跑吞吐**：

```bash
cd /root/.dev/ml-sys/cs336/assignment1-basics && uv run python -c "
from pathlib import Path
from scripts.tokenizer_experiments import load_tok, benchmark_throughput
ROOT = Path('.')
with open(ROOT/'data'/'TinyStoriesV2-GPT4-train.txt', encoding='utf-8') as f:
    text = f.read(2_000_000)
for name in ('tinystories_bpe', 'owt_bpe'):
    bps = benchmark_throughput(text, load_tok(name))
    print(name, f'{bps/1024/1024:.2f} MiB/s', f'pile_h={825*1024**3/bps/3600:.1f}')
"
```

多跑几次对比波动；换 `min_seconds=5.0` 可让估计更稳（需改 `benchmark_throughput` 调用）。

### 交作业可写（1–2 句）

本机单进程 encode 约 **1.25–1.3 MiB/s**；按此粗算 tokenize Pile（825 GiB）约 **184–186 小时**（未并行）。

---

## II.4 问题 (d)：训练集 encode 落盘与 uint16

### 题目在问什么？

1. 把整个 train/valid **编成 token id 序列**，存成文件供 `train.py` 读。  
2. 解释为什么用 **`uint16`** 而不是 `int32` / `uint8`。

### 流程（从原文到 .npy）

```text
data/owt_train.txt  (UTF-8 文本，含 <|endoftext|>)
  → Tokenizer.from_files(artifacts/owt_bpe/...)
  → tokenizer.encode_iterable(file)   # 流式，不把全文读进 RAM
  → 每个 token id 写成 2 字节 little-endian (struct.pack "<H")
  → 临时 .bin → np.save → artifacts/owt_tokens/owt_train.npy
```

实现：`cs336_basics/tokenization/encode_dataset.py`（与测试/作业一致）。

**重跑命令**（OWT 全量 encode 要很久，TinyStories 较快）：

```bash
cd /root/.dev/ml-sys/cs336/assignment1-basics && uv run python -m cs336_basics.tokenization.encode_dataset --dataset tinystories && uv run python -m cs336_basics.tokenization.encode_dataset --dataset owt
```

只 encode 某一个 split：

```bash
cd /root/.dev/ml-sys/cs336/assignment1-basics && uv run python -m cs336_basics.tokenization.encode_dataset --dataset owt --split train
```

### 磁盘上有什么？（本次检查，文件已存在）

| 数据集 | split | 路径 | token 数 | max_id | 文件大小 |
|--------|-------|------|----------|--------|----------|
| TinyStories | train | `artifacts/tinystories_tokens/tinystories_train.npy` | 540,796,778 | 9,999 | 1.01 GiB |
| TinyStories | valid | `.../tinystories_valid.npy` | 5,461,210 | 9,999 | 10 MiB |
| OWT | train | `artifacts/owt_tokens/owt_train.npy` | 2,727,120,452 | 31,999 | 5.08 GiB |
| OWT | valid | `.../owt_valid.npy` | 66,401,098 | 31,999 | 124 MiB |

数组形状：一维 `numpy` 数组，`dtype=uint16`，每个元素是一个 **token id**（不是 byte）。

**OWT train 头尾 10 个 id**（§0 命令 `"d"` 字段；**不要用 `head`/`tail` 打开 .npy**）：

```text
first10: [2001, 3384, 696, 361, 473, 284, 3890, 2046, 361, 1880]
last10:  [3103, 288, 8304, 294, 548, 2693, 29983, 6956, 14, 0]
```

用 tokenizer **decode 前 10 个 id** 应得到可读英文开头：

```text
"What wouldn't you do to save someone you love"
```

（与 OWT 原文开头一致，说明 encode 正确。）

### 为什么 uint16？

| 类型 | 能表示的 id 范围 | 每个 id 占磁盘 | 本作业是否够用 |
|------|------------------|----------------|----------------|
| `uint8` | 0–255 | 1 byte | ✗ 词表 10k/32k |
| **`uint16`** | **0–65,535** | **2 bytes** | **✓** max_id 31999 |
| `int32` | 很大 | 4 bytes | ✓ 但浪费一倍空间 |

验证：`len(arr) × 2 ≈ 文件字节数`（`.npy` 有少量 header，所以不完全相等但同量级）。

```text
owt_train: len = 2,727,120,452
2 × len = 5,454,240,904 bytes ≈ 5.08 GiB  （与 ls 看到的 5.1G 一致）
```

`encode_dataset` 在写入时若 `token_id > 65535` 会直接报错，防止静默截断。

### 你自己怎么复现 / 学习？

**查看元信息（不加载进 RAM）**：

```bash
cd /root/.dev/ml-sys/cs336/assignment1-basics && uv run python -c "
import numpy as np
from pathlib import Path
p = Path('artifacts/owt_tokens/owt_train.npy')
a = np.load(p, mmap_mode='r')
print('dtype', a.dtype, 'len', len(a), 'max_id', int(a.max()))
print('first10', a[:10].tolist())
print('size_gb', p.stat().st_size/1024**3)
"
```

**decode 验证**：

```bash
cd /root/.dev/ml-sys/cs336/assignment1-basics && uv run python -c "
import numpy as np
from cs336_basics.tokenization.tokenizer import Tokenizer
from pathlib import Path
tok = Tokenizer.from_files('artifacts/owt_bpe/vocab.json','artifacts/owt_bpe/merges.txt',special_tokens=['<|endoftext|>'])
a = np.load('artifacts/owt_tokens/owt_train.npy', mmap_mode='r')
print(tok.decode(a[:20].tolist()))
"
```

**训练时怎么读**：`train.py` 用 `np.load(path, mmap_mode='r')` 随机切 batch，不把 5 GiB 一次性载入内存。

### 交作业可写（1–2 句）

已将 TinyStories / OWT 的 train·valid 编成 `uint16` 一维数组（路径见上表）；最大 id 31,999 < 2¹⁶，uint16 足够且比 int32 省一半磁盘与内存。

---

## 附录 A：数据流总图

```text
data/*.txt
  → train_bpe (§I.0)     → artifacts/*_bpe/{vocab.json, merges.txt}
  → encode_dataset (§II.4) → artifacts/*_tokens/*.npy  (uint16 token ids)
  → Tokenizer.encode       → (a)(b)(c) 用 scripts/tokenizer_experiments.py
  → train.py mmap .npy     → 训练 LM
```

## 附录 B：脚本与产物索引

| 用途 | 路径 |
|------|------|
| (a)(b)(c)(d) 汇总脚本 | `scripts/tokenizer_experiments.py` |
| 一次完整原始输出（归档） | 本文件 **附录 C** |
| BPE 训练 CLI | `python -m cs336_basics.tokenization.train_bpe` |
| 全量 encode CLI | `python -m cs336_basics.tokenization.encode_dataset` |
| Tokenizer 实现 | `cs336_basics/tokenization/tokenizer.py` |

---

## 附录 C：实验记录（原始脚本输出，2026-07-20）

以下内容由 `scripts/tokenizer_experiments.py` 打印，原封归档（原独立文件 `tokenizer_experiments_last_run.txt` 已删除并入此处）。

```text
========================================================================
CS336 tokenizer_experiments.py results
========================================================================
{
  "a": {
    "tinystories_bytes_per_token": 4.112278761061947,
    "tinystories_sample_bytes": 7435,
    "tinystories_sample_tokens": 1808,
    "owt_bytes_per_token": 4.691150178784267,
    "owt_sample_bytes": 31487,
    "owt_sample_tokens": 6712
  },
  "b": {
    "owt_sample_with_tinystories_tok_bytes_per_token": 3.1892028765319558,
    "owt_sample_tokens_with_tinystories_tok": 9873,
    "owt_sample_with_owt_tok_bytes_per_token": 4.691150178784267,
    "ratio_worse_than_matched": 0.679833890408184
  },
  "c": {
    "bench_chunk_mib": 1.908172607421875,
    "tinystories_tok_bytes_per_sec": 1337021.7581524993,
    "owt_tok_bytes_per_sec": 1320033.9171077928,
    "pile_gb": 825.0,
    "pile_hours_tinystories_tok": 184.04026199744212,
    "pile_hours_owt_tok": 186.40872138028035
  },
  "d": {
    "tinystories_train": {
      "path": "/root/.dev/ml-sys/cs336/assignment1-basics/artifacts/tinystories_tokens/tinystories_train.npy",
      "dtype": "uint16",
      "len": 540796778,
      "max_id": 9999,
      "first10": [
        199,
        430,
        439,
        259,
        398,
        401,
        283,
        259,
        390,
        496
      ],
      "last10": [
        364,
        2412,
        474,
        14,
        339,
        324,
        12,
        317,
        57,
        79
      ],
      "size_gb": 1.0073126144707203
    },
    "tinystories_valid": {
      "path": "/root/.dev/ml-sys/cs336/assignment1-basics/artifacts/tinystories_tokens/tinystories_valid.npy",
      "dtype": "uint16",
      "len": 5461210,
      "max_id": 9999,
      "first10": [
        85,
        862,
        492,
        499,
        266,
        322,
        608,
        370,
        263,
        911
      ],
      "last10": [
        336,
        411,
        2412,
        263,
        4301,
        267,
        405,
        378,
        376,
        14
      ],
      "size_gb": 0.010172415524721146
    },
    "owt_train": {
      "path": "/root/.dev/ml-sys/cs336/assignment1-basics/artifacts/owt_tokens/owt_train.npy",
      "dtype": "uint16",
      "len": 2727120452,
      "max_id": 31999,
      "first10": [
        2001,
        3384,
        696,
        361,
        473,
        284,
        3890,
        2046,
        361,
        1880
      ],
      "last10": [
        3103,
        288,
        8304,
        294,
        548,
        2693,
        29983,
        6956,
        14,
        0
      ],
      "size_gb": 5.0796578004956245
    },
    "owt_valid": {
      "path": "/root/.dev/ml-sys/cs336/assignment1-basics/artifacts/owt_tokens/owt_valid.npy",
      "dtype": "uint16",
      "len": 66401098,
      "max_id": 31999,
      "first10": [
        44,
        4092,
        2148,
        54,
        4192,
        5351,
        12,
        11970,
        14,
        746
      ],
      "last10": [
        19167,
        317,
        480,
        975,
        1268,
        318,
        2467,
        384,
        720,
        0
      ],
      "size_gb": 0.12368180230259895
    }
  }
}
========================================================================
(a) TinyStories sample, TS 10k tok: 4.112 bytes/token (7435 bytes / 1808 tokens)
(a) OWT sample, OWT 32k tok:        4.691 bytes/token (31487 bytes / 6712 tokens)
(b) OWT sample, TS 10k tok:           3.189 bytes/token (9873 tokens); 1.47x more tokens than matched OWT tok
(c) throughput TS tok: 1.28 MiB/s | OWT tok: 1.26 MiB/s
(c) Pile 825 GiB estimate: TS tok 184.0 h | OWT tok 186.4 h
(d) see encoded .npy max_id vs uint16; files under artifacts/*_tokens/
```

