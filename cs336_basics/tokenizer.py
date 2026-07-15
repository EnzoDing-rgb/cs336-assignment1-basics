from __future__ import annotations

import regex as re
import time
from collections import Counter
from multiprocessing import Pool, cpu_count

DEFAULT_PRE_TOKENIZE_WORKERS = 4


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    *,
    pre_tokenize_workers: int = DEFAULT_PRE_TOKENIZE_WORKERS,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 返回值是「一个二元组」
    #   第 1 项 vocab:  dict[int, bytes]
    #   第 2 项 merges: list[tuple[bytes, bytes]]

    vocab = init_vocab(special_tokens)

    corpus = read_corpus(input_path)
    segments = split_special_tokens(corpus, special_tokens)
    pretokens = pre_tokenize(segments, workers=pre_tokenize_workers)
    merges = merge_loop(pretokens, vocab, vocab_size)

    return vocab, merges

# GPT-2 pre-tokenizer pattern (assignment handout)
_GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# def merge_loop(
#     pretokens: list[str],
#     vocab: dict[int, bytes],
#     vocab_size: int,
# ) -> list[tuple[bytes, bytes]]:
#     merges: list[tuple[bytes, bytes]] = []

#   # """
#   #   ┌─────────────────────┬──────────────────────────────────────────────────────────────┐
#   #   │ 变量名              │ 长什么样（例子）                                              │
#   #   ├─────────────────────┼──────────────────────────────────────────────────────────────┤
#   #   │ pretokens           │ list[str]，可重复，整份语料的 pre-token 流                    │
#   #   │                     │ [" the"," the"," iron"," the", ...]                          │
#   #   ├─────────────────────┼──────────────────────────────────────────────────────────────┤
#   #   │ pretoken_counts     │ Counter = {str: int}，每种 pre-token 字符串出现几次           │
#   #   │                     │ {" the": 5000, " iron": 120, " cement": 80}                  │
#   #   ├─────────────────────┼──────────────────────────────────────────────────────────────┤
#   #   │ pretoken_byte_seqs  │ list[list[bytes]]，每种「唯一 pre-token」当前的字节序列        │
#   #   │                     │ [0] [b' ', b't', b'h', b'e']    ← 形状来自 " the"            │
#   #   │                     │ [1] [b' ', b'i', b'r', b'o', b'n'] ← 形状来自 " iron"        │
#   #   │                     │ merge 后会变短，如 [b' t', b'h', b'e']                       │
#   #   ├─────────────────────┼──────────────────────────────────────────────────────────────┤
#   #   │ pretoken_freqs      │ list[int]，与 pretoken_byte_seqs 同下标对齐                   │
#   #   │                     │ [0] 5000  表示 " the" 这条形状在语料里出现 5000 次          │
#   #   │                     │ [1] 120   表示 " iron" 出现 120 次                           │
#   #   ├─────────────────────┼──────────────────────────────────────────────────────────────┤
#   #   │ byte_seq            │ list[bytes]，循环里的「一条」序列 = pretoken_byte_seqs[i]     │
#   #   │                     │ 初始: [b' ', b't', b'h', b'e']                              │
#   #   │                     │ 第1轮merge后: [b' t', b'h', b'e']                            │
#   #   │                     │ 每个元素是一个 token（先是1字节，merge后可能多字节）          │
#   #   ├─────────────────────┼──────────────────────────────────────────────────────────────┤
#   #   │ freq                │ int，这条 byte_seq 对应形状的出现次数（= pretoken_freqs[i]） │
#   #   │                     │ 5000                                                         │
#   #   ├─────────────────────┼──────────────────────────────────────────────────────────────┤
#   #   │ pair                │ tuple[bytes, bytes]，相邻两个 token 组成的二元组              │
#   #   │                     │ (b' ', b't')  (b't', b'h')  (b'h', b'e')                     │
#   #   ├─────────────────────┼──────────────────────────────────────────────────────────────┤
#   #   │ pair_counts         │ Counter = {(bytes,bytes): int}，全局 pair 频次               │
#   #   │                     │ {(b' ', b't'): 18234, (b' ', b'a'): 9102, ...}               │
#   #   ├─────────────────────┼──────────────────────────────────────────────────────────────┤
#   #   │ best_pair           │ tuple[bytes, bytes]，本轮要合并的最高频 pair                  │
#   #   │                     │ (b' ', b't')                                                 │
#   #   ├─────────────────────┼──────────────────────────────────────────────────────────────┤
#   #   │ vocab[next_id]      │ bytes，本轮新造出来的 token                                   │
#   #   │                     │ vocab[257] = b' ' + b't' = b' t'                             │
#   #   ├─────────────────────┼──────────────────────────────────────────────────────────────┤
#   #   │ merges              │ list[tuple[bytes,bytes]]，按顺序记录每次 merge                │
#   #   │                     │ [(b' ', b't'), (b' ', b'a'), (b'h', b'e'), ...]              │
#   #   └─────────────────────┴──────────────────────────────────────────────────────────────┘
#   #   """

#     # pretokens 例子（整份语料里有很多条，允许重复）:
#     #   [" the", " the", " iron", " the", " iron", ...]
#     #
#     # Counter(pretokens) 数「每种 pre-token 字符串出现了几次」，得到类似:
#     #   {" the": 5000, " iron": 120, " cement": 80, ...}
#     # 你可以把它想成: 字符串 -> 出现次数 的字典（自动帮你 +1 计数）
#     pretoken_counts = Counter(pretokens)

#     # pretoken_byte_seqs 和 pretoken_freqs 是两条「对齐」的 list，用同一个下标 i 访问:
#     #
#     #   i=0: pretoken_byte_seqs[0] = [b' ', b't', b'h', b'e']   # 来自 pre-token " the"
#     #        pretoken_freqs[0]     = 5000                         # " the" 在语料里出现 5000 次
#     #
#     #   i=1: pretoken_byte_seqs[1] = [b' ', b'i', b'r', b'o', b'n']
#     #        pretoken_freqs[1]     = 120                          # " iron" 出现 120 次
#     #
#     # 注意: pretoken_freqs 不是 byte 的频次，是「整条 pre-token 字符串」的频次。
#     pretoken_byte_seqs: list[list[bytes]] = []
#     pretoken_freqs: list[int] = []
#     for pretoken, count in pretoken_counts.items():
#         # 例: pretoken=" the" -> pretoken.encode("utf-8")=b' the' -> [b' ', b't', b'h', b'e']
#         byte_seq = [bytes([byte_val]) for byte_val in pretoken.encode("utf-8")]
#         pretoken_byte_seqs.append(byte_seq)
#         pretoken_freqs.append(count)

#     # 例: 已有 special+256 字节后，next_id 从 257 开始往 vocab 里填 merge 出来的新 token
#     next_id = len(vocab)
#     while next_id < vocab_size:
#         # pair_counts 例子（一轮 merge 开始时统计出来的）:
#         #   {(b' ', b't'): 18234, (b' ', b'a'): 9102, (b'h', b'e'): 6401, ...}
#         # 含义: 相邻字节对 (左token, 右token) -> 在语料里一共出现多少次
#         # 例: (b' ', b't') 的 18234 = 每个 pre-token 内部数到的 (空格,t) 对数 × 该 pre-token 的频次，再全局相加
#         pair_counts: Counter[tuple[bytes, bytes]] = Counter()

#         for byte_seq, freq in zip(pretoken_byte_seqs, pretoken_freqs):
#             # byte_seq=[b' ',b't',b'h',b'e'], freq=5000 时:
#             #   i=0 -> pair=(b' ', b't')  贡献 +5000
#             #   i=1 -> pair=(b't', b'h')  贡献 +5000
#             #   i=2 -> pair=(b'h', b'e')  贡献 +5000
#             # 不会在 b'e' 后面跨到下一条 pre-token（每条 byte_seq 单独循环）
#             for i in range(len(byte_seq) - 1):
#                 pair = (byte_seq[i], byte_seq[i + 1])
#                 pair_counts[pair] += freq

#         if not pair_counts:
#             break

#         # 例: 若最高频是 (b' ', b't'):18234 和 (b' ', b'a'):9102 -> 选 (b' ', b't')
#         # 若两个 pair 频次相同，选字典序更大的那个，如 (b'BA', b'A') 胜过 (b'A', b'B')
#         best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))

#         # 例: best_pair=(b' ', b't') 之后
#         #   [b' ', b't', b'h', b'e']  ->  [b' t', b'h', b'e']
#         #   [b' ', b't', b'o']        ->  [b' to']
#         for byte_seq in pretoken_byte_seqs:
#             _merge_pair_in_place(byte_seq, best_pair)

#         # 例: vocab[257] = b' ' + b't' = b' t'；merges 追加 (b' ', b't')
#         vocab[next_id] = best_pair[0] + best_pair[1]
#         merges.append(best_pair)
#         next_id += 1

#     return merges


def merge_loop(
    pretokens: list[str],
    vocab: dict[int, bytes],
    vocab_size: int,
    *,
    progress_interval_s: float | None = None,
) -> list[tuple[bytes, bytes]]:
    """优化版 BPE merge：全局维护 pair_counts，只在 merge 发生处增量改计数。"""
    merges: list[tuple[bytes, bytes]] = []

    pretoken_counts = Counter(pretokens)
    pretoken_byte_seqs: list[list[bytes]] = []
    pretoken_freqs: list[int] = []
    for pretoken, count in pretoken_counts.items():
        # 例: pretoken=" text", count=5000
        #   -> byte_seq=[b' ', b't', b'e', b'x', b't'], pretoken_freqs 里记 5000
        byte_seq = [bytes([b]) for b in pretoken.encode("utf-8")]
        pretoken_byte_seqs.append(byte_seq)
        pretoken_freqs.append(count)

    # 首轮全量统计；之后不再重扫语料
    # 接上例: pair_counts[(b' ',b't')]+=5000, (b't',b'e')+=5000, ...
    pair_counts = _count_all_pairs(pretoken_byte_seqs, pretoken_freqs)

    next_id = len(vocab)
    merges_target = vocab_size - next_id
    merge_start = time.perf_counter()
    last_progress = merge_start
    while next_id < vocab_size:
        if not pair_counts:
            break

        # 例: best_pair=(b' ', b't'), pair_counts[(b' ',b't')]=120000（全局最高）
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        left, right = best_pair
        merged = left + right  # b' t'

        # 只遍历「含这个 pair 的 pre-token 形状」，每条里只动 merge 位置附近的计数
        for byte_seq, freq in zip(pretoken_byte_seqs, pretoken_freqs):
            _apply_best_pair_merge(byte_seq, left, right, freq, pair_counts)

        vocab[next_id] = merged
        merges.append(best_pair)
        next_id += 1

        if progress_interval_s is not None:
            now = time.perf_counter()
            if now - last_progress >= progress_interval_s:
                done = len(merges)
                pct = 100.0 * done / merges_target if merges_target else 100.0
                elapsed = now - merge_start
                print(
                    f"[merge_loop] {done}/{merges_target} merges ({pct:.1f}%) "
                    f"elapsed={elapsed:.0f}s pair={best_pair!r}",
                    flush=True,
                )
                last_progress = now

    return merges


def _count_all_pairs(
    pretoken_byte_seqs: list[list[bytes]],
    pretoken_freqs: list[int],
) -> Counter[tuple[bytes, bytes]]:
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for byte_seq, freq in zip(pretoken_byte_seqs, pretoken_freqs):
        for i in range(len(byte_seq) - 1):
            # 例: byte_seq=[b' ',b't',b'h',b'o'], freq=100, i=0
            #   pair_counts[(b' ',b't')] += 100
            #   i=1 -> (b't',b'h') += 100
            #   i=2 -> (b'h',b'o') += 100
            pair_counts[(byte_seq[i], byte_seq[i + 1])] += freq
    return pair_counts


def _adjust_pair_count(
    pair_counts: Counter[tuple[bytes, bytes]],
    pair: tuple[bytes, bytes],
    delta: int,
) -> None:
    if delta == 0:
        return
    pair_counts[pair] += delta
    # 例: pair_counts[(b' ',b't')]=100, delta=-100 -> 删掉这个 key（全局不再有这个 pair）
    if pair_counts[pair] <= 0:
        del pair_counts[pair]


def _apply_best_pair_merge(
    byte_seq: list[bytes],
    left: bytes,
    right: bytes,
    freq: int,
    pair_counts: Counter[tuple[bytes, bytes]],
) -> None:
    # 在一条 byte_seq 里找 best_pair 的所有出现位置（index 列表），逐个 merge。
    #
    # 例: byte_seq=[b' ',b't',b'h',b' ',b't',b'o'], freq=100, merge (b' ',b't')
    #   positions = [0, 3]   ← 你的 index 列表
    #
    #   i=0: 命中 -> _merge_at_index(..., i=0) -> [b' t',b'h',b' ',b't',b'o']
    #        不 i++（当前位已是 b' t'，还要继续检查）
    #   i=0: (b' t',b'h') 不匹配，i=1
    #   i=1: (b'h',b' ') 不匹配，i=2
    #   i=2: 命中 -> _merge_at_index(..., i=2) -> [b' t',b'h',b' t',b'o']
    #
    # 注意: 不能先 positions=[0,3] 收集完再倒序 merge（重叠 pair 会错），必须从左到右扫。
    i = 0
    while i < len(byte_seq) - 1:
        if byte_seq[i] == left and byte_seq[i + 1] == right:
            _merge_at_index(byte_seq, i, left, right, freq, pair_counts)
        else:
            i += 1


def _merge_at_index(
    byte_seq: list[bytes],
    i: int,
    left: bytes,
    right: bytes,
    freq: int,
    pair_counts: Counter[tuple[bytes, bytes]],
) -> None:
    # 增量更新的核心：只改「位置 i 左边、中间、右边」最多 3 对 pair 的计数。
    #
    # 例: byte_seq=[b'h',b'i',b' ',b't',b'h',b'e'], i=2, left=b' ', right=b't', freq=5000
    #   merge 前局部: ... [b'i'] [b' '][b't'] [b'h'] ...
    #                  index:  1     2   3      4
    #
    #   先减掉 merge 前存在的 3 对:
    #     (b'i', b' ')  -= 5000
    #     (b' ', b't')  -= 5000   ← best_pair 本身
    #     (b't', b'h')  -= 5000
    #
    #   原地改序列: [b'h',b'i',b' t',b'h',b'e']  （长度 6 -> 5）
    #
    #   再加上 merge 后新出现的 2 对:
    #     (b'i', b' t') += 5000
    #     (b' t', b'h') += 5000
    #
    #   (b'h',b'e') 等离得远的 pair 完全不动 —— 这就是比 naive 快的原因。
    merged = left + right

    if i > 0:
        _adjust_pair_count(pair_counts, (byte_seq[i - 1], left), -freq)
    _adjust_pair_count(pair_counts, (left, right), -freq)
    if i + 2 < len(byte_seq):
        _adjust_pair_count(pair_counts, (right, byte_seq[i + 2]), -freq)

    byte_seq[i] = merged
    del byte_seq[i + 1]

    if i > 0:
        _adjust_pair_count(pair_counts, (byte_seq[i - 1], merged), freq)
    if i + 1 < len(byte_seq):
        _adjust_pair_count(pair_counts, (merged, byte_seq[i + 1]), freq)


# def _merge_pair_in_place(byte_seq: list[bytes], pair: tuple[bytes, bytes]) -> None:
#     # 例: byte_seq=[b' ', b't', b'h', b'e'], pair=(b' ', b't')
#     #   i=0: 匹配 -> [b' t', b'h', b'e']     （长度 4 变 3）
#     #   i=0: 不再匹配 (b' ', b't')，i=1
#     #   i=1: 检查 (b'h', b'e') ...
#     left, right = pair
#     i = 0
#     while i < len(byte_seq) - 1:
#         if byte_seq[i] == left and byte_seq[i + 1] == right:
#             byte_seq[i] = left + right
#             del byte_seq[i + 1]
#         else:
#             i += 1


def pre_tokenize(
    segments: list[str],
    *,
    workers: int = DEFAULT_PRE_TOKENIZE_WORKERS,
    progress_interval_s: float | None = None,
) -> list[str]:
    """
    对 segments 做 GPT-2 pre-tokenize，返回 list[str]。

    workers=1  -> 单线程（benchmark 对比用）
    workers>=2 -> multiprocessing，默认 4 个进程
    """
    if workers == 1:
        return pre_tokenize_single(segments, progress_interval_s=progress_interval_s)
    return pre_tokenize_parallel(
        segments,
        workers=workers,
        progress_interval_s=progress_interval_s,
    )


def pre_tokenize_single(
    segments: list[str],
    *,
    progress_interval_s: float | None = None,
) -> list[str]:
    """单线程：对每个 segment 跑 finditer，收集 pre-token。"""
    pretokens: list[str] = []
    start = time.perf_counter()
    last_progress = start
    for i, segment in enumerate(segments, start=1):
        for match in re.finditer(_GPT2_PAT, segment):
            pretokens.append(match.group())
        if progress_interval_s is not None:
            now = time.perf_counter()
            if now - last_progress >= progress_interval_s:
                print(
                    f"[pre_tokenize] {i:,}/{len(segments):,} segments "
                    f"({100 * i / len(segments):.1f}%) pretokens={len(pretokens):,} "
                    f"elapsed={now - start:.0f}s",
                    flush=True,
                )
                last_progress = now
    return pretokens


def _pretokenize_segment_batch(segments: list[str]) -> list[str]:
    """worker：处理一批 segment，返回这批的 pre-token list（供 Pool.map 调用）。"""
    pretokens: list[str] = []
    for segment in segments:
        for match in re.finditer(_GPT2_PAT, segment):
            pretokens.append(match.group())
    return pretokens


def pre_tokenize_parallel(
    segments: list[str],
    *,
    workers: int = DEFAULT_PRE_TOKENIZE_WORKERS,
    progress_interval_s: float | None = None,
) -> list[str]:
    """
    multiprocessing 版：把 segments 切成很多小块，各进程 finditer，最后 extend 合并。

    注意：这是多进程（process），不是多线程（thread）。regex 是 CPU 活，用进程才能绕过 GIL。
    """
    if not segments:
        return []

    n_workers = min(max(workers, 1), len(segments))
    if n_workers <= 1:
        return pre_tokenize_single(segments, progress_interval_s=progress_interval_s)

    # 不要只切 4 大块（每块几十万 segment 会沉默很久）；切细一点方便负载均衡和打进度
    chunk_size = max(5_000, len(segments) // (n_workers * 32))
    chunks = [segments[i : i + chunk_size] for i in range(0, len(segments), chunk_size)]

    print(
        f"[pre_tokenize] {len(segments):,} segments -> {len(chunks):,} chunks, "
        f"{n_workers} workers",
        flush=True,
    )

    pretokens: list[str] = []
    start = time.perf_counter()
    last_progress = start
    done_chunks = 0

    with Pool(n_workers) as pool:
        for partial in pool.imap_unordered(_pretokenize_segment_batch, chunks):
            pretokens.extend(partial)
            done_chunks += 1
            if progress_interval_s is not None:
                now = time.perf_counter()
                if now - last_progress >= progress_interval_s:
                    print(
                        f"[pre_tokenize] {done_chunks:,}/{len(chunks):,} chunks "
                        f"({100 * done_chunks / len(chunks):.1f}%) "
                        f"pretokens={len(pretokens):,} elapsed={now - start:.0f}s",
                        flush=True,
                    )
                    last_progress = now

    return pretokens


def split_special_tokens(corpus: str, special_tokens: list[str]) -> list[str]:
    """
    按 special token 切开大字符串，返回「多段普通文本」。

    ── 输入 ──
    corpus: str
        read_corpus 读出来的整份语料，一个巨大的字符串。
        里面可能夹着很多篇文档，文档之间用 <|endoftext|> 之类标记分隔。

    special_tokens: list[str]
        调用方传进来的「特殊字符串」列表，不是写死在函数里的。
        测试里常见: special_tokens=["<|endoftext|>"]
        以后也可以有多个，比如 ["<|endoftext|>", "<|pad|>"]

    ── 输出 ──
    list[str]
        每一段是一个「不含 special token 本身」的文本片段。
        每一段会单独做后面的 pre-tokenize 和 BPE 统计。
        相邻两段之间原来有一个 special token，但 special token 不会出现在结果里。

    ── 用什么切？不是空格！──
    切分符 = special_tokens 里的那些字符串（精确匹配整个子串）。
    例如 special_tokens=["<|endoftext|>"] 时，切分符就是字面量 "<|endoftext|>"。

    是:
      - 在语料里找到 "<|endoftext|>" 这个完整子串，在它出现的位置下刀
      - 多个 special token 时，任意一个出现都算边界

  ── 举例 ──
    输入 corpus:
        "Doc A text<|endoftext|>Doc B text<|endoftext|>Doc C"

    输出 segments（示意）:
        ["Doc A text", "Doc B text", "Doc C"]

    注意:
      - "<|endoftext|>" 本身不在输出里
      - "Doc A text" 里的空格都还在
      - 三段彼此独立，后面统计 pair 时不能跨段 merge

    边界情况（实现时要处理）:
      - 开头就是 <|endoftext|>  → 可能产生空字符串 ""
      - 连续两个 <|endoftext|>  → 中间也会产生 ""
      - 没有 special token 的语料   → 返回 ["整份 corpus"] 一项
      - 空字符串 segment 通常应跳过，不参与统计

  ── 实现思路（单线程版，先想清再写）──
    1. 用 re.split，分隔 pattern 由 special_tokens 拼出来
    2. 每个 token 要先 re.escape()，因为 token 里可能有 | < > 等正则特殊字符
    3. 多个 token 用 "|" 连接成一个大 pattern
    4. 对 split 得到的 list 过滤掉 "" 空段
    5. return 剩下的 list[str]
    """
    if not special_tokens:
        return [corpus] if corpus else []

    # pattern：传给 re.split 的「分隔符正则」，不是返回值类型。
    # re.split(pattern, text) 在每次匹配到 pattern 的位置切开 text。
    # 例：pattern = r"<\|endoftext\|>"  →  按字面量 <|endoftext|> 下刀
    #
    # 多个 special token 时，pattern = "tokA|tokB" 表示「遇到 A 或 B 都切」；
    # 这里的 | 是正则里的「或」，所以每个 token 要先 re.escape。
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    segments = re.split(pattern, corpus)
    return [segment for segment in segments if segment]


def read_corpus(input_path: str) -> str:
    """
    读训练语料，返回整个文件内容作为一个 Python 字符串 (str)。

    返回的是什么？
      - 类型: str（Unicode 文本），不是 bytes
      - 内容: 文件从头到尾的全部文字，可能几百万字符
      - 例子: "iron cement is a ready for use paste ...\\ntranslator Internet ..."

    路径不要硬编码：测试会传入不同的 input_path。
    """
    with open(input_path, encoding="utf-8") as f:
        text: str = f.read()
    return text


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
  """
  初始化词表的前 256 + len(special_tokens) 个格子。

  「500 个格子」怎么理解？
    vocab_size=500 是最终目标；一开始只有 256 个单字节 + special tokens。
    剩下 500 - 256 - len(special_tokens) 个格子，靠后面的 merge 一个一个填进去。

  ID 怎么分配？（要和 reference 对齐，否则 test 挂）
    1. special_tokens[0] -> ID 0, special_tokens[1] -> ID 1, ...
    2. 256 个单字节 token 接在后面，但顺序不是 0,1,2,...,255，
       而是 GPT-2 规定的 bs 顺序（见 _gpt2_byte_order 和 tests/common.py）。
       所以 ID 1 是 b'!'（字节 33），不是 b'\\x00'（字节 0）。

  返回什么类型？
    dict[int, bytes]：键是 token ID，值是该 token 对应的字节序列。
  """
  vocab: dict[int, bytes] = {}

  # 1) special tokens 占最低的 ID
  next_id = 0
  for token in special_tokens:
    vocab[next_id] = token.encode("utf-8")  # str -> bytes；special token 用 UTF-8 编码
    next_id += 1

  # 2) 256 个单字节 token，按 GPT-2 规定的顺序依次分配 ID
  for byte_val in _gpt2_byte_order():
    vocab[next_id] = bytes([byte_val])  # 单个字节的 bytes 长度为 1
    next_id += 1

  return vocab


def _gpt2_byte_order() -> list[int]:
  """
  返回「256 个单字节 token 应该按什么顺序分配 ID」。

  返回值
  ------
  list[int]，长度 256，每个元素是 0..255 之间的一个字节值，且互不重复。
  例如（配合 init_vocab 且只有一个 special token 时）：
      result[0]  == 33   →  vocab[1] == b'!'
      result[1]  == 34   →  vocab[2] == b'"'
      ...
      result[32] == 32   →  某个 ID 对应 b' '（空格；显示为 Ġ）
      ...
      result[255] == 某个剩余字节

  为什么顺序不是 0,1,2,...,255？
  ------------------------------
  测试要求与 GPT-2 / tiktoken 一致。OpenAI 在构造词表时，
  先把「打印出来好看」的字节排在前面，剩下的控制字符、空格等排在后面。
  tests/common.py 里的 gpt2_bytes_to_unicode() 用同一套 bs 顺序；
  本函数只复刻「顺序」，不管每个字节显示成什么 Unicode 字符。

  与 init_vocab 的关系
  --------------------
  init_vocab 会遍历本函数的返回值，依次执行：
      vocab[next_id] = bytes([byte_val])
  所以这里的顺序直接决定「哪个字节拿哪个 ID」。
  """
  # 第 1 段：ASCII 可打印字符  ! " # ... ~  （字节 33..126，共 94 个）
  # 这就是为什么 reference vocab 里 ID 1 是 b'!' 而不是 b'\\x00'
  bs: list[int] = list(range(ord("!"), ord("~") + 1))

  # 第 2 段：拉丁文补充区  ¡ ¢ £ ... ¬  （字节 161..172，共 12 个）
  bs += list(range(ord("¡"), ord("¬") + 1))

  # 第 3 段：拉丁文扩展  ® ¯ ° ... ÿ  （字节 174..255，共 82 个）
  # 三段合计 94 + 12 + 82 = 188 个「视觉上能直接显示」的字节
  bs += list(range(ord("®"), ord("ÿ") + 1))

  # 第 4 段：剩下 256 - 188 = 68 个字节（含空格 32、换行 10、\\x00 等）
  # 按数值 0,1,2,... 依次追加到末尾；空格 byte 32 会落在这里，而不是最前面
  for b in range(256):
    if b not in bs:
      bs.append(b)

  # 最终：长度恰好 256 的排列，是 [0..255] 的一个重排（permutation）
  assert len(bs) == 256
  assert len(set(bs)) == 256
  return bs
