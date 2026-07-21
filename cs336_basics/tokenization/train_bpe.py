"""Train byte-level BPE：从原始文本学出 vocab.json + merges.txt。

══════════════════════════════════════════════════════════════════════════════
整个文件在干什么（一句话）
══════════════════════════════════════════════════════════════════════════════
  输入:  data/TinyStoriesV2-GPT4-train.txt  （一整坨 str）
  输出:  artifacts/tokenizers/tinystories_bpe/vocab.json + merges.txt
         → 交给 Tokenizer.from_files() 做 encode/decode

  学的是什么：
    vocab  每个 token id 对应一段 bytes（从 256 个单字节 + special 起步，越学越长）
    merges 按顺序记录「哪两个 bytes token 并成了一个新的」——encode 时按这个顺序合并

══════════════════════════════════════════════════════════════════════════════
调用关系（谁调谁）
══════════════════════════════════════════════════════════════════════════════

  测试 / adapter 入口：
    train_bpe(path, vocab_size=10000, special_tokens)  →  (vocab, merges)

  命令行入口：
    main() → _train_staged_cli()  →  同上流水线 + 写盘 + 打 profile

  核心流水线（train_bpe / _train_staged_cli 共用逻辑）：
    init_vocab(special_tokens)           # id 0..256 先占满
    read_corpus(path)                    # 整文件读成 str
    split_special_tokens(corpus, ...)    # 按 <|endoftext|> 切开，丢掉 special 本身
    pre_tokenize(segments)               # GPT-2 正则切成 pretoken 列表
    merge_loop(pretokens, vocab, size)   # 反复 merge，把 vocab 撑到 vocab_size
    _write_vocab / _write_merges         # 落盘

  merge_loop 内部（优化版，不全表重扫）：
    _count_all_pairs          # 初始：语料里每个相邻 bytes pair 出现几次
    循环:
      选 pair_counts 里最多的 pair
      _apply_best_pair_merge  →  _merge_at_index  →  _adjust_pair_count
      vocab[next_id] = left+right; merges.append((left, right))

══════════════════════════════════════════════════════════════════════════════
数据流例子：语料里只有 "aa" 出现 100 次、"ab" 出现 10 次（极简玩具）
══════════════════════════════════════════════════════════════════════════════

  vocab_size = 260  （1 special + 256 单字节 + 再学 3 个 merge，数字仅示意）

  init_vocab(["<|endoftext|>"])
    → {0: b'<|endoftext|>', 1: b'\\x00', ..., 257: b'a', 258: b'b', ...}
    （真实顺序按 gpt2_byte_order()，这里只说明有 257 个起始 id）

  pre_tokenize → pretokens = ["aa"]*100 + ["ab"]*10  （Counter 后两条）

  merge_loop 第 1 轮：
    pair_counts: (b'a',b'a')=100, (b'a',b'b')=10, ...
    best_pair = (b'a', b'a')     # 100 > 10
    所有 pretoken 里的 aa 并成 b'aa'
    vocab[257] = b'aa'
    merges[0] = (b'a', b'a')

  merge_loop 第 2 轮：
  ...

  最终 vocab 有 260 个 id（0..259），merges 有 260-257=3 条。
  encode 时 Tokenizer 按 merges 顺序（rank 0 最先）做 BPE。

  和 encode 的对应：
    train 产出 merges[i] = (left, right)  →  tokenizer._merge_rank[(left,right)] = i
    train 产出 vocab[id] = bytes           →  tokenizer.vocab[id] = bytes
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import psutil

from cs336_basics.tokenization.utils import (
    DEFAULT_PRE_TOKENIZE_WORKERS,
    bytes_to_gpt2_display,
    gpt2_byte_order,
    merge_bytes_at_index,
    pre_tokenize,
    split_special_tokens,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPECIAL_TOKEN = "<|endoftext|>"
VOCAB_SIZE = 10_000

DATASETS: dict[str, tuple[Path, Path]] = {
    "tinystories": (
        REPO_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
        REPO_ROOT / "artifacts" / "tokenizers" / "tinystories_bpe",
    ),
    "owt": (
        REPO_ROOT / "data" / "owt_train.txt",
        REPO_ROOT / "artifacts" / "tokenizers" / "owt_bpe",
    ),
}


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    *,
    pre_tokenize_workers: int = DEFAULT_PRE_TOKENIZE_WORKERS,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """作业 / tests/adapters 用的入口：读文件 → 训练 BPE → 返回内存里的 vocab + merges。

    输入例:
      input_path     = "data/TinyStoriesV2-GPT4-train.txt"
      vocab_size     = 10000
      special_tokens = ["<|endoftext|>"]

    输出例:
      vocab  = {0: b'<|endoftext|>', 1: b'!', ..., 9999: b'someLongBytes...'}
      merges = [(b'h', b'e'), (b' t', b'he'), ...]   # 长度 = vocab_size - len(初始vocab)

    数据流:
      path → read_corpus → str
           → split_special_tokens → ["Once upon...", "Another story...", ...]
           → pre_tokenize → ["Once", " upon", " a", ...]  （带空格的词是一个 pretoken）
           → merge_loop → 填满 vocab 到 10000 个 id
    """
    vocab = init_vocab(special_tokens)

    corpus = read_corpus(input_path)
    segments = split_special_tokens(corpus, special_tokens)
    pretokens = pre_tokenize(segments, workers=pre_tokenize_workers)
    merges = merge_loop(pretokens, vocab, vocab_size)

    return vocab, merges


def merge_loop(
    pretokens: list[str],
    vocab: dict[int, bytes],
    vocab_size: int,
    *,
    progress_interval_s: float | None = None,
) -> list[tuple[bytes, bytes]]:
    """BPE 训练核心：反复合并语料里最热门的相邻 bytes pair，直到 vocab 满。

    下面函数体里用同一套玩具例子贯穿（数字都跟注释对齐）——
    建议先扫一遍「贯穿例子」块，再对着 while 循环看。
    """
    merges: list[tuple[bytes, bytes]] = []

    # ── 贯穿例子：pretokens 里只有两种片段 ─────────────────────────────────
    #   pretokens = ["aa"]*10 + ["ab"]*1
    #   含义：语料里 "aa" 出现 10 次，"ab" 出现 1 次
    #
    #   假设 vocab 已 init_vocab，len(vocab)=257，vocab_size=259（只再学 2 个 merge）
    # ────────────────────────────────────────────────────────────────────────

    pretoken_counts = Counter(pretokens)
    # pretoken_counts = Counter({"aa": 10, "ab": 1})
    # 去重：两种 pretoken 各保留一条「形状」，频率单独记

    pretoken_byte_seqs: list[list[bytes]] = []
    pretoken_freqs: list[int] = []
    for pretoken, count in pretoken_counts.items():
        byte_seq = [bytes([b]) for b in pretoken.encode("utf-8")]
        pretoken_byte_seqs.append(byte_seq)
        pretoken_freqs.append(count)

    # pretoken_byte_seqs = [
    #   [b'a', b'a'],   # 来自 "aa"
    #   [b'a', b'b'],   # 来自 "ab"
    # ]
    # pretoken_freqs = [10, 1]
    #
    # 三张表平行对齐：第 i 行 = 「这种 pretoken 长什么样」+ 「语料里出现几次」

    pair_counts = _count_all_pairs(pretoken_byte_seqs, pretoken_freqs)
    # pair_counts = Counter({
    #   (b'a', b'a'): 10,   # "aa" 里有 1 个相邻对，× 出现 10 次
    #   (b'a', b'b'):  1,   # "ab" 里有 1 个相邻对，× 出现  1 次
    # })
    # 全局「选票箱」：下一轮 merge 谁，就看谁票多

    next_id = len(vocab)          # 例：257（下一个新 token 的 id）
    merges_target = vocab_size - next_id  # 例：259 - 257 = 2（还要学 2 个 merge）
    merge_start = time.perf_counter()
    last_progress = merge_start
    while next_id < vocab_size:   # 例：257 < 259 → 进循环
        if not pair_counts:
            break

        # ── 第 1 轮 ──────────────────────────────────────────────────────
        # best_pair = (b'a', b'b')? 错！10 > 1 →
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        # best_pair = (b'a', b'a')   merged = b'aa'
        left, right = best_pair
        merged = left + right

        # 扫两条 pretoken，凡是有 (a,a) 的地方都并掉，同时增量改 pair_counts
        for byte_seq, freq in zip(pretoken_byte_seqs, pretoken_freqs):
            _apply_best_pair_merge(byte_seq, left, right, freq, pair_counts)
        # 合并后 pretoken_byte_seqs 变成：
        #   [ [b'aa'],      [b'a', b'b'] ]
        #    ↑ aa 那条      ↑ ab 里没有 (a,a)，不变
        # pair_counts 变成：只剩 {(b'a', b'b'): 1}

        vocab[next_id] = merged    # vocab[257] = b'aa'
        merges.append(best_pair)   # merges = [(b'a', b'a')]
        next_id += 1               # next_id = 258
        # ── 第 2 轮 ──────────────────────────────────────────────────────
        # best_pair = (b'a', b'b')，merged = b'ab'
        # ab 那条 → [b'ab']；aa 那条仍是 [b'aa']（没有 a,b 相邻）
        # vocab[258] = b'ab'；merges = [(b'a',b'a'), (b'a',b'b')]
        # next_id = 259 → 259 < 259 假，循环结束
        # ──────────────────────────────────────────────────────────────────

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

    # 返回 merges：encode 时 rank 0 的 (a,a) 比 rank 1 的 (a,b) 先被尝试
    return merges


def _count_all_pairs(
    pretoken_byte_seqs: list[list[bytes]],
    pretoken_freqs: list[int],
) -> Counter[tuple[bytes, bytes]]:
    """把全语料里所有 pretoken 的「相邻 bytes 对」汇总到一个 Counter。

    两件事同时发生：
      1. 在**一条** byte_seq 里：每个相邻位置产出一个 pair
      2. 在**多条** byte_seq 之间：相同的 pair 累加到同一个 key（按 freq 加权）

    输入例（重点：(t,h) 出现在两种 pretoken 里，要加在一起）:
      pretoken_byte_seqs = [
        [b't', b'h', b'e'],    # 语料片段 "the"
        [b' ', b't', b'h'],    # 语料片段 " th"（前导空格）
      ]
      pretoken_freqs = [100, 10]   # "the" 出现 100 次，" th" 出现 10 次

    输出例:
      Counter({
        (b't', b'h'): 100 + 10,   # ← 跨 pretoken 汇总！两种形状里都有 (t,h)
        (b'h', b'e'): 100,        # 只有 "the" 里有
        (b' ', b't'):  10,        # 只有 " th" 里有
      })

    merge_loop 用这张全局表决定：下一轮全语料 merge 哪一个 pair。
    """
    # 贯穿例子见上方 docstring；"the"×100 + " th"×10
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()

    for byte_seq, freq in zip(pretoken_byte_seqs, pretoken_freqs):
        # 外圈：一种 pretoken 形状 + 它在语料里出现几次
        # 第 1 圈: byte_seq=[t,h,e],  freq=100
        # 第 2 圈: byte_seq=[ ,t,h],  freq= 10

        for i in range(len(byte_seq) - 1):
            # 内圈：在这条 byte_seq 里，每个相邻位置贡献一个 pair
            # 第 1 圈: i=0→(t,h), i=1→(h,e)
            # 第 2 圈: i=0→( ,t), i=1→(t,h)  ← 和第 1 圈的 (t,h) 是同一个 key

            pair_counts[(byte_seq[i], byte_seq[i + 1])] += freq
            # += freq：这种 pretoken 在语料里出现了 freq 次，这个 pair 就算 freq 票
            #
            # 累加过程（同一个 Counter，key 相同就加在一起）:
            #   (t,h): 100        ← 来自 "the"
            #   (h,e): 100
            #   ( ,t):  10        ← 来自 " th"
            #   (t,h): 110        ← (t,h) 再加 10，跨 pretoken 汇总

    return pair_counts
    # Counter({(b't',b'h'): 110, (b'h',b'e'): 100, (b' ',b't'): 10})


def _adjust_pair_count(
    pair_counts: Counter[tuple[bytes, bytes]],
    pair: tuple[bytes, bytes],
    delta: int,
) -> None:
    """pair_counts 里某 pair 的票数 += delta；减到 0 就删掉 key。

    输入例:
      pair_counts[(b't', b'h')] = 110
      _adjust_pair_count(..., (b't', b'h'), -50)
      → pair_counts[(b't', b'h')] = 60

    输入例（减光）:
      pair_counts[(b'x', b'y')] = 3
      _adjust_pair_count(..., (b'x', b'y'), -3)
      → key (b'x', b'y') 从 dict 里消失
    """
    if delta == 0:
        return
    pair_counts[pair] += delta
    if pair_counts[pair] <= 0:
        del pair_counts[pair]


def _apply_best_pair_merge(
    byte_seq: list[bytes],
    left: bytes,
    right: bytes,
    freq: int,
    pair_counts: Counter[tuple[bytes, bytes]],
) -> None:
    """在某一条 pretoken 的 byte_seq 里，把所有 (left, right) 非重叠地 merge 掉。

    输入例:
      byte_seq = [b'a', b'a', b'b']   # pretoken "aab"
      left, right = b'a', b'a'
      freq = 50   # 这个 pretoken 在语料里出现 50 次

    过程:
      i=0: byte_seq[0:2] == (a,a) → _merge_at_index → [b'aa', b'b']
      i=1: 只剩 (aa,b)，不是 (a,a) → 结束

    注意 merge 后 i 不 +1：同一位置可能还能继续合（本例不会）。
    """
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
    """在 byte_seq[i] 和 byte_seq[i+1] 处做一次 merge，并增量更新 pair_counts。

    输入例:
      byte_seq = [b't', b'h', b'e']，i=0，left=b't'，right=b'h'，freq=100

    merge 前相邻 pair 及票数贡献（每条 ×100）:
      (t,h): -100   (h,e): 要减掉右边被拆的影响 ...

    merge 后:
      byte_seq = [b'th', b'e']
      新 pair (th,e): +100

    调用 merge_bytes_at_index 原地改 byte_seq（和 tokenizer encode 用的是同一工具函数）。
    """
    if i > 0:
        _adjust_pair_count(pair_counts, (byte_seq[i - 1], left), -freq)
    _adjust_pair_count(pair_counts, (left, right), -freq)
    if i + 2 < len(byte_seq):
        _adjust_pair_count(pair_counts, (right, byte_seq[i + 2]), -freq)

    merged = merge_bytes_at_index(byte_seq, i, left, right)

    if i > 0:
        _adjust_pair_count(pair_counts, (byte_seq[i - 1], merged), freq)
    if i + 1 < len(byte_seq):
        _adjust_pair_count(pair_counts, (merged, byte_seq[i + 1]), freq)


def read_corpus(input_path: str) -> str:
    """把整个训练文件读成一个 Python str。

    输入例:  input_path = "data/TinyStoriesV2-GPT4-train.txt"
    输出例:  "Once upon a time...<|endoftext|>Another story..."  （几 GB 的一个 str）

    大语料（OWT）会占很多内存；CLI 路径 _train_staged_cli 读完会尽快 split 并 del corpus。
    """
    with open(input_path, encoding="utf-8") as f:
        return f.read()


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """BPE 起点词表：先 special tokens，再 256 个单字节 token。

    输入例:
      special_tokens = ["<|endoftext|>"]

    输出例:
      {
        0: b'<|endoftext|>',
        1: b'\\x00',      # 按 gpt2_byte_order()，不是简单 0..255 顺序
        ...
        256: b'\\xff',    # 共 256 个单字节
      }
      len(vocab) == 257

    之后 merge_loop 从 id=257 开始往 vocab 里 append 新学到的多字节 token。
  """
    vocab: dict[int, bytes] = {}

    next_id = 0
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    for byte_val in gpt2_byte_order():
        vocab[next_id] = bytes([byte_val])
        next_id += 1

    return vocab


@dataclass
class StageTiming:
    """CLI profiling 用：记录某个阶段花了多少秒。

    例: StageTiming(stage="merge_loop", seconds=842.3)
    """

    stage: str
    seconds: float


@contextmanager
def _timed_stage(timings: list[StageTiming], stage: str) -> Iterator[None]:
    """with _timed_stage(timings, "merge_loop"): ...  →  自动记耗时并打印。

    例输出:
      [bpe] merge_loop ...
      ... 训练跑很久 ...
      [bpe] merge_loop done (842.3s)
      timings 里多一条 StageTiming("merge_loop", 842.3)
    """
    print(f"[bpe] {stage} ...", flush=True)
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    timings.append(StageTiming(stage, elapsed))
    print(f"[bpe] {stage} done ({elapsed:.1f}s)", flush=True)


def _train_staged_cli(
    input_path: Path,
    output_dir: Path,
    *,
    dataset: str,
    vocab_size: int,
    workers: int,
    profile: bool,
) -> None:
    """命令行训练入口：分阶段跑 train_bpe 流水线，写产物 + profile 报告。

    输入例:
      input_path  = data/TinyStoriesV2-GPT4-train.txt
      output_dir  = artifacts/tokenizers/tinystories_bpe
      vocab_size  = 10000
      workers     = 8

    输出（写到 output_dir）:
      vocab.json          {"0": "<|endoftext|>", "257": "Ġthe", ...}
      merges.txt          每行一对 GPT-2 display merge
      profile_report.json 各阶段耗时、峰值内存
      cprofile.txt        仅 --profile 时

    和 train_bpe() 的区别：这里管计时、多进程 pre_tokenize、落盘、打日志；测试只调 train_bpe()。
    """
    timings: list[StageTiming] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    wall_start = time.perf_counter()

    def sample_memory() -> None:
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)

    profiler = cProfile.Profile()
    if profile:
        profiler.enable()

    sample_memory()
    with _timed_stage(timings, "init_vocab"):
        vocab = init_vocab([SPECIAL_TOKEN])

    with _timed_stage(timings, "read_corpus"):
        corpus = read_corpus(str(input_path))

    with _timed_stage(timings, "split_special_tokens"):
        segments = split_special_tokens(corpus, [SPECIAL_TOKEN])
        del corpus
        num_segments = len(segments)

    with _timed_stage(timings, "pre_tokenize"):
        pretokens = pre_tokenize(segments, workers=workers, progress_interval_s=30.0)
        del segments
        num_pretokens = len(pretokens)

    with _timed_stage(timings, "merge_loop"):
        merges = merge_loop(pretokens, vocab, vocab_size, progress_interval_s=30.0)
        del pretokens

    if profile:
        profiler.disable()

    sample_memory()
    wall_seconds = time.perf_counter() - wall_start
    peak_mib = peak_rss / (1024 * 1024)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_vocab(vocab, output_dir / "vocab.json")
    _write_merges(merges, output_dir / "merges.txt")

    longest_id, longest_bytes = max(vocab.items(), key=lambda item: len(item[1]))
    slowest = max(timings, key=lambda t: t.seconds)
    cprofile_text = ""
    if profile:
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(30)
        cprofile_text = stream.getvalue()
        (output_dir / "cprofile.txt").write_text(cprofile_text, encoding="utf-8")

    report = {
        "dataset": dataset,
        "wall_seconds": round(wall_seconds, 3),
        "peak_rss_mib": round(peak_mib, 1),
        "num_segments": num_segments,
        "num_pretokens": num_pretokens,
        "slowest_stage": slowest.stage,
        "stage_timings": [asdict(t) for t in timings],
        "longest_token": {
            "id": longest_id,
            "length_bytes": len(longest_bytes),
            "display": bytes_to_gpt2_display(longest_bytes),
        },
    }
    (output_dir / "profile_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _print_cli_summary(
        dataset=dataset,
        output_dir=output_dir,
        wall_seconds=wall_seconds,
        peak_mib=peak_mib,
        num_segments=num_segments,
        num_pretokens=num_pretokens,
        timings=timings,
        slowest=slowest,
        longest_id=longest_id,
        longest_bytes=longest_bytes,
        vocab_size=len(vocab),
        num_merges=len(merges),
    )


def _write_vocab(vocab: dict[int, bytes], path: Path) -> None:
    """内存 vocab → vocab.json（GPT-2 display 格式，给 Tokenizer.from_files 读）。

    输入例:
      vocab = {0: b'<|endoftext|>', 257: b' the'}

    写出文件内容例:
      {"0": "<|endoftext|>", "257": "Ġthe"}
      # b' the' 里的前导空格在文件里显示成 Ġ
    """
    payload = {str(i): bytes_to_gpt2_display(b) for i, b in vocab.items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_merges(merges: list[tuple[bytes, bytes]], path: Path) -> None:
    """内存 merges → merges.txt（一行一对，顺序 = 训练时学到的优先级）。

    输入例:
      merges = [(b' ', b't'), (b' t', b'he')]

    写出文件内容例:
      Ġ t
      Ġt he

    第 0 行 merge 在 encode 时 rank=0（最先尝试合并）。
    """
    lines = [f"{bytes_to_gpt2_display(a)} {bytes_to_gpt2_display(b)}" for a, b in merges]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _print_cli_summary(
    *,
    dataset: str,
    output_dir: Path,
    wall_seconds: float,
    peak_mib: float,
    num_segments: int,
    num_pretokens: int,
    timings: list[StageTiming],
    slowest: StageTiming,
    longest_id: int,
    longest_bytes: bytes,
    vocab_size: int,
    num_merges: int,
) -> None:
    """训练结束在终端打一张可读的汇总表。

    输出例:
      === tinystories ===
      wall=1234.567s  peak_rss=8192.0 MiB
      segments=500,000  pretokens=12,000,000  vocab=10,000  merges=9,743
      artifacts -> /path/to/artifacts/tokenizers/tinystories_bpe
      ...
      => slowest: merge_loop
    """
    print(f"\n=== {dataset} ===")
    print(f"wall={wall_seconds:.3f}s  peak_rss={peak_mib:.1f} MiB")
    print(f"segments={num_segments:,}  pretokens={num_pretokens:,}  vocab={vocab_size:,}  merges={num_merges:,}")
    print(f"artifacts -> {output_dir.resolve()}\n")
    print("stage profile:")
    for t in sorted(timings, key=lambda x: x.seconds, reverse=True):
        pct = 100 * t.seconds / wall_seconds if wall_seconds else 0
        print(f"  {t.stage:22s} {t.seconds:8.3f}s ({pct:5.1f}%)")
    print(f"=> slowest: {slowest.stage}")
    print(f"=> longest token: id={longest_id} len={len(longest_bytes)} {bytes_to_gpt2_display(longest_bytes)!r}")


def main() -> None:
    """CLI: python -m cs336_basics.tokenization.train_bpe --dataset tinystories

    例:
      uv run python -m cs336_basics.tokenization.train_bpe --dataset tinystories
      uv run python -m cs336_basics.tokenization.train_bpe --dataset owt --vocab-size 32000 --workers 8

    读 DATASETS[args.dataset] 里的 (input_path, output_dir)，调 _train_staged_cli。
    """
    parser = argparse.ArgumentParser(description="Train byte-level BPE and serialize vocab/merges.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
        help="tinystories (small) or owt (large)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--profile", action="store_true", help="Enable cProfile (slower)")
    args = parser.parse_args()

    input_path, output_dir = DATASETS[args.dataset]
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing data file: {input_path}")

    print(
        f"[bpe] dataset={args.dataset} input={input_path} workers={args.workers}",
        flush=True,
    )

    _train_staged_cli(
        input_path,
        output_dir,
        dataset=args.dataset,
        vocab_size=args.vocab_size,
        workers=args.workers,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
