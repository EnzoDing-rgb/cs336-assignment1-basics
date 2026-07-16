from __future__ import annotations

import time
from functools import lru_cache
from multiprocessing import Pool

import regex as re

DEFAULT_PRE_TOKENIZE_WORKERS = 4

# GPT-2 pre-tokenizer pattern (assignment handout)
_GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """每个 byte (0..255) -> GPT-2 文件里用的可打印字符（与 tests/common.py 同一套顺序）。

    为什么需要：很多 byte 不是可打印字符（空格、换行、控制字符），直接写进 vocab.json / merges.txt
    会很难看、也容易坏。GPT-2 给每个 byte 分配一个“看起来正常”的 unicode 字符来落盘。

    例:
      mapping = gpt2_bytes_to_unicode()
      mapping[33]  == '!'     # 本来就可打印，原样保留
      mapping[32]  == 'Ġ'     # 空格 ' ' 不可打印友好，映射到 Ġ
      mapping[0]   == 'Ā'     # NUL 控制字符，映射到 Ā
      bytes_to_gpt2_display(b' the') == 'Ġthe'
    """
    # 先放入 188 个“看起来正常”的 byte：这些在文件里直接用 chr(byte) 表示
    byte_values = (
        list(range(ord("!"), ord("~") + 1))  # 33..126  ASCII 可见字符
        + list(range(ord("¡"), ord("¬") + 1))  # 161..172
        + list(range(ord("®"), ord("ÿ") + 1))  # 174..255
    )
    # 与 byte_values 一一对应的 unicode code point；起步时和 byte 相同
    unicode_code_points = byte_values[:]

    # 再把剩下 68 个麻烦 byte（空格/控制字符等）追加进来，
    # 并把它们映射到 256, 257, ... 这样 chr(...) 一定可打印
    # 例: 空格 byte=32 不在上面三组里 -> 追加后映射到某个 >=256 的 code point -> 'Ġ'
    next_shifted_offset = 0
    already_included = set(byte_values)
    for byte_value in range(256):
        if byte_value not in already_included:
            byte_values.append(byte_value)
            unicode_code_points.append(256 + next_shifted_offset)
            next_shifted_offset += 1

    return dict(
        zip(
            byte_values,
            (chr(code_point) for code_point in unicode_code_points),
            strict=True,
        )
    )


def gpt2_byte_order() -> list[int]:
    """GPT-2 vocab 初始化顺序：256 个单字节 token 的 byte value 列表。"""
    return list(gpt2_bytes_to_unicode().keys())


@lru_cache
def gpt2_unicode_to_byte() -> dict[str, int]:
    return {char: byte_val for byte_val, char in gpt2_bytes_to_unicode().items()}


def gpt2_display_to_bytes(display: str) -> bytes:
    decoder = gpt2_unicode_to_byte()
    return bytes([decoder[char] for char in display])


def bytes_to_gpt2_display(token: bytes) -> str:
    encoder = gpt2_bytes_to_unicode()
    return "".join(encoder[byte_val] for byte_val in token)


def merge_bytes_at_index(byte_seq: list[bytes], i: int, left: bytes, right: bytes) -> bytes:
    """在 byte_seq[i:i+2] 合并 left+right，原地改序列，返回 merged。"""
    merged = left + right
    byte_seq[i] = merged
    del byte_seq[i + 1]
    return merged


def apply_merge_pair(byte_seq: list[bytes], left: bytes, right: bytes) -> None:
    """encode 用：对一条 byte_seq 应用一次 merge pair（从左到右扫）。"""
    i = 0
    while i < len(byte_seq) - 1:
        if byte_seq[i] == left and byte_seq[i + 1] == right:
            merge_bytes_at_index(byte_seq, i, left, right)
        else:
            i += 1


def split_text_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """encode 用：按 special token 切开文本，保留 special token 本身（空串也保留）。"""
    if not special_tokens:
        return [text]

    # 长的先匹配，避免 '<|endoftext|><|endoftext|>' 被拆成两个 '<|endoftext|>'
    ordered = sorted(special_tokens, key=len, reverse=True)
    pattern = "(" + "|".join(re.escape(token) for token in ordered) + ")"
    return re.split(pattern, text)


def split_special_tokens(corpus: str, special_tokens: list[str]) -> list[str]:
    """训练用：按 special token 切开语料，丢弃 special token，只保留普通文本段。"""
    if not special_tokens:
        return [corpus] if corpus else []

    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    segments = re.split(pattern, corpus)
    return [segment for segment in segments if segment]


def pre_tokenize(
    segments: list[str],
    *,
    workers: int = DEFAULT_PRE_TOKENIZE_WORKERS,
    progress_interval_s: float | None = None,
) -> list[str]:
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
    if not segments:
        return []

    n_workers = min(max(workers, 1), len(segments))
    if n_workers <= 1:
        return pre_tokenize_single(segments, progress_interval_s=progress_interval_s)

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
