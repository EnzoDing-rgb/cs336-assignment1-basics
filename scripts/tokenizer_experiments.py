#!/usr/bin/env python3
"""CS336 A1 tokenizer experiments: (a)–(d) metrics for reports/tokenization.md."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenization.tokenizer import Tokenizer

ROOT = Path("/root/.dev/ml-sys/cs336/assignment1-basics")
SPECIAL = "<|endoftext|>"
SEP = "=" * 72


def load_tok(name: str) -> Tokenizer:
    d = ROOT / "artifacts" / name
    return Tokenizer.from_files(
        str(d / "vocab.json"),
        str(d / "merges.txt"),
        special_tokens=[SPECIAL],
    )


def sample_documents(path: Path, n: int = 10) -> list[str]:
    """First n non-empty documents split on <|endoftext|> (streaming, safe for huge files)."""
    docs: list[str] = []
    buf = ""
    with open(path, encoding="utf-8") as f:
        while len(docs) < n:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                if buf.strip():
                    docs.append(buf)
                break
            buf += chunk
            while SPECIAL in buf and len(docs) < n:
                doc, _, buf = buf.partition(SPECIAL)
                if doc.strip():
                    docs.append(doc)
    return docs[:n]


def compression_ratio_bytes_per_token(docs: list[str], tokenizer: Tokenizer) -> tuple[float, int, int]:
    total_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
    ids: list[int] = []
    for doc in docs:
        ids.extend(tokenizer.encode(doc))
    num_tokens = len(ids)
    ratio = total_bytes / num_tokens if num_tokens else float("nan")
    return ratio, total_bytes, num_tokens


def benchmark_throughput(text: str, tokenizer: Tokenizer, *, min_seconds: float = 2.0) -> float:
    """Return bytes/sec over encode(text) repeated until min_seconds elapsed."""
    nbytes = len(text.encode("utf-8"))
    if nbytes == 0:
        return 0.0
    # warmup
    tokenizer.encode(text)
    start = time.perf_counter()
    reps = 0
    while time.perf_counter() - start < min_seconds:
        tokenizer.encode(text)
        reps += 1
    elapsed = time.perf_counter() - start
    return (nbytes * reps) / elapsed


def summarize_npy(path: Path) -> dict:
    arr = np.load(path, mmap_mode="r")
    return {
        "path": str(path),
        "dtype": str(arr.dtype),
        "len": int(len(arr)),
        "max_id": int(arr.max()),
        "first10": arr[:10].tolist(),
        "last10": arr[-10:].tolist(),
        "size_gb": path.stat().st_size / (1024**3),
    }


def main() -> None:
    ts_tok = load_tok("tinystories_bpe")
    owt_tok = load_tok("owt_bpe")

    ts_docs = sample_documents(ROOT / "data" / "TinyStoriesV2-GPT4-train.txt", 10)
    owt_docs = sample_documents(ROOT / "data" / "owt_train.txt", 10)

    ts_on_ts, ts_b, ts_tok_n = compression_ratio_bytes_per_token(ts_docs, ts_tok)
    owt_on_owt, owt_b, owt_tok_n = compression_ratio_bytes_per_token(owt_docs, owt_tok)
    owt_on_ts, _, owt_ts_tok_n = compression_ratio_bytes_per_token(owt_docs, ts_tok)

    with open(ROOT / "data" / "TinyStoriesV2-GPT4-train.txt", encoding="utf-8") as f:
        bench_text = f.read(2_000_000)
    ts_bps = benchmark_throughput(bench_text, ts_tok)
    owt_bps = benchmark_throughput(bench_text, owt_tok)
    pile_gb = 825.0
    pile_bytes = pile_gb * (1024**3)
    pile_sec_ts = pile_bytes / ts_bps
    pile_sec_owt = pile_bytes / owt_bps

    ts_train = summarize_npy(ROOT / "artifacts" / "tinystories_tokens" / "tinystories_train.npy")
    ts_valid = summarize_npy(ROOT / "artifacts" / "tinystories_tokens" / "tinystories_valid.npy")
    owt_train = summarize_npy(ROOT / "artifacts" / "owt_tokens" / "owt_train.npy")
    owt_valid = summarize_npy(ROOT / "artifacts" / "owt_tokens" / "owt_valid.npy")

    out = {
        "a": {
            "tinystories_bytes_per_token": ts_on_ts,
            "tinystories_sample_bytes": ts_b,
            "tinystories_sample_tokens": ts_tok_n,
            "owt_bytes_per_token": owt_on_owt,
            "owt_sample_bytes": owt_b,
            "owt_sample_tokens": owt_tok_n,
        },
        "b": {
            "owt_sample_with_tinystories_tok_bytes_per_token": owt_on_ts,
            "owt_sample_tokens_with_tinystories_tok": owt_ts_tok_n,
            "owt_sample_with_owt_tok_bytes_per_token": owt_on_owt,
            "ratio_worse_than_matched": owt_on_ts / owt_on_owt,
        },
        "c": {
            "bench_chunk_mib": len(bench_text.encode("utf-8")) / (1024**2),
            "tinystories_tok_bytes_per_sec": ts_bps,
            "owt_tok_bytes_per_sec": owt_bps,
            "pile_gb": pile_gb,
            "pile_hours_tinystories_tok": pile_sec_ts / 3600,
            "pile_hours_owt_tok": pile_sec_owt / 3600,
        },
        "d": {
            "tinystories_train": ts_train,
            "tinystories_valid": ts_valid,
            "owt_train": owt_train,
            "owt_valid": owt_valid,
        },
    }

    print(SEP)
    print("CS336 tokenizer_experiments.py results")
    print(SEP)
    print(json.dumps(out, indent=2))
    print(SEP)
    print("(a) TinyStories sample, TS 10k tok: %.3f bytes/token (%d bytes / %d tokens)" % (ts_on_ts, ts_b, ts_tok_n))
    print("(a) OWT sample, OWT 32k tok:        %.3f bytes/token (%d bytes / %d tokens)" % (owt_on_owt, owt_b, owt_tok_n))
    print("(b) OWT sample, TS 10k tok:           %.3f bytes/token (%d tokens); %.2fx more tokens than matched OWT tok" % (owt_on_ts, owt_ts_tok_n, owt_ts_tok_n / owt_tok_n))
    print("(c) throughput TS tok: %.2f MiB/s | OWT tok: %.2f MiB/s" % (ts_bps / (1024**2), owt_bps / (1024**2)))
    print("(c) Pile 825 GiB estimate: TS tok %.1f h | OWT tok %.1f h" % (pile_sec_ts / 3600, pile_sec_owt / 3600))
    print("(d) see encoded .npy max_id vs uint16; files under artifacts/*_tokens/")


if __name__ == "__main__":
    main()
