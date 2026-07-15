#!/usr/bin/env python3
"""Train byte-level BPE on a corpus, profile stages, serialize vocab/merges."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import psutil

from cs336_basics.tokenizer import (
    init_vocab,
    merge_loop,
    pre_tokenize,
    read_corpus,
    split_special_tokens,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECIAL_TOKEN = "<|endoftext|>"
VOCAB_SIZE = 10_000

# dataset name -> (input file, output directory)
DATASETS: dict[str, tuple[Path, Path]] = {
    "tinystories": (
        REPO_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
        REPO_ROOT / "artifacts" / "tinystories_bpe",
    ),
    "owt": (
        REPO_ROOT / "data" / "owt_train.txt",
        REPO_ROOT / "artifacts" / "owt_bpe",
    ),
}


@dataclass
class StageTiming:
    stage: str
    seconds: float


@contextmanager
def timed_stage(timings: list[StageTiming], stage: str) -> Iterator[None]:
    print(f"[bpe] {stage} ...", flush=True)
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    timings.append(StageTiming(stage, elapsed))
    print(f"[bpe] {stage} done ({elapsed:.1f}s)", flush=True)


def train_staged(
    input_path: Path,
    output_dir: Path,
    *,
    dataset: str,
    vocab_size: int,
    workers: int,
    profile: bool,
) -> None:
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
    with timed_stage(timings, "init_vocab"):
        vocab = init_vocab([SPECIAL_TOKEN])

    with timed_stage(timings, "read_corpus"):
        corpus = read_corpus(str(input_path))

    with timed_stage(timings, "split_special_tokens"):
        segments = split_special_tokens(corpus, [SPECIAL_TOKEN])
        del corpus
        num_segments = len(segments)

    with timed_stage(timings, "pre_tokenize"):
        pretokens = pre_tokenize(segments, workers=workers, progress_interval_s=30.0)
        del segments
        num_pretokens = len(pretokens)

    with timed_stage(timings, "merge_loop"):
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
            "display": _token_display(longest_bytes),
        },
    }
    (output_dir / "profile_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _print_summary(
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


@lru_cache
def _gpt2_byte_display() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


def _token_display(token: bytes) -> str:
    dec = _gpt2_byte_display()
    return "".join(dec[b] for b in token)


def _write_vocab(vocab: dict[int, bytes], path: Path) -> None:
    payload = {str(i): _token_display(b) for i, b in vocab.items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_merges(merges: list[tuple[bytes, bytes]], path: Path) -> None:
    lines = [f"{_token_display(a)} {_token_display(b)}" for a, b in merges]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _print_summary(
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
    print(f"\n=== {dataset} ===")
    print(f"wall={wall_seconds:.3f}s  peak_rss={peak_mib:.1f} MiB")
    print(f"segments={num_segments:,}  pretokens={num_pretokens:,}  vocab={vocab_size:,}  merges={num_merges:,}")
    print(f"artifacts -> {output_dir.resolve()}\n")
    print("stage profile:")
    for t in sorted(timings, key=lambda x: x.seconds, reverse=True):
        pct = 100 * t.seconds / wall_seconds if wall_seconds else 0
        print(f"  {t.stage:22s} {t.seconds:8.3f}s ({pct:5.1f}%)")
    print(f"=> slowest: {slowest.stage}")
    print(f"=> longest token: id={longest_id} len={len(longest_bytes)} {_token_display(longest_bytes)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
        help="tinystories (small) or owt (large)",
    )
    parser.add_argument("--workers", type=int, default=4)
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

    train_staged(
        input_path,
        output_dir,
        dataset=args.dataset,
        vocab_size=args.vocab_size,
        workers=args.workers,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
