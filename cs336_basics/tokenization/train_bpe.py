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
        REPO_ROOT / "artifacts" / "tinystories_bpe",
    ),
    "owt": (
        REPO_ROOT / "data" / "owt_train.txt",
        REPO_ROOT / "artifacts" / "owt_bpe",
    ),
}


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    *,
    pre_tokenize_workers: int = DEFAULT_PRE_TOKENIZE_WORKERS,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
    """优化版 BPE merge：全局维护 pair_counts，只在 merge 发生处增量改计数。"""
    merges: list[tuple[bytes, bytes]] = []

    pretoken_counts = Counter(pretokens)
    pretoken_byte_seqs: list[list[bytes]] = []
    pretoken_freqs: list[int] = []
    for pretoken, count in pretoken_counts.items():
        byte_seq = [bytes([b]) for b in pretoken.encode("utf-8")]
        pretoken_byte_seqs.append(byte_seq)
        pretoken_freqs.append(count)

    pair_counts = _count_all_pairs(pretoken_byte_seqs, pretoken_freqs)

    next_id = len(vocab)
    merges_target = vocab_size - next_id
    merge_start = time.perf_counter()
    last_progress = merge_start
    while next_id < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        left, right = best_pair
        merged = left + right

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
    if pair_counts[pair] <= 0:
        del pair_counts[pair]


def _apply_best_pair_merge(
    byte_seq: list[bytes],
    left: bytes,
    right: bytes,
    freq: int,
    pair_counts: Counter[tuple[bytes, bytes]],
) -> None:
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
    with open(input_path, encoding="utf-8") as f:
        return f.read()


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
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
    stage: str
    seconds: float


@contextmanager
def _timed_stage(timings: list[StageTiming], stage: str) -> Iterator[None]:
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
    payload = {str(i): bytes_to_gpt2_display(b) for i, b in vocab.items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_merges(merges: list[tuple[bytes, bytes]], path: Path) -> None:
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
