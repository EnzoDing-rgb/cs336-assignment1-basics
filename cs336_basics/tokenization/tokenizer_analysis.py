from __future__ import annotations

import argparse
import random
import time
from collections.abc import Iterator
from pathlib import Path

from cs336_basics.tokenization.tokenizer import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
SPECIAL_TOKEN = "<|endoftext|>"
PILE_BYTES = 825 * (1 << 30)  # 作业：The Pile ≈ 825GB（按 GiB 计）

CORPORA: dict[str, Path] = {
    "tinystories": REPO_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
    "owt": REPO_ROOT / "data" / "owt_train.txt",
}
TOKENIZERS: dict[str, Path] = {
    "tinystories": REPO_ROOT / "artifacts" / "tinystories_bpe",
    "owt": REPO_ROOT / "artifacts" / "owt_bpe",
}


def iter_documents(path: Path, special_token: str = SPECIAL_TOKEN) -> Iterator[str]:
    """流式按 <|endoftext|> 切开文档，避免把整份 OWT 读进内存。"""
    leftover = ""
    with open(path, encoding="utf-8") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            leftover += chunk
            parts = leftover.split(special_token)
            leftover = parts[-1]
            for doc in parts[:-1]:
                doc = doc.strip()
                if doc:
                    yield doc
    leftover = leftover.strip()
    if leftover:
        yield leftover


def sample_documents(path: Path, num_docs: int, seed: int) -> list[str]:
    """蓄水池抽样：扫一遍语料，均匀随机抽出 num_docs 篇文档。"""
    rng = random.Random(seed)
    reservoir: list[str] = []
    for index, doc in enumerate(iter_documents(path)):
        if index < num_docs:
            reservoir.append(doc)
        else:
            j = rng.randint(0, index)
            if j < num_docs:
                reservoir[j] = doc
    if len(reservoir) < num_docs:
        raise RuntimeError(f"Only found {len(reservoir)} documents in {path}, need {num_docs}")
    return reservoir


def compression_ratio(tokenizer: Tokenizer, documents: list[str]) -> tuple[int, int, float]:
    """compression ratio = utf-8 bytes / token 个数（越大 = 每个 token 覆盖越多字节）。"""
    total_bytes = 0
    total_tokens = 0
    for doc in documents:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))
    if total_tokens == 0:
        raise RuntimeError("Encoded 0 tokens; check tokenizer / documents")
    return total_bytes, total_tokens, total_bytes / total_tokens


def measure_throughput(
    tokenizer: Tokenizer,
    documents: list[str],
    *,
    warmup: bool = True,
) -> tuple[int, float, float]:
    """测 encode 吞吐：返回 (utf8_bytes, seconds, bytes_per_second)。"""
    total_bytes = sum(len(doc.encode("utf-8")) for doc in documents)
    if total_bytes == 0:
        raise RuntimeError("Documents are empty; nothing to benchmark")

    if warmup:
        # 先跑一轮，减少第一次 import/缓存带来的噪声
        for doc in documents:
            tokenizer.encode(doc)

    start = time.perf_counter()
    for doc in documents:
        tokenizer.encode(doc)
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        raise RuntimeError("Benchmark elapsed non-positive time")
    return total_bytes, elapsed, total_bytes / elapsed


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def load_tokenizer(artifact_dir: Path) -> Tokenizer:
    vocab_path = artifact_dir / "vocab.json"
    merges_path = artifact_dir / "merges.txt"
    if not vocab_path.is_file() or not merges_path.is_file():
        raise FileNotFoundError(
            f"Missing tokenizer artifacts in {artifact_dir}. "
            f"Train first, e.g. "
            f"`uv run python -m cs336_basics.tokenization.train_bpe --dataset ...`"
        )
    return Tokenizer.from_files(str(vocab_path), str(merges_path), special_tokens=[SPECIAL_TOKEN])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenizer analysis: compression ratio (bytes/token) + encode throughput "
            "(bytes/s) and Pile-time estimate."
        )
    )
    parser.add_argument(
        "--corpus",
        nargs="+",
        choices=sorted(CORPORA),
        default=["tinystories"],
        help="从哪些语料抽样文档（可多选）",
    )
    parser.add_argument(
        "--tokenizer",
        nargs="+",
        choices=sorted(TOKENIZERS),
        default=["tinystories"],
        help="用哪些已训练 tokenizer（可多选；需已有 artifacts/）",
    )
    parser.add_argument("--num-docs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pile-gb",
        type=float,
        default=825.0,
        help="外推用的 Pile 大小（GB，按 GiB=2^30 bytes）",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="吞吐测试前不做 warmup encode",
    )
    args = parser.parse_args()

    sampled: dict[str, list[str]] = {}
    for corpus_name in args.corpus:
        corpus_path = CORPORA[corpus_name]
        if not corpus_path.is_file():
            raise FileNotFoundError(f"Missing corpus: {corpus_path}")
        print(f"[sample] {corpus_name} <- {corpus_path} (n={args.num_docs}, seed={args.seed})")
        sampled[corpus_name] = sample_documents(corpus_path, args.num_docs, args.seed)
        preview_bytes = sum(len(doc.encode("utf-8")) for doc in sampled[corpus_name])
        print(f"         got {len(sampled[corpus_name])} docs, {preview_bytes:,} utf-8 bytes total")

    tokenizers: dict[str, Tokenizer] = {}
    for tok_name in args.tokenizer:
        artifact_dir = TOKENIZERS[tok_name]
        print(f"[load] tokenizer={tok_name} <- {artifact_dir}")
        tokenizers[tok_name] = load_tokenizer(artifact_dir)

    print("\n=== compression ratio = utf-8_bytes / num_tokens ===")
    print(f"{'corpus':<12} {'tokenizer':<12} {'bytes':>12} {'tokens':>12} {'bytes/token':>12}")
    for corpus_name, documents in sampled.items():
        for tok_name, tokenizer in tokenizers.items():
            n_bytes, n_tokens, ratio = compression_ratio(tokenizer, documents)
            print(
                f"{corpus_name:<12} {tok_name:<12} {n_bytes:>12,} {n_tokens:>12,} {ratio:>12.3f}"
            )

    pile_bytes = args.pile_gb * (1 << 30)
    print("\n=== encode throughput（讲解）===")
    print(
        "墙钟时间 wall-clock：墙上时钟走过的真实秒数（你盯着手表看到的时间）。\n"
        "  对比 CPU time：所有核加起来的计算时间；多核时 CPU time 可以大于墙钟时间。\n"
        "  我们这里用 time.perf_counter() 测的就是墙钟：从开始 encode 到结束过了多久。\n"
        "\n"
        "throughput 在测什么：\n"
        "  输入 = 文档的 UTF-8 字节；输出 = list[int]（每个 int 是词表里的 token id）。\n"
        "  计时范围 = 反复调用 tokenizer.encode(doc) 的墙钟时间（含 pre-tokenize + BPE + 查 id）。\n"
        "  不含：读盘、蓄水池抽样扫全库（那些已经在上面的 [sample] 做完了）。\n"
        "\n"
        "公式：bytes/s = (这些文档的 UTF-8 总字节) / (encode 它们的墙钟秒数)\n"
        "      MB/s    = bytes/s / 2^20（MiB/s）\n"
        "      pile_eta = 825GiB / bytes/s（按当前实现速度外推 tokenize The Pile 要多久）"
    )
    print("\n=== encode throughput（数字）===")
    print(
        f"{'corpus':<12} {'tokenizer':<12} {'bytes':>12} {'seconds':>10} "
        f"{'bytes/s':>14} {'MB/s':>10} {'pile_eta':>14}"
    )
    for corpus_name, documents in sampled.items():
        for tok_name, tokenizer in tokenizers.items():
            n_bytes, elapsed, bps = measure_throughput(
                tokenizer,
                documents,
                warmup=not args.skip_warmup,
            )
            pile_seconds = pile_bytes / bps
            mbps = bps / (1 << 20)  # MiB/s，和脚本里 GiB 口径一致
            print(
                f"{corpus_name:<12} {tok_name:<12} {n_bytes:>12,} {elapsed:>10.3f} "
                f"{bps:>14,.0f} {mbps:>10.2f} {format_duration(pile_seconds):>14}"
            )
    print(f"(pile_eta = 按 {args.pile_gb:g} GiB 外推；MB/s = MiB/s)")


if __name__ == "__main__":
    main()
