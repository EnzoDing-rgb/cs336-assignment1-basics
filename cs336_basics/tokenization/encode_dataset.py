from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from cs336_basics.tokenization.tokenizer import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
SPECIAL_TOKEN = "<|endoftext|>"

DATASETS: dict[str, dict[str, Path]] = {
    "tinystories": {
        "train": REPO_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
        "valid": REPO_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt",
        "tokenizer": REPO_ROOT / "artifacts" / "tinystories_bpe",
        "output_dir": REPO_ROOT / "artifacts" / "tinystories_tokens",
    },
    "owt": {
        "train": REPO_ROOT / "data" / "owt_train.txt",
        "valid": REPO_ROOT / "data" / "owt_valid.txt",
        "tokenizer": REPO_ROOT / "artifacts" / "owt_bpe",
        "output_dir": REPO_ROOT / "artifacts" / "owt_tokens",
    },
}


def load_tokenizer(artifact_dir: Path) -> Tokenizer:
    vocab_path = artifact_dir / "vocab.json"
    merges_path = artifact_dir / "merges.txt"
    if not vocab_path.is_file() or not merges_path.is_file():
        raise FileNotFoundError(
            f"Missing tokenizer in {artifact_dir}. Train it first with train_bpe."
        )
    return Tokenizer.from_files(str(vocab_path), str(merges_path), special_tokens=[SPECIAL_TOKEN])


def encode_file_to_uint16(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> int:
    """流式 encode 文本文件，写成 uint16 的 .npy。

    为何 uint16：词表 10K/32K 的 id 都 < 65536，两字节够用；uint8 不够，int32 浪费一倍空间。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".bin")

    num_tokens = 0
    max_id = 0
    with open(input_path, encoding="utf-8") as text_file, open(tmp_path, "wb") as bin_file:
        for token_id in tokenizer.encode_iterable(text_file):
            if token_id < 0 or token_id > 0xFFFF:
                raise ValueError(f"token id {token_id} does not fit in uint16")
            bin_file.write(struct.pack("<H", token_id))
            num_tokens += 1
            if token_id > max_id:
                max_id = token_id
            if num_tokens % 5_000_000 == 0:
                print(f"  ... {num_tokens:,} tokens", flush=True)

    tokens = np.fromfile(tmp_path, dtype="<u2")
    np.save(output_path, tokens)
    tmp_path.unlink(missing_ok=True)

    print(
        f"[done] {input_path.name} -> {output_path} "
        f"tokens={num_tokens:,} max_id={max_id} dtype={tokens.dtype}",
        flush=True,
    )
    return num_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode a text split into a uint16 NumPy array of token ids."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument(
        "--split",
        nargs="+",
        choices=["train", "valid"],
        default=["train", "valid"],
        help="要 encode 哪些划分（默认 train+valid）",
    )
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    tokenizer = load_tokenizer(cfg["tokenizer"])
    print(f"[load] tokenizer <- {cfg['tokenizer']}")

    # 确认词表能塞进 uint16
    vocab_size = len(tokenizer.vocab)
    print(f"[check] vocab_size={vocab_size} fits_uint16={vocab_size <= 0x10000}")

    for split in args.split:
        input_path = cfg[split]
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        output_path = cfg["output_dir"] / f"{args.dataset}_{split}.npy"
        print(f"[encode] {split}: {input_path} -> {output_path}", flush=True)
        encode_file_to_uint16(tokenizer, input_path, output_path)


if __name__ == "__main__":
    main()
