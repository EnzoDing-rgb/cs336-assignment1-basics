#!/usr/bin/env python3
"""Generate TinyStories continuations over a (temperature × top-p) grid.

Locked protocol:
  - ckpt: artifacts/checkpoints/tinystories_bs128/20260721_0343/ckpt_iter5000.pt
  - prompt: Once upon a time, there was a boy named Enzo.
  - T ∈ {0, 0.5, 1.0, 1.3}, p ∈ {0.5, 0.8, 0.95}  → 12 runs
  - max_tokens=300, seed=42 (reset per run)
  - dumps → artifacts/stories/

  uv run python scripts/sweep_stories.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml

from cs336_basics.decode import EOS_TOKEN, decode
from cs336_basics.model.transformer import TransformerLM, compute_d_ff
from cs336_basics.tokenization.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CKPT = (
    ROOT
    / "artifacts/checkpoints/tinystories_bs128/20260721_0343/ckpt_iter5000.pt"
)
DEFAULT_CONFIG = (
    ROOT / "artifacts/checkpoints/tinystories_bs128/20260721_0343/run_config.yaml"
)
DEFAULT_VOCAB = ROOT / "artifacts/tinystories_bpe/vocab.json"
DEFAULT_MERGES = ROOT / "artifacts/tinystories_bpe/merges.txt"
DEFAULT_OUT = ROOT / "artifacts/stories"

PROMPT = "Once upon a time, there was a boy named Enzo."
TEMPERATURES = (0.0, 0.5, 1.0, 1.3)
TOP_PS = (0.5, 0.8, 0.95)
MAX_TOKENS = 300
SEED = 42


def load_model(ckpt_path: Path, config_path: Path, device: torch.device) -> TransformerLM:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m = cfg["model"]
    d_ff = m["d_ff"] if m.get("d_ff") is not None else compute_d_ff(int(m["d_model"]))
    model = TransformerLM(
        vocab_size=int(m["vocab_size"]),
        context_length=int(m["context_length"]),
        d_model=int(m["d_model"]),
        num_layers=int(m["num_layers"]),
        num_heads=int(m["num_heads"]),
        d_ff=int(d_ff),
        rope_theta=float(m["rope_theta"]),
        device=device,
    )
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(obj["model"])
    model.eval()
    return model


def tag(temperature: float, top_p: float) -> str:
    return f"T{temperature:g}_p{top_p:g}"


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep temperature × top-p story generation")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    p.add_argument("--merges", type=Path, default=DEFAULT_MERGES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--prompt", type=str, default=PROMPT)
    args = p.parse_args()

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] ckpt={args.ckpt}")
    model = load_model(args.ckpt, args.config, device)
    tokenizer = Tokenizer.from_files(
        str(args.vocab),
        str(args.merges),
        special_tokens=[EOS_TOKEN],
    )
    context_length = int(yaml.safe_load(args.config.read_text(encoding="utf-8"))["model"]["context_length"])
    prompt_n = len(tokenizer.encode(args.prompt))
    eos_id = tokenizer.encode(EOS_TOKEN)[0]

    rows: list[dict[str, object]] = []
    for temperature in TEMPERATURES:
        for top_p in TOP_PS:
            name = tag(temperature, top_p)
            print(f"[gen] {name} …", flush=True)
            # multinomial requires generator on the same device as probs
            g = torch.Generator(device=device)
            g.manual_seed(args.seed)
            text = decode(
                model,
                tokenizer,
                args.prompt,
                max_tokens=args.max_tokens,
                context_length=context_length,
                temperature=temperature,
                top_p=top_p,
                device=device,
                generator=g,
            )
            ids = tokenizer.encode(text)
            # decode() stops after appending EOS; encode(text) may or may not keep specials
            # depending on tokenizer — count new tokens from length delta of full encode.
            new_n = max(0, len(ids) - prompt_n)
            hit_eos = EOS_TOKEN in text or (len(ids) > prompt_n and ids[-1] == eos_id)

            txt_path = args.out_dir / f"{name}.txt"
            meta = {
                "tag": name,
                "temperature": temperature,
                "top_p": top_p,
                "seed": args.seed,
                "max_tokens": args.max_tokens,
                "prompt": args.prompt,
                "ckpt": str(args.ckpt.relative_to(ROOT)),
                "n_prompt_tokens": prompt_n,
                "n_total_tokens_encoded": len(ids),
                "n_new_tokens_approx": new_n,
                "hit_eos": hit_eos,
                "n_chars": len(text),
            }
            header = "\n".join(f"# {k}: {v}" for k, v in meta.items())
            txt_path.write_text(f"{header}\n\n{text}\n", encoding="utf-8")
            (args.out_dir / f"{name}.json").write_text(
                json.dumps({**meta, "text": text}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            rows.append(meta)
            print(
                f"[done] {name}  new_tokens≈{new_n}  hit_eos={hit_eos}  chars={len(text)}",
                flush=True,
            )

    summary_path = args.out_dir / "summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {summary_path}")
    print(f"[write] {len(rows)} stories under {args.out_dir}")


if __name__ == "__main__":
    main()
