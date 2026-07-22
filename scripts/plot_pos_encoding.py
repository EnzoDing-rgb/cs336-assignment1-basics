#!/usr/bin/env python3
"""RoPE vs NoPE：一张 valid-loss 学习曲线.

  uv run python scripts/plot_pos_encoding.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

_CJK = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
if Path(_CJK).is_file():
    font_manager.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "plots" / "pos_encoding" / "rope_vs_nope_valid.png"

# (experiment_name, legend, color)
RUNS = (
    ("tinystories_rope_b64_lr1.8e-3", "RoPE", "#3182bd"),
    ("tinystories_no_rope_b64_lr1.8e-3", "NoPE", "#e6550d"),
)


def latest_metrics(experiment_name: str) -> Path | None:
    ckpt = ROOT / "artifacts" / "checkpoints" / experiment_name
    if not ckpt.is_dir():
        return None
    best: Path | None = None
    best_mtime = -1.0
    for p in ckpt.glob("*/metrics.csv"):
        m = p.stat().st_mtime
        if m > best_mtime:
            best_mtime = m
            best = p
    return best


def load_valid_curve(metrics: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    with open(metrics, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("valid_loss"):
                continue
            try:
                loss = float(row["valid_loss"])
            except ValueError:
                continue
            if loss != loss:
                break
            steps.append(int(float(row["step"])))
            losses.append(loss)
    return steps, losses


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for name, label, color in RUNS:
        m = latest_metrics(name)
        if m is None:
            print(f"[plot] missing {name}")
            continue
        s, y = load_valid_curve(m)
        if not s:
            print(f"[plot] empty valid curve {name}")
            continue
        ax.plot(s, y, color=color, linewidth=2.2, label=label)
        ax.text(
            s[-1] + 250,
            y[-1],
            f"{label}\n{y[-1]:.3f}",
            color=color,
            fontsize=8,
            va="center",
            ha="left",
            clip_on=False,
        )

    ax.set_title("RoPE vs NoPE · valid loss（B=64, lr_max=1.8e-3, 20k steps）")
    ax.set_xlabel("step")
    ax.set_ylabel("valid loss")
    ax.set_xlim(0, 22000)
    ax.grid(True, color="0.85", linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"[write] {OUT}")


if __name__ == "__main__":
    main()
