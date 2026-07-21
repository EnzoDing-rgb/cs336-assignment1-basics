#!/usr/bin/env python3
"""One PNG, two panels (train loss).

左：post_norm vs pre_norm（同色=同 LR；实线=post，虚线=pre）
右：none_norm vs pre_norm（同上）

  uv run python scripts/plot_norm_ablation.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "plots" / "norm_ablation" / "norm_ablation_summary.png"

LR_GRID = ("1.8e-4", "1.8e-3", "1.8e-2", "9e-2")
COLORS = {
    "1.8e-4": "#1f77b4",
    "1.8e-3": "#2ca02c",
    "1.8e-2": "#ff7f0e",
    "9e-2": "#d62728",
}


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


def load_train_curve(metrics: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    with open(metrics, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("train_loss"):
                continue
            try:
                loss = float(row["train_loss"])
            except ValueError:
                continue
            if loss != loss:  # NaN
                steps.append(int(float(row["step"])))
                losses.append(float("nan"))
                break
            steps.append(int(float(row["step"])))
            losses.append(loss)
            if loss > 20.0:
                break
    return steps, losses


def plot_panel(ax, *, variant: str, title: str) -> None:
    """variant is post_norm or none_norm; always overlay pre_norm at each LR."""
    for tag in LR_GRID:
        color = COLORS[tag]
        for placement, style, width in (
            ("pre_norm", "--", 1.8),
            (variant, "-", 2.0),
        ):
            name = f"tinystories_{placement}_lr{tag}"
            m = latest_metrics(name)
            if m is None:
                print(f"[plot] missing {name}")
                continue
            s, y = load_train_curve(m)
            if not s:
                print(f"[plot] empty {name}")
                continue
            label = f"{placement} @ {tag}"
            ax.plot(s, y, color=color, linestyle=style, linewidth=width, label=label)

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("train loss")
    ax.set_ylim(0.0, 22.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", ncol=1)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    plot_panel(axes[0], variant="post_norm", title="post_norm vs pre_norm")
    plot_panel(axes[1], variant="none_norm", title="none_norm vs pre_norm")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"[write] {OUT}")


if __name__ == "__main__":
    main()
