#!/usr/bin/env python3
"""Batch-size sweep 交付图：单张 PNG，左右两栏。

左：各 B 的 valid loss vs tokens seen（同预算学习曲线）
右：满 token 跑完的 wall time vs batch size

  uv run python scripts/plot_batch_size.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sweep_batch_size import (  # noqa: E402
    BATCH_GRID,
    CONTEXT,
    TOKEN_BUDGET,
    BatchJob,
    Phase,
    final_valid,
    iters_for_batch,
    is_done,
    last_step,
    latest_metrics,
    wall_s_final,
)

# BATCH_GRID 已含 512；未完成的档 plot 时自动 skip

OUT = ROOT / "artifacts" / "plots" / "batch_size" / "batch_size_summary.png"


def load_valid_series(metrics: Path, B: int) -> tuple[list[float], list[float]]:
    """返回 (tokens_seen, valid_loss)。"""
    tokens: list[float] = []
    vals: list[float] = []
    with metrics.open() as f:
        next(csv.reader(f), None)
        for row in csv.reader(f):
            if not row or len(row) < 4 or not row[3].strip():
                continue
            step = int(float(row[0]))
            tokens.append(step * B * CONTEXT)
            vals.append(float(row[3]))
    return tokens, vals


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    curves: list[tuple[int, list[float], list[float]]] = []
    walls: list[tuple[int, float, float]] = []  # B, wall_s, final_valid

    for B in BATCH_GRID:
        job = BatchJob(B, Phase.FULL)
        m = latest_metrics(job.ckpt_dir)
        if m is None or not is_done(job.ckpt_dir, iters_for_batch(B)):
            print(f"skip incomplete B={B}")
            continue
        tok, val = load_valid_series(m, B)
        curves.append((B, tok, val))
        w = wall_s_final(m)
        v = final_valid(m)
        if w is not None and v is not None:
            walls.append((B, w, v))
            print(f"B={B} final_valid={v:.4f} wall_s={w:.1f} steps={last_step(m)}")

    if len(curves) < 2:
        raise SystemExit("need ≥2 completed full runs")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    cmap = plt.get_cmap("viridis")
    n = len(curves)
    for i, (B, tok, val) in enumerate(curves):
        color = cmap(i / max(n - 1, 1))
        ax_l.plot([t / 1e6 for t in tok], val, color=color, linewidth=1.8, label=f"B={B}")
    ax_l.axvline(TOKEN_BUDGET / 1e6, color="#999999", linestyle=":", linewidth=1.0, alpha=0.8)
    ax_l.set_xlabel("tokens seen (×10⁶)")
    ax_l.set_ylabel("valid loss")
    ax_l.set_title("Learning curves (equal token budget)")
    ax_l.grid(True, alpha=0.35)
    ax_l.legend(fontsize=8, loc="upper right")

    Bs = [b for b, _, _ in walls]
    Ws = [w / 60.0 for _, w, _ in walls]  # minutes
    ax_r.plot(Bs, Ws, "o-", color="#1f4e79", markersize=8, linewidth=2)
    for b, wmin, v in [(b, w / 60.0, v) for b, w, v in walls]:
        ax_r.annotate(f"{wmin:.0f}m", (b, wmin), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax_r.set_xscale("log", base=2)
    ax_r.set_xticks(Bs)
    ax_r.set_xticklabels([str(b) for b in Bs])
    ax_r.set_xlabel("batch size")
    ax_r.set_ylabel("wall time to token budget (min)")
    ax_r.set_title("Wall clock vs batch size")
    ax_r.grid(True, which="both", alpha=0.35)

    fig.suptitle("TinyStories batch-size experiment", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
