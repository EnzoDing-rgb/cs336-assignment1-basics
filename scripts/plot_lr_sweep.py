#!/usr/bin/env python3
"""TinyStories 9 档 LR sweep 汇总图（与 scripts/sweep_lr.py 同一网格）。

读各档 metrics.csv，画两张图：
  1. final_valid_vs_lr.png  —— 横轴 lr_max（log），纵轴最终 valid loss（主图）
  2. valid_curves_overlay.png —— 各档 valid loss vs step 叠在一起

未完成的档自动跳过。Usage（repo root）：

  uv run python scripts/plot_lr_sweep.py
  uv run python scripts/plot_lr_sweep.py --out-dir artifacts/plots/lr_sweep
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sweep_lr import DONE_MIN_STEP, SWEEP, candidate_ckpt_dirs, last_step, latest_metrics  # noqa: E402

TARGET_VALID = 1.45


def load_valid(path: Path) -> list[tuple[int, float]]:
    valid: list[tuple[int, float]] = []
    with path.open() as f:
        next(csv.reader(f), None)
        for row in csv.reader(f):
            if not row:
                continue
            step = int(float(row[0]))
            if len(row) > 3 and row[3].strip():
                valid.append((step, float(row[3])))
    return valid


def main() -> None:
    p = argparse.ArgumentParser(description="Plot 9-point LR sweep summary")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts/plots/lr_sweep",
        help="Output directory",
    )
    p.add_argument(
        "--include-partial",
        action="store_true",
        help="连未跑满的档也画进 overlay（主图仍只用完成档）",
    )
    args = p.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded_final: list[tuple[str, float, list[tuple[int, float]]]] = []
    loaded_curves: list[tuple[str, float, list[tuple[int, float]]]] = []

    for run in SWEEP:
        best: Path | None = None
        best_s = -1
        for d in candidate_ckpt_dirs(run):
            m = latest_metrics(d)
            if m is None:
                continue
            s = last_step(m)
            if s > best_s:
                best_s = s
                best = m
        if best is None:
            print(f"skip (missing): lr={run.tag}")
            continue
        valid = load_valid(best)
        if not valid:
            print(f"skip (empty valid): lr={run.tag}  {best}")
            continue
        if best_s >= DONE_MIN_STEP:
            loaded_final.append((run.tag, run.lr_max, valid))
            loaded_curves.append((run.tag, run.lr_max, valid))
        elif args.include_partial:
            print(f"partial overlay: lr={run.tag}  step={best_s}")
            loaded_curves.append((run.tag, run.lr_max, valid))
        else:
            print(f"skip (not done): lr={run.tag}  step={best_s}")

    if len(loaded_final) < 2:
        raise SystemExit("need ≥2 completed runs to plot final_valid_vs_lr")

    lrs = [lr for _, lr, _ in loaded_final]
    finals = [v[-1][1] for _, _, v in loaded_final]
    best_i = min(range(len(finals)), key=lambda i: finals[i])
    best_tag = loaded_final[best_i][0]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(lrs, finals, "o-", color="#1f4e79", markersize=8, linewidth=2, zorder=3)
    ax.axhline(TARGET_VALID, color="#c0392b", linestyle="--", linewidth=1.2, label=f"target ≤ {TARGET_VALID}")
    ax.scatter(
        [lrs[best_i]],
        [finals[best_i]],
        s=160,
        facecolors="none",
        edgecolors="#27ae60",
        linewidths=2.2,
        zorder=4,
        label=f"best: lr={best_tag} → {finals[best_i]:.3f}",
    )
    for lr, final in zip(lrs, finals):
        ax.annotate(
            f"{final:.3f}",
            (lr, final),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#333333",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathrm{lr\_max}$ (log scale)")
    ax.set_ylabel("final valid loss @ 20k steps")
    ax.set_title(f"TinyStories LR sweep ({len(loaded_final)}/9 done): final valid vs lr_max")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig1 = out_dir / "final_valid_vs_lr.png"
    fig.savefig(fig1, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    cmap = plt.get_cmap("viridis")
    n = len(loaded_curves)
    for i, (tag, _, valid) in enumerate(loaded_curves):
        steps = [s for s, _ in valid]
        vals = [v for _, v in valid]
        color = cmap(i / max(n - 1, 1))
        lw = 2.4 if tag == best_tag else 1.4
        ax.plot(steps, vals, color=color, linewidth=lw, label=f"lr={tag}")
    ax.axhline(TARGET_VALID, color="#c0392b", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("step")
    ax.set_ylabel("valid loss")
    ax.set_title("Valid loss trajectories by lr_max")
    ax.set_ylim(1.35, 2.8)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig2 = out_dir / "valid_curves_overlay.png"
    fig.savefig(fig2, dpi=160)
    plt.close(fig)

    print("final valid @ 20k (completed only):")
    for tag, _, valid in loaded_final:
        mark = "  ← best" if tag == best_tag else ""
        print(f"  lr_max={tag:>7}  valid={valid[-1][1]:.4f}{mark}")
    print(f"wrote {fig1}")
    print(f"wrote {fig2}")


if __name__ == "__main__":
    main()
