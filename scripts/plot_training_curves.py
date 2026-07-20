#!/usr/bin/env python3
"""事后重画 loss 曲线（与 train 结束时 finalize 用同一函数）。

Usage（repo root）：

  uv run python scripts/plot_training_curves.py \\
    --metrics artifacts/checkpoints/tinystories_smoke/20260720_1419/metrics.csv

输入：metrics.csv（列 step,wall_s,train_loss,valid_loss,lr）
输出：<out_dir>/loss_vs_steps.png 与 loss_vs_wallclock.png
默认 out_dir = metrics 同级的 curves/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cs336_basics.logging import plot_metrics_csv


def main() -> None:
    p = argparse.ArgumentParser(description="Plot train/valid loss from metrics.csv")
    p.add_argument("--metrics", type=Path, required=True, help="Path to metrics.csv")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <metrics_parent>/curves)",
    )
    args = p.parse_args()
    out_dir = args.out_dir or (args.metrics.parent / "curves")
    # 例：返回 (Path(".../loss_vs_steps.png"), Path(".../loss_vs_wallclock.png"))
    steps_png, wall_png = plot_metrics_csv(args.metrics, out_dir)
    print(f"wrote {steps_png}")
    print(f"wrote {wall_png}")


if __name__ == "__main__":
    main()
