#!/usr/bin/env python3
"""TinyStories learning-rate sweep (assignment: learning_rate / edge of stability).

已有基线（勿重复跑，除非你想复现）:
  configs/tinystories_small.yaml
  optim.lr_max=3e-4, lr_min=3e-5
  run: artifacts/checkpoints/tinystories_small/20260720_1436/
  final valid_loss ≈ 1.53

本脚本再跑 4 档（step 数与其它超参与基线相同，只改 lr 日程高度）:

  lr_max   预期角色
  ------   --------
  1e-4     偏小、稳、loss 可能更差
  6e-4     主攻 valid ≤ 1.45
  1e-3     更猛，可能最好或开始不稳
  3e-3     预期发散（edge of stability 外侧）

单卡顺序执行（约 4×30–35min）。并行会抢同一张 GPU 显存，默认不做。

Usage（repo root）:

  uv run python scripts/sweep_lr.py
  uv run python scripts/sweep_lr.py --dry-run
  uv run python scripts/sweep_lr.py --only 6e-4,1e-3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tinystories_small.yaml"


@dataclass(frozen=True)
class LrRun:
    """一档学习率设定。

    例：LrRun("6e-4", 6e-4, 6e-5)
      → experiment_name=tinystories_lr6e-4
      → checkpoint_dir=artifacts/checkpoints/tinystories_lr6e-4
      → override optim.lr_max / optim.lr_min
    """

    tag: str
    lr_max: float
    lr_min: float
    note: str


# 与聊天里拍板的 4 档一致（基线 3e-4 已跑过，不在此列表）
SWEEP: list[LrRun] = [
    LrRun("1e-4", 1e-4, 1e-5, "smaller than baseline; expect stable, higher final loss"),
    LrRun("6e-4", 6e-4, 6e-5, "above baseline; primary candidate for valid≤1.45"),
    LrRun("1e-3", 1e-3, 1e-4, "aggressive; may be best or shaky"),
    LrRun("3e-3", 3e-3, 3e-4, "expect divergence (edge of stability)"),
]


def build_cmd(run: LrRun) -> list[str]:
    """构造一条 train 命令（list 形式，给 subprocess）。

    输入例：LrRun("6e-4", 6e-4, 6e-5, ...)
    输出例：
      ["uv", "run", "python", "-m", "cs336_basics.train",
       "--config", ".../tinystories_small.yaml",
       "--override", "experiment_name=tinystories_lr6e-4",
       "--override", "train.checkpoint_dir=artifacts/checkpoints/tinystories_lr6e-4",
       "--override", "optim.lr_max=0.0006",
       "--override", "optim.lr_min=6e-05"]
    """
    name = f"tinystories_lr{run.tag}"
    ckpt = f"artifacts/checkpoints/{name}"
    return [
        "uv",
        "run",
        "python",
        "-m",
        "cs336_basics.train",
        "--config",
        str(CONFIG),
        "--override",
        f"experiment_name={name}",
        "--override",
        f"train.checkpoint_dir={ckpt}",
        "--override",
        f"optim.lr_max={run.lr_max}",
        "--override",
        f"optim.lr_min={run.lr_min}",
    ]


def parse_only(s: str | None) -> set[str] | None:
    """--only 6e-4,1e-3 → {"6e-4", "1e-3"}；None 表示跑全部。"""
    if not s:
        return None
    return {p.strip() for p in s.split(",") if p.strip()}


def main() -> None:
    p = argparse.ArgumentParser(description="Sequential TinyStories LR sweep")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的命令，不真正训练",
    )
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="只跑部分 tag，逗号分隔，例：6e-4,3e-3",
    )
    args = p.parse_args()
    only = parse_only(args.only)

    print("=" * 72)
    print("LR sweep plan (baseline 3e-4 already done → valid≈1.53)")
    print("Fixed: max_iters=20000 and all other yaml knobs; vary lr_max/lr_min only.")
    print("Device: whatever is in the yaml (cuda:0). Sequential on one GPU.")
    print("=" * 72)

    selected = [r for r in SWEEP if only is None or r.tag in only]
    if only is not None:
        unknown = only - {r.tag for r in SWEEP}
        if unknown:
            print(f"unknown --only tags: {sorted(unknown)}", file=sys.stderr)
            sys.exit(2)
    if not selected:
        print("nothing to run", file=sys.stderr)
        sys.exit(2)

    for i, run in enumerate(selected, 1):
        cmd = build_cmd(run)
        print(f"\n[{i}/{len(selected)}] lr_max={run.lr_max:g}  lr_min={run.lr_min:g}  ({run.note})")
        print(" ", " ".join(cmd))
        if args.dry_run:
            continue
        # check=False：发散/异常退出也继续下一档，方便交「含 divergent」的曲线
        result = subprocess.run(cmd, cwd=ROOT)
        print(f"[{i}/{len(selected)}] exit_code={result.returncode}")

    print("\nDone. Compare curves under artifacts/checkpoints/tinystories_lr*/**/curves/")
    print("Baseline curves: artifacts/checkpoints/tinystories_small/20260720_1436/curves/")
    print("Also see reports/experiment_log.md for auto-appended sections.")


if __name__ == "__main__":
    main()
