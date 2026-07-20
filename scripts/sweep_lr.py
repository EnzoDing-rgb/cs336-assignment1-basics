#!/usr/bin/env python3
"""TinyStories learning-rate sweep（作业 learning_rate / edge of stability）.

================================================================================
总表（7 档；只改 lr 高峰，其余与 tinystories_small.yaml 相同）

锁死不变：
  - warmup 爬到 lr_max，再 cosine 降到 lr_min（lr_min = lr_max/10）
  - weight_decay=0.1、betas、warmup 步数、max_iters=20000、模型与数据
  - 自变量：仅 lr_max（及成比例的 lr_min）

  tag      lr_max    lr_min     角色                      状态
  ------   --------  ---------  ------------------------  ------
  1e-4     1e-4      1e-5       偏小、稳、loss 可能偏高      待跑
  1.8e-4   1.8e-4    1.8e-5     偏小                       待跑
  3e-4     3e-4      3e-5       中段                       已有
  5.6e-4   5.6e-4    5.6e-5     冲 valid≤1.45              待跑
  1e-3     1e-3      1e-4       更猛，可能接近最佳            待跑
  1.8e-3   1.8e-3    1.8e-4     贴边 / 可能不稳              待跑
  3.2e-3   3.2e-3    3.2e-4     预期发散（edge 外侧）        待跑

已有 = artifacts/checkpoints/tinystories_small/20260720_1436/
       （就是 lr_max=3e-4 那次；final valid≈1.53）
       默认跳过不重跑。要重跑该档：--include-existing
================================================================================

单卡顺序跑。Usage（repo root）：

  uv run python scripts/sweep_lr.py
  uv run python scripts/sweep_lr.py --dry-run
  uv run python scripts/sweep_lr.py --only 5.6e-4,3.2e-3
  uv run python scripts/sweep_lr.py --include-existing
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tinystories_small.yaml"


@dataclass(frozen=True)
class LrRun:
    """一档 lr。

    例：LrRun("5.6e-4", 5.6e-4, 5.6e-5, "...", already_done=False)
      → experiment_name=tinystories_lr5.6e-4
      → checkpoint_dir=artifacts/checkpoints/tinystories_lr5.6e-4
    """

    tag: str
    lr_max: float
    lr_min: float
    role: str
    already_done: bool = False
    existing_run_dir: str | None = None


SWEEP: list[LrRun] = [
    LrRun("1e-4", 1e-4, 1e-5, "偏小、稳、loss 可能偏高"),
    LrRun("1.8e-4", 1.8e-4, 1.8e-5, "偏小"),
    LrRun(
        "3e-4",
        3e-4,
        3e-5,
        "中段",
        already_done=True,
        existing_run_dir="artifacts/checkpoints/tinystories_small/20260720_1436",
    ),
    LrRun("5.6e-4", 5.6e-4, 5.6e-5, "冲 valid≤1.45"),
    LrRun("1e-3", 1e-3, 1e-4, "更猛，可能接近最佳"),
    LrRun("1.8e-3", 1.8e-3, 1.8e-4, "贴边 / 可能不稳"),
    LrRun("3.2e-3", 3.2e-3, 3.2e-4, "预期发散（edge 外侧）"),
]


def build_cmd(run: LrRun) -> list[str]:
    name = f"tinystories_lr{run.tag}"
    ckpt = f"artifacts/checkpoints/{name}"
    # -u：子进程 stdout 行缓冲，nohup 重定向时也能马上看到 iter 日志
    return [
        "uv",
        "run",
        "python",
        "-u",
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
    if not s:
        return None
    return {p.strip() for p in s.split(",") if p.strip()}


def print_table(rows: list[LrRun]) -> None:
    print("tag       lr_max     status   role")
    print("-" * 72)
    for r in rows:
        status = "已有" if r.already_done else "待跑"
        print(f"{r.tag:<8}  {r.lr_max:<8.1e}  {status:<6}  {r.role}")
        if r.already_done and r.existing_run_dir:
            print(f"           └─ {r.existing_run_dir}/")


def main() -> None:
    # 本脚本自己的 print 也立刻进 nohup 日志
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    p = argparse.ArgumentParser(description="Sequential TinyStories LR sweep (7-point grid)")
    p.add_argument("--dry-run", action="store_true", help="只打印，不训练")
    p.add_argument("--only", type=str, default=None, help="只跑这些 tag，例：5.6e-4,3.2e-3")
    p.add_argument(
        "--include-existing",
        action="store_true",
        help="连「已有」3e-4 也重跑（默认跳过）",
    )
    args = p.parse_args()
    only = parse_only(args.only)

    print("=" * 72, flush=True)
    print("LR sweep：7 档总表（已有标出；默认只跑待跑）", flush=True)
    print("=" * 72, flush=True)
    print_table(SWEEP)

    selected = [r for r in SWEEP if only is None or r.tag in only]
    if only is not None:
        unknown = only - {r.tag for r in SWEEP}
        if unknown:
            print(f"unknown --only tags: {sorted(unknown)}", file=sys.stderr)
            sys.exit(2)

    to_run = [r for r in selected if args.include_existing or not r.already_done]
    skipped = [r for r in selected if r.already_done and not args.include_existing]

    if skipped:
        print(f"\n跳过已有 {len(skipped)} 档: {[r.tag for r in skipped]}", flush=True)
    if not to_run:
        print("没有需要新跑的档（试试 --include-existing）", flush=True)
        return

    print(f"\n将顺序执行 {len(to_run)} 档（单卡，约 {len(to_run)}×30–35min）:\n", flush=True)
    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    for i, run in enumerate(to_run, 1):
        cmd = build_cmd(run)
        print(f"[{i}/{len(to_run)}] lr_max={run.lr_max:g}  ({run.role})", flush=True)
        print(" ", " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=ROOT, env=child_env)
        print(f"[{i}/{len(to_run)}] exit_code={result.returncode}", flush=True)

    print("\n对比曲线:", flush=True)
    print("  已有: artifacts/checkpoints/tinystories_small/20260720_1436/curves/", flush=True)
    print("  新跑: artifacts/checkpoints/tinystories_lr*/**/curves/", flush=True)
    print("  日志: reports/experiment_log.md", flush=True)


if __name__ == "__main__":
    main()
