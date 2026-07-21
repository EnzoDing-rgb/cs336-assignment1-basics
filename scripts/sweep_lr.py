#!/usr/bin/env python3
"""TinyStories learning-rate sweep（作业 learning_rate / edge of stability）.

================================================================================
9 档网格，一次扫完（只改 lr 高峰，其余与 tinystories_small.yaml 相同）

锁死不变：
  - warmup → cosine；lr_min = lr_max / 10
  - weight_decay=0.1、max_iters=20000、模型与数据
  - 自变量：仅 lr_max（及成比例的 lr_min）

  #  tag      lr_max
  1  1e-4     1e-4
  2  1.8e-4   1.8e-4
  3  3e-4     3e-4
  4  5.6e-4   5.6e-4
  5  1e-3     1e-3
  6  1.8e-3   1.8e-3
  7  3.2e-3   3.2e-3
  8  5.6e-3   5.6e-3
  9  1e-2     1e-2

完成 / 进行中由磁盘自动判定（metrics.csv 最后 step）：
  - step ≥ 19800 → 已完成，跳过
  - 有 metrics 但未满   → 进行中，跳过（避免双开）
  - 无 metrics         → 待跑

规范命令（repo root；中断后续跑也是这一条）：

  uv run python scripts/sweep_lr.py

  nohup uv run python -u scripts/sweep_lr.py \\
    > artifacts/logs/lr/sweep_lr.log 2>&1 &

画图：uv run python scripts/plot_lr_sweep.py
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tinystories_small.yaml"

# tinystories_small.yaml: max_iters=20000, eval_interval=600 → 最后一笔 valid 常在 19800
DONE_MIN_STEP = 19800


@dataclass(frozen=True)
class LrRun:
    """一档 lr。

    例：LrRun("5.6e-3", 5.6e-3, 5.6e-4, "…")
      → experiment_name=tinystories_lr5.6e-3
      → checkpoint_dir=artifacts/checkpoints/tinystories_lr5.6e-3
    """

    tag: str
    lr_max: float
    lr_min: float
    role: str
    # 若历史 run 不在 tinystories_lr{tag}/ 下，列出额外探测目录（相对 ROOT）
    extra_ckpt_dirs: tuple[str, ...] = ()


SWEEP: list[LrRun] = [
    LrRun("1e-4", 1e-4, 1e-5, "偏小"),
    LrRun("1.8e-4", 1.8e-4, 1.8e-5, "偏小"),
    LrRun(
        "3e-4",
        3e-4,
        3e-5,
        "中段",
        extra_ckpt_dirs=("artifacts/checkpoints/tinystories_small",),
    ),
    LrRun("5.6e-4", 5.6e-4, 5.6e-5, "擦边目标"),
    LrRun("1e-3", 1e-3, 1e-4, "达标区"),
    LrRun("1.8e-3", 1.8e-3, 1.8e-4, "甜区"),
    LrRun("3.2e-3", 3.2e-3, 3.2e-4, "甜区右侧"),
    LrRun("5.6e-3", 5.6e-3, 5.6e-4, "更大 lr"),
    LrRun("1e-2", 1e-2, 1e-3, "最大档"),
]


class RunStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"


def default_ckpt_dir(run: LrRun) -> Path:
    return ROOT / "artifacts" / "checkpoints" / f"tinystories_lr{run.tag}"


def candidate_ckpt_dirs(run: LrRun) -> list[Path]:
    dirs = [ROOT / d for d in run.extra_ckpt_dirs]
    dirs.append(default_ckpt_dir(run))
    # 去重且保序
    out: list[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out


def latest_metrics(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.is_dir():
        return None
    csvs = sorted(ckpt_dir.glob("*/metrics.csv"))
    return csvs[-1] if csvs else None


def last_step(metrics: Path) -> int:
    last = 0
    with metrics.open() as f:
        next(csv.reader(f), None)
        for row in csv.reader(f):
            if row:
                last = int(float(row[0]))
    return last


def discover(run: LrRun) -> tuple[RunStatus, Path | None, int]:
    """返回 (状态, metrics 路径或 None, last_step)。"""
    best_metrics: Path | None = None
    best_step = -1
    for d in candidate_ckpt_dirs(run):
        m = latest_metrics(d)
        if m is None:
            continue
        step = last_step(m)
        if step > best_step:
            best_step = step
            best_metrics = m
    if best_metrics is None:
        return RunStatus.PENDING, None, 0
    if best_step >= DONE_MIN_STEP:
        return RunStatus.DONE, best_metrics, best_step
    return RunStatus.IN_PROGRESS, best_metrics, best_step


def build_cmd(run: LrRun) -> list[str]:
    name = f"tinystories_lr{run.tag}"
    ckpt = f"artifacts/checkpoints/{name}"
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


def print_table(rows: list[tuple[LrRun, RunStatus, Path | None, int]]) -> None:
    print("tag       lr_max     status        step   role")
    print("-" * 78)
    for run, status, metrics, step in rows:
        step_s = str(step) if metrics is not None else "—"
        print(f"{run.tag:<8}  {run.lr_max:<8.1e}  {status.value:<12}  {step_s:<5}  {run.role}")
        if metrics is not None:
            print(f"           └─ {metrics.parent.relative_to(ROOT)}/")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    p = argparse.ArgumentParser(description="TinyStories 9-point LR sweep (one go)")
    p.add_argument("--dry-run", action="store_true", help="只打印，不训练")
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="调试用：只考虑这些 tag（规范路径不要用）",
    )
    p.add_argument(
        "--include-existing",
        action="store_true",
        help="强制重跑已完成档（进行中仍跳过，避免双开）",
    )
    args = p.parse_args()
    only = parse_only(args.only)

    selected = [r for r in SWEEP if only is None or r.tag in only]
    if only is not None:
        unknown = only - {r.tag for r in SWEEP}
        if unknown:
            print(f"unknown --only tags: {sorted(unknown)}", file=sys.stderr)
            sys.exit(2)

    discovered = [(r, *discover(r)) for r in selected]

    print("=" * 78, flush=True)
    print("LR sweep：9 档 one-go（完成/进行中由磁盘判定）", flush=True)
    print("=" * 78, flush=True)
    print_table(discovered)

    to_run: list[LrRun] = []
    skipped_done: list[str] = []
    skipped_ip: list[str] = []
    for run, status, _, _ in discovered:
        if status is RunStatus.DONE and not args.include_existing:
            skipped_done.append(run.tag)
            continue
        if status is RunStatus.IN_PROGRESS:
            skipped_ip.append(run.tag)
            continue
        # PENDING，或 DONE 且 --include-existing
        to_run.append(run)

    if skipped_done:
        print(f"\n跳过已完成 {len(skipped_done)} 档: {skipped_done}", flush=True)
    if skipped_ip:
        print(f"跳过进行中 {len(skipped_ip)} 档: {skipped_ip}", flush=True)
    if not to_run:
        print("没有待跑档（网格已齐或仅剩进行中）。", flush=True)
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

    print("\n画图: uv run python scripts/plot_lr_sweep.py", flush=True)
    print("日志: reports/experiment_log.md", flush=True)


if __name__ == "__main__":
    main()
