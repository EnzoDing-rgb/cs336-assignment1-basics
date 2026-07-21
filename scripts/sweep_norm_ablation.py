#!/usr/bin/env python3
"""Norm ablation 4×3 sweep（官方 one-go 网格）.

  placements: pre_norm | post_norm | none_norm
  lr_max:     1.8e-4 | 1.8e-3 | 1.8e-2 | 9e-2

底座：configs/tinystories_small.yaml（B=32, 20k, grad_clip=1.0）
提前停：NaN/Inf 或 train_loss>20（cs336_basics.train）
已完成 job 自动 skip；可反复启动续跑。

  uv run python scripts/sweep_norm_ablation.py
  nohup uv run python -u scripts/sweep_norm_ablation.py \\
    > artifacts/logs/norm_ablation/sweep_norm_ablation.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tinystories_small.yaml"
LOG_DIR = ROOT / "artifacts" / "logs" / "norm_ablation"

PLACEMENTS = ("pre_norm", "post_norm", "none_norm")
LR_GRID: list[tuple[str, float]] = [
    ("1.8e-4", 1.8e-4),
    ("1.8e-3", 1.8e-3),
    ("1.8e-2", 1.8e-2),
    ("9e-2", 9e-2),
]

DONE_MIN_STEP = 19800
ABORT_TRAIN_LOSS = 20.0


@dataclass(frozen=True)
class Job:
    placement: str
    tag: str
    lr_max: float

    @property
    def experiment_name(self) -> str:
        return f"tinystories_{self.placement}_lr{self.tag}"

    @property
    def ckpt_dir(self) -> Path:
        return ROOT / "artifacts" / "checkpoints" / self.experiment_name

    @property
    def lr_min(self) -> float:
        return self.lr_max / 10.0


def latest_metrics(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.is_dir():
        return None
    best: Path | None = None
    best_mtime = -1.0
    for p in ckpt_dir.glob("*/metrics.csv"):
        m = p.stat().st_mtime
        if m > best_mtime:
            best_mtime = m
            best = p
    return best


def last_train_step_and_loss(metrics: Path) -> tuple[int, float | None]:
    last_step = 0
    last_loss: float | None = None
    with open(metrics, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("train_loss"):
                continue
            last_step = int(float(row["step"]))
            try:
                last_loss = float(row["train_loss"])
            except ValueError:
                last_loss = float("nan")
    return last_step, last_loss


def is_finished(job: Job) -> bool:
    m = latest_metrics(job.ckpt_dir)
    if m is None:
        return False
    step, loss = last_train_step_and_loss(m)
    if step >= DONE_MIN_STEP:
        return True
    if loss is not None and (not math.isfinite(loss) or loss > ABORT_TRAIN_LOSS):
        return True
    return False


def build_cmd(job: Job) -> list[str]:
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
        f"experiment_name={job.experiment_name}",
        "--override",
        f"train.checkpoint_dir=artifacts/checkpoints/{job.experiment_name}",
        "--override",
        f"model.norm_placement={job.placement}",
        "--override",
        f"optim.lr_max={job.lr_max}",
        "--override",
        f"optim.lr_min={job.lr_min}",
    ]


def all_jobs() -> list[Job]:
    return [
        Job(placement, tag, lr)
        for placement in PLACEMENTS
        for tag, lr in LR_GRID
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Norm ablation 4×3 sweep")
    p.add_argument("--only-placements", type=str, default=None)
    p.add_argument("--only-lrs", type=str, default=None)
    args = p.parse_args()

    placements = (
        {x.strip() for x in args.only_placements.split(",") if x.strip()}
        if args.only_placements
        else set(PLACEMENTS)
    )
    lr_tags = (
        {x.strip() for x in args.only_lrs.split(",") if x.strip()}
        if args.only_lrs
        else {t for t, _ in LR_GRID}
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in all_jobs() if j.placement in placements and j.tag in lr_tags]
    print(f"[norm] official 4×3 → {len(jobs)} jobs", flush=True)

    summary_rows: list[dict[str, object]] = []
    for job in jobs:
        if is_finished(job):
            m = latest_metrics(job.ckpt_dir)
            assert m is not None
            step, loss = last_train_step_and_loss(m)
            print(f"[skip] {job.experiment_name}  step={step} train_loss={loss}", flush=True)
            summary_rows.append(
                {
                    "experiment": job.experiment_name,
                    "placement": job.placement,
                    "lr_tag": job.tag,
                    "lr_max": job.lr_max,
                    "status": "skipped_done",
                    "last_step": step,
                    "last_train_loss": loss,
                    "metrics": str(m.relative_to(ROOT)),
                }
            )
            continue

        cmd = build_cmd(job)
        print(f"[run] {job.experiment_name}", flush=True)
        print(" ".join(cmd), flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.run(cmd, cwd=ROOT, env=env)
        m = latest_metrics(job.ckpt_dir)
        step, loss = (0, None) if m is None else last_train_step_and_loss(m)
        summary_rows.append(
            {
                "experiment": job.experiment_name,
                "placement": job.placement,
                "lr_tag": job.tag,
                "lr_max": job.lr_max,
                "status": f"exit_{proc.returncode}",
                "last_step": step,
                "last_train_loss": loss,
                "metrics": "" if m is None else str(m.relative_to(ROOT)),
            }
        )
        if proc.returncode != 0:
            print(f"[warn] {job.experiment_name} exit={proc.returncode}", flush=True)

    out = LOG_DIR / "sweep_norm_ablation_summary.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    print(f"[write] {out}", flush=True)


if __name__ == "__main__":
    main()
