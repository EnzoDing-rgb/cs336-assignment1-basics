#!/usr/bin/env python3
"""RoPE vs NoPE（pos_encoding）一对一消融.

协议（grilling 锁定）：
  · 底座 configs/tinystories_small.yaml
  · B=64, max_iters=20000, lr_max=1.8e-3, lr_min=1.8e-4
  · model.pos_encoding ∈ {rope, no_rope}；其余（含 pre_norm）不动
  · 两边都从零训；abort 规则由 cs336_basics.train（>200 / 非有限）

  uv run python scripts/sweep_pos_encoding.py
  nohup uv run python -u scripts/sweep_pos_encoding.py \\
    > artifacts/logs/pos_encoding/sweep_pos_encoding.log 2>&1 &
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
LOG_DIR = ROOT / "artifacts" / "logs" / "pos_encoding"

BATCH_SIZE = 64
LR_MAX = 1.8e-3
LR_MIN = LR_MAX / 10.0
LR_TAG = "1.8e-3"
POS_GRID = ("rope", "no_rope")

DONE_MIN_STEP = 19800
ABORT_TRAIN_LOSS = 200.0


@dataclass(frozen=True)
class Job:
    pos_encoding: str

    @property
    def experiment_name(self) -> str:
        return f"tinystories_{self.pos_encoding}_b64_lr{LR_TAG}"

    @property
    def ckpt_dir(self) -> Path:
        return ROOT / "artifacts" / "checkpoints" / self.experiment_name


def all_jobs() -> list[Job]:
    return [Job(p) for p in POS_GRID]


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


def run_job(job: Job) -> int:
    cmd = [
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
        f"model.pos_encoding={job.pos_encoding}",
        "--override",
        f"train.batch_size={BATCH_SIZE}",
        "--override",
        f"optim.lr_max={LR_MAX}",
        "--override",
        f"optim.lr_min={LR_MIN}",
    ]
    print(f"[run] {job.experiment_name}", flush=True)
    print(" ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "pos_encoding",
        "batch_size",
        "lr_max",
        "status",
        "last_step",
        "last_train_loss",
        "metrics",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将跑/跳过的 job，不启动训练",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    print(f"[pos] RoPE vs NoPE → {len(POS_GRID)} jobs  B={BATCH_SIZE} lr={LR_MAX}", flush=True)
    for job in all_jobs():
        if is_finished(job):
            m = latest_metrics(job.ckpt_dir)
            assert m is not None
            step, loss = last_train_step_and_loss(m)
            print(
                f"[skip] {job.experiment_name}  step={step} train_loss={loss}",
                flush=True,
            )
            summary_rows.append(
                {
                    "experiment": job.experiment_name,
                    "pos_encoding": job.pos_encoding,
                    "batch_size": BATCH_SIZE,
                    "lr_max": LR_MAX,
                    "status": "skipped_done",
                    "last_step": step,
                    "last_train_loss": loss,
                    "metrics": str(m.relative_to(ROOT)),
                }
            )
            continue

        if args.dry_run:
            print(f"[dry] would run {job.experiment_name}", flush=True)
            continue

        rc = run_job(job)
        m = latest_metrics(job.ckpt_dir)
        step, loss = (0, None) if m is None else last_train_step_and_loss(m)
        summary_rows.append(
            {
                "experiment": job.experiment_name,
                "pos_encoding": job.pos_encoding,
                "batch_size": BATCH_SIZE,
                "lr_max": LR_MAX,
                "status": f"exit_{rc}",
                "last_step": step,
                "last_train_loss": loss,
                "metrics": "" if m is None else str(m.relative_to(ROOT)),
            }
        )

    if not args.dry_run:
        write_summary(summary_rows, LOG_DIR / "sweep_pos_encoding_summary.csv")


if __name__ == "__main__":
    main()
