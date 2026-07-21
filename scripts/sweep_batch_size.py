#!/usr/bin/env python3
"""TinyStories batch-size sweep（作业 batch_size_experiment）— one-go。

协议（grill 对齐）：
  - 同 token：TOKEN_BUDGET = 20000 × 32 × 256
  - 网格：8,16,32,64,128,256（≤ mem-probe 通过的最大 B）
  - 每档从零训；B=32 也完整重跑
  - LR：lr0=1.8e-3×(B/32)；短跑 20% token × {0.5,1,2}；长跑用赢家
  - warmup/cosine 随 iters(B) 缩放；lr_min=lr_max/10
  - 日志按 token 对齐（每 32×256×600 token）
  - OOM：先 mem-probe；失败跳过更大 B
  - 命名：tinystories_bs{B} / tinystories_bs{B}_probe_{m}x

  uv run python scripts/sweep_batch_size.py
  nohup uv run python -u scripts/sweep_batch_size.py \\
    > artifacts/logs/batch_size/sweep_batch_size.log 2>&1 &
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

BASE_ITERS = 20000
BASE_BATCH = 32
CONTEXT = 256
TOKEN_BUDGET = BASE_ITERS * BASE_BATCH * CONTEXT

BASE_LR = 1.8e-3
BASE_WARMUP = 1000
BASE_LOG_EVERY_STEPS = 600
TOKENS_PER_LOG = BASE_BATCH * CONTEXT * BASE_LOG_EVERY_STEPS

BATCH_GRID = [8, 16, 32, 64, 128, 256, 512]
LR_MULTS = (0.5, 1.0, 2.0)
PROBE_FRAC = 0.20
DONE_FRAC = 0.95  # 最后一笔 log 常略早于 max_iters（按 interval 对齐）


class Phase(Enum):
    PROBE_LR = "probe"
    FULL = "full"


@dataclass(frozen=True)
class BatchJob:
    batch_size: int
    phase: Phase
    lr_mult: float | None = None

    @property
    def tag(self) -> str:
        if self.phase is Phase.PROBE_LR:
            assert self.lr_mult is not None
            m = f"{self.lr_mult:g}".replace(".", "p")
            return f"bs{self.batch_size}_probe_{m}x"
        return f"bs{self.batch_size}"

    @property
    def experiment_name(self) -> str:
        return f"tinystories_{self.tag}"

    @property
    def ckpt_dir(self) -> Path:
        return ROOT / "artifacts" / "checkpoints" / self.experiment_name


def iters_for_batch(B: int, frac: float = 1.0) -> int:
    return max(1, int(round(TOKEN_BUDGET / (B * CONTEXT) * frac)))


def lr0(B: int) -> float:
    return BASE_LR * (B / BASE_BATCH)


def log_interval_for(B: int) -> int:
    return max(1, int(round(TOKENS_PER_LOG / (B * CONTEXT))))


def warmup_for(iters: int) -> int:
    w = max(1, int(round(BASE_WARMUP * iters / BASE_ITERS)))
    return min(w, max(1, iters - 1))


def overrides_for(B: int, iters: int, lr_max: float, name: str, *, save_ckpt: bool) -> list[str]:
    log_iv = log_interval_for(B)
    if save_ckpt:
        ckpt_tokens = BASE_BATCH * CONTEXT * 6000
        ckpt_iv = max(log_iv, int(round(ckpt_tokens / (B * CONTEXT))))
    else:
        ckpt_iv = iters + 1
    return [
        f"experiment_name={name}",
        f"train.checkpoint_dir=artifacts/checkpoints/{name}",
        f"train.batch_size={B}",
        f"train.max_iters={iters}",
        f"train.eval_interval={log_iv}",
        f"train.checkpoint_interval={ckpt_iv}",
        f"logging.log_interval={log_iv}",
        f"optim.lr_max={lr_max}",
        f"optim.lr_min={lr_max / 10.0}",
        f"optim.warmup_iters={warmup_for(iters)}",
        f"optim.cosine_cycle_iters={iters}",
    ]


def build_cmd(overrides: list[str]) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "-m",
        "cs336_basics.train",
        "--config",
        str(CONFIG),
    ]
    for o in overrides:
        cmd.extend(["--override", o])
    return cmd


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


def final_valid(metrics: Path) -> float | None:
    last = None
    with metrics.open() as f:
        next(csv.reader(f), None)
        for row in csv.reader(f):
            if len(row) > 3 and row[3].strip():
                last = float(row[3])
    return last


def wall_s_final(metrics: Path) -> float | None:
    """最后一行的 wall_s（不论 train/valid）。"""
    last = None
    with metrics.open() as f:
        next(csv.reader(f), None)
        for row in csv.reader(f):
            if len(row) > 1 and row[1].strip():
                last = float(row[1])
    return last


def is_done(ckpt_dir: Path, target_iters: int) -> bool:
    m = latest_metrics(ckpt_dir)
    return m is not None and last_step(m) >= int(target_iters * DONE_FRAC)


def is_in_progress(ckpt_dir: Path, target_iters: int) -> bool:
    m = latest_metrics(ckpt_dir)
    if m is None:
        return False
    s = last_step(m)
    return 0 < s < int(target_iters * DONE_FRAC)


def run_train(overrides: list[str]) -> int:
    cmd = build_cmd(overrides)
    print(" ", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def probe_batch_fits(B: int, steps: int = 2) -> bool:
    print(f"[probe-mem] trying B={B} for {steps} steps…", flush=True)
    name = f"tinystories_bs{B}_memprobe"
    code = run_train(overrides_for(B, steps, lr0(B), name, save_ckpt=False))
    print(f"[probe-mem] B={B} → {'OK' if code == 0 else 'FAIL'}", flush=True)
    return code == 0


def select_batches(grid: list[int]) -> list[int]:
    ok: list[int] = []
    for B in grid:
        if probe_batch_fits(B):
            ok.append(B)
        else:
            print(f"[probe-mem] stop at B={B}; keeping {ok}", flush=True)
            break
    return ok


def run_lr_probes(B: int) -> float:
    results: list[tuple[float, float]] = []
    for mult in LR_MULTS:
        job = BatchJob(B, Phase.PROBE_LR, lr_mult=mult)
        target = iters_for_batch(B, PROBE_FRAC)
        lr_max = lr0(B) * mult
        if is_done(job.ckpt_dir, target):
            print(f"[probe-lr] skip done {job.experiment_name}", flush=True)
        elif is_in_progress(job.ckpt_dir, target):
            print(f"[probe-lr] in progress {job.experiment_name} — not relaunching", flush=True)
        else:
            print(
                f"[probe-lr] B={B} ×{mult:g} iters={target} lr_max={lr_max:g}",
                flush=True,
            )
            code = run_train(
                overrides_for(B, target, lr_max, job.experiment_name, save_ckpt=False)
            )
            if code != 0:
                print(f"[probe-lr] FAIL {job.experiment_name} code={code}", flush=True)
                continue
        m = latest_metrics(job.ckpt_dir)
        if m is None:
            continue
        v = final_valid(m)
        if v is None:
            continue
        results.append((v, lr_max))
        print(f"[probe-lr] {job.experiment_name} valid={v:.4f}", flush=True)

    if not results:
        raise RuntimeError(f"no successful LR probes for B={B}")
    results.sort(key=lambda t: t[0])
    best_v, best_lr = results[0]
    print(f"[probe-lr] B={B} winner lr_max={best_lr:g} (valid={best_v:.4f})", flush=True)
    out = ROOT / "artifacts" / "checkpoints" / f"tinystories_bs{B}" / "chosen_lr.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"{best_lr}\n# probe_valid={best_v}\n", encoding="utf-8")
    return best_lr


def run_full(B: int, lr_max: float) -> None:
    job = BatchJob(B, Phase.FULL)
    target = iters_for_batch(B, 1.0)
    if is_done(job.ckpt_dir, target):
        print(f"[full] skip done {job.experiment_name}", flush=True)
        return
    if is_in_progress(job.ckpt_dir, target):
        print(f"[full] in progress {job.experiment_name} — not relaunching", flush=True)
        return
    print(f"[full] B={B} iters={target} lr_max={lr_max:g}", flush=True)
    code = run_train(overrides_for(B, target, lr_max, job.experiment_name, save_ckpt=True))
    print(f"[full] B={B} exit={code}", flush=True)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    p = argparse.ArgumentParser(description="TinyStories batch-size sweep")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-mem-probe", action="store_true")
    p.add_argument("--only-batches", type=str, default=None)
    args = p.parse_args()

    grid = BATCH_GRID
    if args.only_batches:
        grid = [int(x.strip()) for x in args.only_batches.split(",") if x.strip()]

    print("=" * 72, flush=True)
    print("Batch-size sweep — equal token budget", flush=True)
    print(f"  TOKEN_BUDGET={TOKEN_BUDGET}", flush=True)
    print(f"  grid={grid}", flush=True)
    print("=" * 72, flush=True)
    for B in grid:
        it = iters_for_batch(B)
        print(
            f"  B={B:<4d} full_iters={it:<6d} lr0={lr0(B):.4g} "
            f"log_every={log_interval_for(B)} warmup={warmup_for(it)}",
            flush=True,
        )

    if args.dry_run:
        return

    batches = grid if args.skip_mem_probe else select_batches(grid)
    print(f"\nRunning batches: {batches}", flush=True)
    if not batches:
        sys.exit(1)

    summary = ROOT / "artifacts" / "logs" / "batch_size" / "batch_size_sweep_summary.csv"
    lines = ["batch_size,chosen_lr,full_valid,full_wall_s,full_steps,status"]

    for B in batches:
        try:
            lr = run_lr_probes(B)
            run_full(B, lr)
            job = BatchJob(B, Phase.FULL)
            m = latest_metrics(job.ckpt_dir)
            if m and is_done(job.ckpt_dir, iters_for_batch(B)):
                lines.append(
                    f"{B},{lr},{final_valid(m)},{wall_s_final(m)},{last_step(m)},done"
                )
            else:
                lines.append(f"{B},{lr},,,,incomplete_or_failed")
        except Exception as e:
            print(f"[error] B={B}: {e}", flush=True)
            lines.append(f"{B},,,,error:{type(e).__name__}")

    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsummary → {summary}", flush=True)
    print("plot: uv run python scripts/plot_batch_size.py", flush=True)


if __name__ == "__main__":
    main()
