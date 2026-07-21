#!/usr/bin/env python3
"""Overnight closer: wait for main sweep → B=256 → plot → DONE flag.

报告 Markdown 不自动套话生成；完成后写 artifacts/logs/batch_size/batch_size_DONE
供监控检测，对话里再写 reports/batch_size.md。
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "artifacts" / "logs" / "batch_size" / "batch_size_sweep_summary.csv"
DONE = ROOT / "artifacts" / "logs" / "batch_size" / "batch_size_DONE"
PLOT = ROOT / "artifacts" / "plots" / "batch_size" / "batch_size_summary.png"
POLL = 60


def pids_matching(substr: str) -> list[str]:
    r = subprocess.run(["pgrep", "-af", substr], capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        if "watch_batch_size" in line or "overnight_batch" in line:
            continue
        if substr not in line:
            continue
        out.append(line.split(None, 1)[0])
    return out


def wait_quiet(label: str) -> None:
    print(f"[overnight] waiting for {label}…", flush=True)
    while True:
        sp = pids_matching("scripts/sweep_batch_size.py")
        tr = pids_matching("cs336_basics.train")
        print(f"[overnight] {label}: sweep_pids={sp} train_n={len(tr)}", flush=True)
        if not sp and not tr:
            return
        # main sweep may still be parent waiting on child train
        if sp and not tr:
            # brief lull between jobs
            time.sleep(15)
            tr2 = pids_matching("cs336_basics.train")
            sp2 = pids_matching("scripts/sweep_batch_size.py")
            if not tr2 and not sp2:
                return
            if not tr2 and sp2:
                # parent alive between runs — keep waiting
                pass
        time.sleep(POLL)


def run_256() -> None:
    # skip if already fully done
    from sweep_batch_size import BatchJob, Phase, is_done, iters_for_batch

    job = BatchJob(256, Phase.FULL)
    if is_done(job.ckpt_dir, iters_for_batch(256)):
        print("[overnight] B=256 already done", flush=True)
        return
    print("[overnight] running B=256 follow-up", flush=True)
    log = ROOT / "artifacts" / "logs" / "batch_size" / "sweep_batch_size_bs256.log"
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n# overnight follow-up {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        proc = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-u",
                "scripts/sweep_batch_size.py",
                "--only-batches",
                "256",
            ],
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    print(f"[overnight] B=256 exit={proc.returncode}", flush=True)


def plot() -> None:
    print("[overnight] plotting", flush=True)
    subprocess.run(
        ["uv", "run", "python", "scripts/plot_batch_size.py"],
        cwd=ROOT,
        check=False,
    )


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))

    DONE.unlink(missing_ok=True)
    print("[overnight] start", flush=True)

    # Wait until main 8..128 sweep finished (summary written + no sweep proc)
    while True:
        sp = pids_matching("scripts/sweep_batch_size.py")
        print(
            f"[overnight] main: summary={SUMMARY.is_file()} sweep_pids={sp}",
            flush=True,
        )
        if SUMMARY.is_file() and not sp:
            # ensure no train left from main
            time.sleep(10)
            if not pids_matching("cs336_basics.train") and not pids_matching(
                "scripts/sweep_batch_size.py"
            ):
                break
        time.sleep(POLL)

    print("[overnight] main grid finished", flush=True)
    run_256()
    plot()

    lines = [
        f"finished_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"summary={SUMMARY}",
        f"plot={PLOT}",
        f"plot_exists={PLOT.is_file()}",
    ]
    DONE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[overnight] ALL DONE", flush=True)
    print(DONE.read_text(), flush=True)


if __name__ == "__main__":
    main()
