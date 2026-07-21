#!/usr/bin/env python3
"""Wait for norm ablation sweep → plot → remind to write report."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts/logs/norm_ablation/sweep_norm_ablation.log"
SUMMARY = ROOT / "artifacts/logs/norm_ablation/sweep_norm_ablation_summary.csv"
POLL = 120


def sweep_alive() -> bool:
    r = subprocess.run(
        ["pgrep", "-af", "scripts/sweep_norm_ablation.py"],
        capture_output=True,
        text=True,
    )
    for line in r.stdout.splitlines():
        if "watch_norm" in line:
            continue
        if "sweep_norm_ablation.py" in line:
            return True
    return False


def main() -> None:
    print("[watch-norm] waiting for sweep…", flush=True)
    while True:
        alive = sweep_alive()
        print(f"[watch-norm] alive={alive} summary={SUMMARY.is_file()}", flush=True)
        if SUMMARY.is_file() and not alive:
            # ensure summary is from the successful re-run (has skipped_done or exit_0)
            text = SUMMARY.read_text(encoding="utf-8")
            if "exit_1" in text and "exit_0" not in text and "skipped_done" not in text:
                print("[watch-norm] summary looks like failed OOM run; keep waiting", flush=True)
                time.sleep(POLL)
                continue
            break
        subprocess.run(
            ["python3", "/root/gpu_coord/gpu_coord.py", "heartbeat", "--id", "norm-ablation"],
            capture_output=True,
        )
        time.sleep(POLL)

    print("[watch-norm] plotting…", flush=True)
    subprocess.run(
        ["uv", "run", "python", "scripts/plot_norm_ablation.py"],
        cwd=ROOT,
        check=False,
    )
    print("[watch-norm] DONE — write reports/norm_ablation.md", flush=True)
    subprocess.run(
        ["python3", "/root/gpu_coord/gpu_coord.py", "release", "--id", "norm-ablation"],
        capture_output=True,
    )
    subprocess.run(
        ["python3", "/root/gpu_coord/gpu_coord.py", "restore-vllm", "--id", "norm-ablation"],
        capture_output=True,
    )


if __name__ == "__main__":
    main()
