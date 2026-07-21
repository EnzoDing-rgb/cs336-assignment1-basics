#!/usr/bin/env python3
"""盯 batch-size sweep：主网格结束后补跑 B=256（若尚未完成），再画图。"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "artifacts" / "logs" / "batch_size" / "batch_size_sweep_summary.csv"
POLL = 120


def sweep_pids() -> list[str]:
    r = subprocess.run(
        ["pgrep", "-af", "scripts/sweep_batch_size.py"],
        capture_output=True,
        text=True,
    )
    alive: list[str] = []
    for line in r.stdout.splitlines():
        if "watch_batch_size" in line:
            continue
        if "sweep_batch_size.py" not in line:
            continue
        pid = line.split(None, 1)[0]
        alive.append(pid)
    return alive


def main() -> None:
    print("[watch-bs] waiting for main sweep to finish…", flush=True)
    while True:
        if SUMMARY.is_file() and not sweep_pids():
            print("[watch-bs] main sweep done", flush=True)
            break
        print(
            f"[watch-bs] waiting… summary={SUMMARY.is_file()} pids={sweep_pids()}",
            flush=True,
        )
        time.sleep(POLL)

    # 补跑 256（显存已清；mem-probe 会再确认）
    print("[watch-bs] launching B=256 follow-up…", flush=True)
    log = ROOT / "artifacts" / "logs" / "batch_size" / "sweep_batch_size_bs256.log"
    with log.open("a", encoding="utf-8") as f:
        f.write("\n# follow-up B=256\n")
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
    print(f"[watch-bs] B=256 sweep exit={proc.returncode}", flush=True)

    print("[watch-bs] plotting…", flush=True)
    subprocess.run(
        ["uv", "run", "python", "scripts/plot_batch_size.py"],
        cwd=ROOT,
        check=False,
    )
    print("[watch-bs] ALL DONE — write reports/batch_size.md next", flush=True)


if __name__ == "__main__":
    main()
