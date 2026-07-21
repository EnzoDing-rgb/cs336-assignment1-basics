#!/usr/bin/env python3
"""End-to-end: wait for GPU quiet → sweep 4×3 → plot → release GPU."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "artifacts/logs/norm_ablation/sweep_norm_ablation_summary.csv"
LOG = ROOT / "artifacts/logs/norm_ablation/sweep_norm_ablation.log"
POLL = 60
COORD_ID = "norm-ablation"


def pgrep(substr: str) -> list[str]:
    r = subprocess.run(["pgrep", "-af", substr], capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        if "watch_norm_ablation" in line or "overnight_norm" in line:
            continue
        if substr not in line:
            continue
        out.append(line)
    return out


def heartbeat() -> None:
    subprocess.run(
        ["python3", "/root/gpu_coord/gpu_coord.py", "heartbeat", "--id", COORD_ID],
        capture_output=True,
    )


def wait_until_quiet() -> None:
    """Wait until no train / no other sweep is holding the GPU."""
    print("[e2e] waiting for current train to finish…", flush=True)
    while True:
        trains = pgrep("cs336_basics.train")
        sweeps = pgrep("scripts/sweep_norm_ablation.py")
        print(f"[e2e] trains={len(trains)} sweeps={len(sweeps)}", flush=True)
        if not trains and not sweeps:
            return
        # If only an old sweep is idle-waiting between jobs, kill it so we can take over.
        # (Active train alone → keep waiting.)
        if not trains and sweeps:
            print("[e2e] killing idle/old sweep to take over", flush=True)
            subprocess.run(["pkill", "-f", "scripts/sweep_norm_ablation.py"], check=False)
            time.sleep(5)
            continue
        heartbeat()
        time.sleep(POLL)


def summary_ok() -> bool:
    if not SUMMARY.is_file():
        return False
    text = SUMMARY.read_text(encoding="utf-8")
    # official grid tags must appear
    for tag in ("1.8e-4", "1.8e-3", "1.8e-2", "9e-2"):
        if tag not in text:
            return False
    for place in ("pre_norm", "post_norm", "none_norm"):
        if place not in text:
            return False
    return True


def main() -> None:
    wait_until_quiet()

    print("[e2e] launching official 4×3 sweep", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n# e2e relaunch {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    proc = subprocess.run(
        ["uv", "run", "python", "-u", "scripts/sweep_norm_ablation.py"],
        cwd=ROOT,
        stdout=LOG.open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    print(f"[e2e] sweep exit={proc.returncode}", flush=True)

    print("[e2e] plotting…", flush=True)
    subprocess.run(
        ["uv", "run", "python", "scripts/plot_norm_ablation.py"],
        cwd=ROOT,
        check=False,
    )

    if summary_ok():
        print("[e2e] summary looks complete", flush=True)
    else:
        print("[e2e] WARNING: summary incomplete", flush=True)

    subprocess.run(
        ["python3", "/root/gpu_coord/gpu_coord.py", "release", "--id", COORD_ID],
        capture_output=True,
    )
    subprocess.run(
        ["python3", "/root/gpu_coord/gpu_coord.py", "restore-vllm", "--id", COORD_ID],
        capture_output=True,
    )
    print("[e2e] DONE — write reports/norm_ablation.md from results", flush=True)


if __name__ == "__main__":
    main()
