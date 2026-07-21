#!/usr/bin/env python3
"""盯完第 8 档 (5.6e-3) → 确认/拉起第 9 档 (1e-2) → 跑完后重画图。

不杀现有训练。若父 sweep 会自动开第 9 档则只等待；否则自行
`uv run python scripts/sweep_lr.py`（会跳过 done / in_progress）。

Usage（repo root）：

  nohup uv run python -u scripts/watch_lr_finish.py \\
    > artifacts/watch_lr_finish.log 2>&1 &
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sweep_lr import DONE_MIN_STEP, RunStatus, discover  # noqa: E402
from sweep_lr import SWEEP  # noqa: E402

POLL_S = 60
# 第 8 档结束后，给父进程一点时间去开第 9 档
HANDOFF_WAIT_S = 180


def run_by_tag(tag: str):
    return next(r for r in SWEEP if r.tag == tag)


def status_line(tag: str) -> str:
    st, metrics, step = discover(run_by_tag(tag))
    where = metrics.parent.name if metrics else "—"
    return f"{tag}: {st.value} step={step} dir={where}"


def wait_until(tag: str, want: RunStatus, label: str) -> None:
    print(f"[watch] waiting for {label} ({tag} → {want.value}) …", flush=True)
    while True:
        st, _, step = discover(run_by_tag(tag))
        print(f"[watch] {status_line(tag)}", flush=True)
        if st is want:
            print(f"[watch] OK: {tag} is {want.value}", flush=True)
            return
        if want is RunStatus.DONE and st is RunStatus.DONE:
            return
        time.sleep(POLL_S)


def ensure_ninth_started() -> None:
    """第 8 档已完成后：等父进程开第 9，否则自己拉起。"""
    st8, _, _ = discover(run_by_tag("5.6e-3"))
    if st8 is not RunStatus.DONE:
        raise RuntimeError("ensure_ninth_started called before 8th done")

    st9, _, _ = discover(run_by_tag("1e-2"))
    if st9 in (RunStatus.IN_PROGRESS, RunStatus.DONE):
        print(f"[watch] 9th already {st9.value}, no launch", flush=True)
        return

    print(f"[watch] 9th pending; wait up to {HANDOFF_WAIT_S}s for parent sweep…", flush=True)
    deadline = time.time() + HANDOFF_WAIT_S
    while time.time() < deadline:
        st9, _, step = discover(run_by_tag("1e-2"))
        print(f"[watch] handoff check: {status_line('1e-2')}", flush=True)
        if st9 in (RunStatus.IN_PROGRESS, RunStatus.DONE):
            print("[watch] parent started 9th", flush=True)
            return
        time.sleep(30)

    print("[watch] parent did not start 9th; launching sweep_lr.py", flush=True)
    cmd = ["uv", "run", "python", "-u", "scripts/sweep_lr.py"]
    log = ROOT / "artifacts" / "sweep_lr.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write("\n# launched by watch_lr_finish.py for 9th\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    print(f"[watch] launched pid={proc.pid} log={log}", flush=True)

    # 等到出现 in_progress / done，或超时报错
    deadline = time.time() + 300
    while time.time() < deadline:
        st9, _, _ = discover(run_by_tag("1e-2"))
        if st9 in (RunStatus.IN_PROGRESS, RunStatus.DONE):
            print(f"[watch] 9th started: {st9.value}", flush=True)
            return
        time.sleep(15)
    raise RuntimeError("failed to start 9th within 5 min")


def plot() -> None:
    print("[watch] plotting…", flush=True)
    subprocess.run(
        ["uv", "run", "python", "scripts/plot_lr_sweep.py"],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    print("[watch] start", flush=True)
    print(f"[watch] DONE_MIN_STEP={DONE_MIN_STEP}", flush=True)
    print(f"[watch] {status_line('5.6e-3')}", flush=True)
    print(f"[watch] {status_line('1e-2')}", flush=True)

    wait_until("5.6e-3", RunStatus.DONE, "8th complete")
    ensure_ninth_started()
    wait_until("1e-2", RunStatus.DONE, "9th complete")
    plot()

    print("[watch] ALL DONE", flush=True)
    print(f"[watch] {status_line('5.6e-3')}", flush=True)
    print(f"[watch] {status_line('1e-2')}", flush=True)
    print("[watch] figures → artifacts/plots/lr_sweep/", flush=True)


if __name__ == "__main__":
    main()
