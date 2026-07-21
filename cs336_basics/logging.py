"""Experiment metrics: CSV + wall-clock + loss curves.

训练循环只调用 RunLogger；CSV / 画图 / 追加 experiment_log 的细节全在这里。
注意：本文件名是 cs336_basics/logging.py。若要用标准库 logging，写
  import logging as std_logging
避免 import logging 时导入本模块。
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO


# CSV 列名。一行要么填 train_loss+lr，要么填 valid_loss（另一侧留空）。
# 例：
#   step,wall_s,train_loss,valid_loss,lr
#   10,1.052502,7.747178,,2.32500000e-04
#   10,1.071643,,7.577111,
CSV_FIELDS = ("step", "wall_s", "train_loss", "valid_loss", "lr")

# 作业 experiment_log 手记；finalize 时追加一小节「事实」（路径/loss/墙钟）
DEFAULT_EXPERIMENT_LOG = Path("misc/experiment_log.md")


def make_run_dir(checkpoint_dir: str | Path, *, when: datetime | None = None) -> Path:
    """在 checkpoint_dir 下建「到分钟」的时间戳子目录。

    例：
      checkpoint_dir = Path("artifacts/checkpoints/tinystories_smoke")
      返回 → artifacts/checkpoints/tinystories_smoke/20260720_1419/
      若同分钟已存在 → .../20260720_1419_2/ （再撞则 _3 …）

    目录内稍后会有：run_config.yaml, metrics.csv, curves/, ckpt_*.pt
    """
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M")
    base = Path(checkpoint_dir)
    run_dir = base / stamp
    if run_dir.exists():
        n = 2
        while (base / f"{stamp}_{n}").exists():
            n += 1
        run_dir = base / f"{stamp}_{n}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "curves").mkdir(exist_ok=True)
    return run_dir


@dataclass
class RunLogger:
    """一个训练 run 的 metrics 记录器。

    典型用法（train.py）：
      run_dir = make_run_dir("artifacts/checkpoints/tinystories_smoke")
      logger = RunLogger.create(run_dir)          # 创建空 metrics.csv
      logger.t0 = time.perf_counter()             # 训练循环开始时重置墙钟
      logger.log_train(step=10, loss=7.75, lr=2.3e-4)
      logger.log_valid(step=10, loss=7.58)
      logger.finalize(experiment_name="tinystories_smoke")  # 画图 + 追加 experiment_log
      logger.close()
    """

    run_dir: Path
    metrics_path: Path
    t0: float
    _file: TextIO
    _writer: csv.DictWriter

    @classmethod
    def create(cls, run_dir: str | Path) -> RunLogger:
        """打开 run_dir/metrics.csv，写表头，返回 logger。

        输入：run_dir = Path(".../20260720_1419")
        输出：RunLogger；磁盘上已有 metrics.csv（仅 header）
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "curves").mkdir(exist_ok=True)
        metrics_path = run_dir / "metrics.csv"
        f = open(metrics_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()
        return cls(
            run_dir=run_dir,
            metrics_path=metrics_path,
            t0=time.perf_counter(),
            _file=f,
            _writer=writer,
        )

    def wall_s(self) -> float:
        """距 t0 的墙钟秒数。例：t0 后 1.4s → 返回 1.4"""
        return time.perf_counter() - self.t0

    def log_train(self, step: int, loss: float, lr: float) -> None:
        """追加一行训练指标。

        输入例：step=10, loss=7.747178, lr=2.325e-4
        写出例：10,1.052502,7.747178,,2.32500000e-04
        """
        self._writer.writerow(
            {
                "step": step,
                "wall_s": f"{self.wall_s():.6f}",
                "train_loss": f"{loss:.6f}",
                "valid_loss": "",
                "lr": f"{lr:.8e}",
            }
        )
        self._file.flush()

    def log_valid(self, step: int, loss: float) -> None:
        """追加一行验证指标。

        输入例：step=10, loss=7.577111
        写出例：10,1.071643,,7.577111,
        """
        self._writer.writerow(
            {
                "step": step,
                "wall_s": f"{self.wall_s():.6f}",
                "train_loss": "",
                "valid_loss": f"{loss:.6f}",
                "lr": "",
            }
        )
        self._file.flush()

    def finalize(self, *, experiment_name: str | None = None) -> None:
        """刷盘 → 画两张 PNG →（可选）往 misc/experiment_log.md 追加一小节事实。

        输出文件例：
          run_dir/curves/loss_vs_steps.png
          run_dir/curves/loss_vs_wallclock.png
        """
        self._file.flush()
        plot_metrics_csv(self.metrics_path, self.run_dir / "curves")
        print(f"[log] metrics -> {self.metrics_path}")
        print(f"[log] curves  -> {self.run_dir / 'curves'}")
        if experiment_name is not None:
            append_experiment_log_section(
                experiment_name=experiment_name,
                run_dir=self.run_dir,
                metrics_path=self.metrics_path,
            )

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def _read_series(metrics_path: Path) -> dict[str, list[float]]:
    """把 metrics.csv 拆成 train / valid 两条序列，供画图。

    输入：metrics.csv
    输出例：
      {
        "steps_t": [0, 1, ..., 19], "wall_t": [0.62, ...], "train": [9.24, ...],
        "steps_v": [5, 10, 15],     "wall_v": [0.87, ...], "valid": [8.44, ...],
      }
    """
    steps_t: list[float] = []
    wall_t: list[float] = []
    train: list[float] = []
    steps_v: list[float] = []
    wall_v: list[float] = []
    valid: list[float] = []

    with open(metrics_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = float(row["step"])
            wall = float(row["wall_s"])
            if row.get("train_loss"):
                steps_t.append(step)
                wall_t.append(wall)
                train.append(float(row["train_loss"]))
            if row.get("valid_loss"):
                steps_v.append(step)
                wall_v.append(wall)
                valid.append(float(row["valid_loss"]))

    return {
        "steps_t": steps_t,
        "wall_t": wall_t,
        "train": train,
        "steps_v": steps_v,
        "wall_v": wall_v,
        "valid": valid,
    }


def plot_metrics_csv(metrics_path: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    """读 metrics.csv，写出 loss_vs_steps.png 与 loss_vs_wallclock.png。

    输入：metrics_path=.../metrics.csv, out_dir=.../curves
    输出：(Path(.../loss_vs_steps.png), Path(.../loss_vs_wallclock.png))
    """
    import matplotlib.pyplot as plt

    metrics_path = Path(metrics_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = _read_series(metrics_path)

    def _draw(x_t, y_t, x_v, y_v, xlabel: str, title: str, out: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        if y_t:
            ax.plot(x_t, y_t, label="train", linewidth=1.5)
        if y_v:
            ax.plot(x_v, y_v, label="valid", marker="o", markersize=3, linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)

    steps_png = out_dir / "loss_vs_steps.png"
    wall_png = out_dir / "loss_vs_wallclock.png"
    _draw(
        s["steps_t"],
        s["train"],
        s["steps_v"],
        s["valid"],
        xlabel="gradient step",
        title="Loss vs steps",
        out=steps_png,
    )
    _draw(
        s["wall_t"],
        s["train"],
        s["wall_v"],
        s["valid"],
        xlabel="wall-clock (s)",
        title="Loss vs wall-clock",
        out=wall_png,
    )
    return steps_png, wall_png


def _last_losses(metrics_path: Path) -> tuple[str, str, str]:
    """从 CSV 取最后一次 train_loss / valid_loss 以及最大 wall_s。

    返回例：("7.059415", "7.244363", "1.38")
    """
    last_train, last_valid, last_wall = "—", "—", "—"
    with open(metrics_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            last_wall = row["wall_s"]
            if row.get("train_loss"):
                last_train = row["train_loss"]
            if row.get("valid_loss"):
                last_valid = row["valid_loss"]
    try:
        last_wall = f"{float(last_wall):.2f}"
    except ValueError:
        pass
    return last_train, last_valid, last_wall


def append_experiment_log_section(
    *,
    experiment_name: str,
    run_dir: Path,
    metrics_path: Path,
    log_path: Path = DEFAULT_EXPERIMENT_LOG,
) -> None:
    """往 experiment_log.md 追加一小节「机器可写的事实」；Notes 留给人填。

    故意很薄：只写路径 / loss / 墙钟，不生成长篇分析（那部分交给人/LLM）。
    """
    stamp = run_dir.name
    day = datetime.now().strftime("%Y-%m-%d")
    last_train, last_valid, wall = _last_losses(metrics_path)
    section = f"""
### {day} · {experiment_name} · {stamp}

- **Run dir:** `{run_dir.as_posix()}/`
- **Wall (train loop):** {wall} s
- **Last train_loss / valid_loss:** {last_train} / {last_valid}
- **Curves:** `{run_dir.as_posix()}/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_

"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.is_file():
        log_path.write_text(
            "# Experiment log（CS336 A1 · experiment_log）\n\n## Runs\n",
            encoding="utf-8",
        )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(section)
    print(f"[log] appended experiment section -> {log_path}")
