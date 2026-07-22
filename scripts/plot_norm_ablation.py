#!/usr/bin/env python3
"""One PNG, two panels — train loss curves for norm placement ablation.

编码（刻意不用「彩虹 = LR」）：
  · 色系 = placement：蓝 pre_norm / 橙 post_norm / 红 none_norm
  · 同色系深浅 = lr_max（浅→深：1.8e-4 → 9e-2）
  · 线型 = 结局：实线=收得还行；虚线=跑满但明显差；炸点=粗短线+▲+标注框

  uv run python scripts/plot_norm_ablation.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

_CJK = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
if Path(_CJK).is_file():
    font_manager.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
    plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "plots" / "norm_ablation" / "norm_ablation_summary.png"

LR_GRID = ("1.8e-4", "1.8e-3", "1.8e-2", "9e-2")

# placement → 固定色系；同系 4 档浅→深对应 LR
PLACEMENT_COLORS: dict[str, dict[str, str]] = {
    "pre_norm": {  # 蓝：基准 / 好
        "1.8e-4": "#9ecae1",
        "1.8e-3": "#6baed6",
        "1.8e-2": "#3182bd",
        "9e-2": "#08519c",
    },
    "post_norm": {  # 橙：中间档
        "1.8e-4": "#fdd0a2",
        "1.8e-3": "#fdae6b",
        "1.8e-2": "#e6550d",
        "9e-2": "#a63603",
    },
    "none_norm": {  # 红：高风险
        "1.8e-4": "#fcbba1",
        "1.8e-3": "#fc9272",
        "1.8e-2": "#de2d26",
        "9e-2": "#a50f15",
    },
}

ABORT_LOSS = 200.0
# 跑满但 final train loss 高于此 → 虚线（「还行/蹲死」）
STUCK_LOSS = 4.0
YLIM = (0.0, 10.5)
DISPLAY_CLIP = YLIM[1]


def latest_metrics(experiment_name: str) -> Path | None:
    ckpt = ROOT / "artifacts" / "checkpoints" / experiment_name
    if not ckpt.is_dir():
        return None
    best: Path | None = None
    best_mtime = -1.0
    for p in ckpt.glob("*/metrics.csv"):
        m = p.stat().st_mtime
        if m > best_mtime:
            best_mtime = m
            best = p
    return best


def load_train_curve(metrics: Path) -> tuple[list[int], list[float], bool]:
    steps: list[int] = []
    losses: list[float] = []
    aborted = False
    with open(metrics, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("train_loss"):
                continue
            try:
                loss = float(row["train_loss"])
            except ValueError:
                continue
            if loss != loss:  # NaN
                steps.append(int(float(row["step"])))
                losses.append(float("nan"))
                aborted = True
                break
            steps.append(int(float(row["step"])))
            losses.append(loss)
            if loss > ABORT_LOSS:
                aborted = True
                break
    return steps, losses, aborted


def clip_for_display(
    steps: list[int], losses: list[float], *, clip: float
) -> tuple[list[int], list[float]]:
    s_out: list[int] = []
    y_out: list[float] = []
    for s, y in zip(steps, losses):
        if y != y:
            s_out.append(s)
            y_out.append(y)
            return s_out, y_out
        if y > clip:
            s_out.append(s)
            y_out.append(clip)
            if y > ABORT_LOSS:
                return s_out, y_out
            continue
        s_out.append(s)
        y_out.append(y)
    return s_out, y_out


def fmt_loss(loss: float) -> str:
    if loss != loss:
        return "NaN"
    if loss >= 1.0e6:
        return f"{loss:.1e}"
    if loss >= 100.0:
        return f"{loss:.0f}"
    return f"{loss:.2f}"


def style_for_outcome(*, aborted: bool, final_loss: float) -> tuple[str, float]:
    """Returns (linestyle, linewidth)."""
    if aborted:
        return "-", 2.8
    if final_loss > STUCK_LOSS:
        return "--", 2.0
    return "-", 1.9


def plot_panel(
    ax,
    *,
    variant: str,
    title: str,
    ylim: tuple[float, float],
    clip: float | None = None,
    annotate_blasts: bool = False,
    annotate_stuck: bool = False,
) -> None:
    blast_text_slots = {
        "9e-2": (2400, 9.15),
        "1.8e-2": (5600, 7.85),
        "1.8e-3": (9800, 6.55),
    }

    # pre 先画、variant 后画，variant 压在上面更醒目
    for tag in LR_GRID:
        for placement in ("pre_norm", variant):
            name = f"tinystories_{placement}_lr{tag}"
            m = latest_metrics(name)
            if m is None:
                print(f"[plot] missing {name}")
                continue
            s_raw, y_raw, aborted = load_train_curve(m)
            if not s_raw:
                print(f"[plot] empty {name}")
                continue

            s, y = s_raw, y_raw
            if clip is not None:
                s, y = clip_for_display(s_raw, y_raw, clip=clip)

            color = PLACEMENT_COLORS[placement][tag]
            ls, lw = style_for_outcome(aborted=aborted, final_loss=y_raw[-1])
            # pre 略细一点，当参照；variant 更粗
            if placement == "pre_norm" and not aborted:
                lw = max(1.4, lw - 0.4)

            label = f"{placement} @ {tag}"
            ax.plot(s, y, color=color, linestyle=ls, linewidth=lw, label=label, zorder=3)

            # 跑满的曲线：末端标出具体 lr（炸点标注框里已有 lr，不再重复）
            if not aborted:
                lr_y_nudge = {
                    "1.8e-4": -0.28,
                    "1.8e-3": -0.05,
                    "1.8e-2": 0.22,
                    "9e-2": 0.55,
                }[tag]
                # pre / variant 左右错开，避免两族末端叠字
                x_nudge = 280 if placement == "pre_norm" else 1100
                if placement == variant:
                    lr_y_nudge += 0.12
                ax.text(
                    s[-1] + x_nudge,
                    min(ylim[1] - 0.2, max(ylim[0] + 0.15, y[-1] + lr_y_nudge)),
                    f"lr={tag}",
                    color=color,
                    fontsize=7,
                    va="center",
                    ha="left",
                    clip_on=False,
                    zorder=5,
                )

            if aborted and annotate_blasts:
                step_b = s_raw[-1]
                loss_b = y_raw[-1]
                ax.scatter(
                    [s[-1]],
                    [ylim[1]],
                    color=color,
                    marker="^",
                    s=90,
                    zorder=6,
                    clip_on=False,
                    edgecolors="black",
                    linewidths=0.6,
                )
                tx, ty = blast_text_slots.get(tag, (step_b + 1500, 8.5))
                ax.annotate(
                    f"{placement}@{tag}\n炸 @ step {step_b}\nloss → {fmt_loss(loss_b)}",
                    xy=(s[-1], ylim[1]),
                    xytext=(tx, ty),
                    fontsize=7.5,
                    color=color,
                    ha="left",
                    va="top",
                    fontweight="bold",
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        lw=1.4,
                        connectionstyle="arc3,rad=0.12",
                    ),
                    bbox=dict(
                        boxstyle="round,pad=0.28",
                        facecolor="white",
                        edgecolor=color,
                        alpha=0.95,
                        linewidth=1.2,
                    ),
                    zorder=7,
                )

            if (
                annotate_stuck
                and placement == "post_norm"
                and tag == "9e-2"
                and not aborted
            ):
                ax.annotate(
                    f"post@9e-2 蹲死 ~{y_raw[-1]:.1f}\n跑满 20k · loss 卡在高位",
                    xy=(s[-1], y[-1]),
                    xytext=(10800, 7.5),
                    fontsize=7.5,
                    color=color,
                    ha="left",
                    va="top",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                    bbox=dict(
                        boxstyle="round,pad=0.28",
                        facecolor="white",
                        edgecolor=color,
                        alpha=0.95,
                    ),
                    zorder=7,
                )

    if clip is not None:
        ax.axhline(clip, color="0.65", linestyle=":", linewidth=1.0, zorder=1)
        ax.text(
            200,
            clip - 0.35,
            f"显示截断 = {clip:g}",
            fontsize=7,
            color="0.4",
            va="top",
        )

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("train loss")
    ax.set_ylim(*ylim)
    ax.set_xlim(0, 23500)  # 给末端 lr=… 标签留空
    ax.grid(True, color="0.85", linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)


def _legend_handles() -> list[Line2D]:
    """色系=placement；四档色块直接写出 lr；线型=结局。"""
    handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            color=PLACEMENT_COLORS["pre_norm"]["1.8e-3"],
            lw=2.4,
            label="pre_norm（蓝）",
        ),
        Line2D(
            [0],
            [0],
            color=PLACEMENT_COLORS["post_norm"]["1.8e-3"],
            lw=2.4,
            label="post_norm（橙）",
        ),
        Line2D(
            [0],
            [0],
            color=PLACEMENT_COLORS["none_norm"]["1.8e-3"],
            lw=2.4,
            label="none_norm（红）",
        ),
        Line2D([0], [0], color="0.35", lw=1.8, linestyle="-", label="实线：收得还行"),
        Line2D(
            [0],
            [0],
            color="0.35",
            lw=1.8,
            linestyle="--",
            label=f"虚线：跑满但 loss>{STUCK_LOSS:g}",
        ),
        Line2D(
            [0],
            [0],
            color=PLACEMENT_COLORS["none_norm"]["9e-2"],
            lw=2.6,
            marker="^",
            markevery=[0],
            markersize=8,
            label="▲：炸（见标注框）",
        ),
    ]
    # 用蓝色系四档明示 lr（橙/红同一深浅阶）
    for tag in LR_GRID:
        handles.append(
            Line2D(
                [0],
                [0],
                color=PLACEMENT_COLORS["pre_norm"][tag],
                lw=2.2,
                label=f"lr={tag}",
            )
        )
    return handles


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True)
    plot_panel(
        axes[0],
        variant="post_norm",
        title="post_norm vs pre_norm",
        ylim=YLIM,
        annotate_stuck=True,
    )
    plot_panel(
        axes[1],
        variant="none_norm",
        title="none_norm vs pre_norm",
        ylim=YLIM,
        clip=DISPLAY_CLIP,
        annotate_blasts=True,
    )

    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "色系=placement（蓝 pre / 橙 post / 红 none）· 同色系深浅与末端标签 = lr",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"[write] {OUT}")


if __name__ == "__main__":
    main()
