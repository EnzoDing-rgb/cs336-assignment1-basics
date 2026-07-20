"""RoPE θ(pos, k) trend figure with a readable caption under the plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

THETA = 10_000.0
D_K = 64
OUT = Path(__file__).resolve().parent / "rope_theta_trend.png"


def rope_angle(pos: np.ndarray, k: np.ndarray) -> np.ndarray:
  return pos[:, None] * (THETA ** (-(2 * k[None, :]) / D_K))


CAPTION = """
How to read these two panels
• Left: color = rotation angle θ(pos, k). Horizontal axis = big-cell seat pos (0 = first token, …).
  Vertical axis = small-cell pair index k. Each head has 64 small-cells, grouped as 32 pairs:
  pair k uses small-cells (2k, 2k+1). Brighter = larger θ = that pair is twisted harder.
• Right: same formula, but only a few curves. For any fixed k, θ grows linearly with pos.
  Steeper curve = that pair reacts more strongly when the seat moves one step.

What changes with position (the question you care about)
• Bigger pos (later big-cell) → larger θ → rotate MORE. Seat 0 → θ = 0 → no rotation.
• For the SAME pos: smaller k (earlier small-cell pair, e.g. cells 0–1) → larger θ → rotate MORE.
  Larger k (later pair, e.g. cells 62–63) → tiny θ → barely rotates.
• So “smaller index earns more” is only true for small-cell pair index k, NOT for big-cell pos.
  Earlier seats rotate less; earlier pairs (at a fixed seat) rotate more.

Why do earlier small-cell pairs rotate more? Isn’t every pair “just another slot”?
• Before RoPE, you are right: the 32 pairs are interchangeable slots — nothing intrinsic says
  pair 0 should care about position more than pair 31.
• RoPE deliberately ASSIGNS them different angular speeds via θ = pos / Θ^(2k/d_k).
  Small k → fast spin; large k → slow spin. The ordering is a convention of this formula;
  what matters is having BOTH fast and slow pairs, not that “low slots are semantically special”.

Why we want fast + slow pairs at all (intuition)
• Attention after RoPE mainly feels RELATIVE seat difference (how far apart two big-cells are).
• Fast pairs (small k): one seat step changes the angle a lot → good at telling nearby seats apart
  (3 vs 4), but the angle wraps / blurs over long distances — a fine ruler.
• Slow pairs (large k): nearby seats look almost the same, but distance accumulates over long range
  — a coarse ruler.
• Together they give multi-scale position cues (same idea as Fourier / sinusoidal features):
  fine + coarse rulers, not “early slots are more important content”.
""".strip()


def main() -> None:
  pos = np.arange(0, 32)
  k = np.arange(0, 32)
  angles = rope_angle(pos, k)

  fig = plt.figure(figsize=(11.5, 9.2))
  gs = GridSpec(2, 2, height_ratios=[1.05, 1.15], hspace=0.35, wspace=0.28)

  ax0 = fig.add_subplot(gs[0, 0])
  ax1 = fig.add_subplot(gs[0, 1])
  ax_cap = fig.add_subplot(gs[1, :])
  ax_cap.axis("off")

  im = ax0.imshow(
    angles.T,
    origin="lower",
    aspect="auto",
    cmap="magma",
    extent=(-0.5, 31.5, -0.5, 31.5),
  )
  ax0.set_xlabel("pos — big-cell seat (earlier → later in the sentence)")
  ax0.set_ylabel("k — small-cell pair index\n(earlier pairs → later pairs)")
  ax0.set_title(r"Heatmap of rotation angle $\theta(\mathrm{pos}, k)$")
  cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
  cbar.set_label(r"$\theta$ (radians); brighter = rotate more")

  demo_ks = [0, 4, 8, 16, 31]
  for kk in demo_ks:
    ax1.plot(pos, angles[:, kk], marker="o", markersize=3, label=rf"pair $k={kk}$")
  ax1.set_xlabel("pos — big-cell seat")
  ax1.set_ylabel(r"$\theta$ (radians)")
  ax1.set_title("θ vs pos for a few small-cell pairs")
  ax1.legend(title="which pair", fontsize=8)
  ax1.grid(True, alpha=0.3)

  fig.suptitle(
    rf"RoPE schedule  $\theta(\mathrm{{pos}},k)=\mathrm{{pos}}\cdot\Theta^{{-2k/d_k}}$"
    rf"   ($\Theta={THETA:g}$, $d_k={D_K}$, 32 pairs)",
    fontsize=12,
    y=0.98,
  )

  ax_cap.text(
    0.0,
    1.0,
    CAPTION,
    transform=ax_cap.transAxes,
    va="top",
    ha="left",
    family="DejaVu Sans",
    fontsize=8.2,
    linespacing=1.35,
    wrap=False,
  )

  fig.savefig(OUT, dpi=160, bbox_inches="tight")
  plt.close(fig)
  print(f"wrote {OUT}")


if __name__ == "__main__":
  main()
