"""Toy SGD learning-rate experiment (assignment: learning_rate_tuning).

Runs the handout example: minimize mean(weights^2) with decaying-lr SGD,
for lr in {1e1, 1e2, 1e3}, 10 steps each. Prints loss each step so you can
see decay vs divergence.

Usage (from repo root):

  uv run python scripts/run_lr_tuning_toy.py
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
  """Handout SGD: θ ← θ - (α / √(t+1)) * grad."""

  def __init__(self, params, lr: float = 1e-3) -> None:
    if lr < 0:
      raise ValueError(f"Invalid learning rate: {lr}")
    defaults = {"lr": lr}
    super().__init__(params, defaults)

  def step(self, closure: Optional[Callable] = None):
    loss = None if closure is None else closure()
    for group in self.param_groups:
      lr = group["lr"]
      for p in group["params"]:
        if p.grad is None:
          continue
        state = self.state[p]
        t = state.get("t", 0)
        grad = p.grad.data
        p.data -= lr / math.sqrt(t + 1) * grad
        state["t"] = t + 1
    return loss


def run_one_lr(lr: float, steps: int = 10, seed: int = 0) -> list[float]:
  torch.manual_seed(seed)
  weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
  opt = SGD([weights], lr=lr)
  losses: list[float] = []
  for _ in range(steps):
    opt.zero_grad()
    loss = (weights**2).mean()
    losses.append(float(loss.detach().cpu()))
    loss.backward()
    opt.step()
  return losses


def format_loss(loss: float, cap: float = 10000.0) -> str:
  """Human-readable loss: normal decimals, tiny → ~0, huge → >10000."""
  if not math.isfinite(loss):
    return "inf (blew up)"
  if loss > cap:
    return f"> {cap:.0f}  (too large; diverging)"
  if loss < 1e-4:
    return f"{loss:.4f}  (~0)"
  if loss < 1:
    return f"{loss:.4f}"
  return f"{loss:.2f}"


def describe_trend(losses: list[float]) -> str:
  start, end = losses[0], losses[-1]
  peak = max(losses)
  if end > start * 2 or peak > start * 10 or not math.isfinite(end):
    return "diverges (loss grows / blows up)"
  if end < start * 0.5:
    return "decays quickly"
  if end < start:
    return "decays (slower)"
  return "stays flat or drifts up"


def main() -> None:
  lrs = [10.0, 100.0, 1000.0]
  steps = 10
  print("=" * 60)
  print("Toy SGD LR tuning: loss = mean(weights^2), steps =", steps)
  print("Same init seed for fair comparison.")
  print("Loss display: normal numbers; tiny shown as ~0; huge shown as >10000.")
  print("=" * 60)

  for lr in lrs:
    losses = run_one_lr(lr, steps=steps, seed=0)
    print(f"\nlr = {lr:.0f}")
    print("-" * 40)
    for t, loss in enumerate(losses):
      print(f"  step {t:2d}:  loss = {format_loss(loss)}")
    print(f"  start → end: {format_loss(losses[0])} → {format_loss(losses[-1])}")
    print(f"  trend: {describe_trend(losses)}")

  print("\n" + "=" * 60)
  print("Deliverable hint: compare the three trends in 1–2 sentences.")
  print("=" * 60)


if __name__ == "__main__":
  main()
