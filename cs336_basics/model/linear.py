"""Linear transform: y = x @ W^T (+ optional bias).

和 nn.Linear 同族：继承 torch.nn.Module（不是 Parameter）。
Module 是「一层」的容器；可训练权重用 nn.Parameter 挂在 self 上。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class Linear(nn.Module):
  """Affine map on the last dimension: (..., in_features) -> (..., out_features)."""

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    bias: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ) -> None:
    super().__init__()
    std = math.sqrt(2.0 / (in_features + out_features))
    weight = torch.randn(out_features, in_features, device=device, dtype=dtype) * std
    self.weight = nn.Parameter(weight.clamp(-3 * std, 3 * std))
    if bias:
      self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
    else:
      self.register_parameter("bias", None)

  def forward(self, x: Float[Tensor, "... in_features"]) -> Float[Tensor, "... out_features"]:
    out = x @ self.weight.T
    if self.bias is not None:
      out = out + self.bias
    return out
