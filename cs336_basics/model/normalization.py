"""RMSNorm and softmax."""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
  """Numerically stable softmax along `dim`.

  Subtracts the max along `dim` before exp so large logits do not overflow.
  Softmax is invariant to adding a constant to all entries along that dim.
  """
  x_max = torch.amax(x, dim=dim, keepdim=True)
  exp_x = torch.exp(x - x_max)
  return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


class RMSNorm(nn.Module):
  """Root mean square layer norm on the last dimension."""

  def __init__(
    self,
    d_model: int,
    eps: float = 1e-5,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ) -> None:
    super().__init__()
    self.eps = eps
    # 可学习缩放向量 γ，形状 (d_model,)，例如 (768,)；与 W1/W2/W3 无关
    self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

  def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
    in_dtype = x.dtype
    x = x.to(torch.float32)
    # rms：从当前输入 x 算出的标量尺度（每个大格子一个），形状 (..., 1)
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
    # self.weight 即缩放因子：先除以 rms，再逐维乘 γ
    result = x / rms * self.weight
    return result.to(in_dtype)
