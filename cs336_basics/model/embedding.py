"""Token embedding: 整数 token id → d_model 维向量（查表）。"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class Embedding(nn.Module):
  """Lookup table: token_ids (...,) -> (..., d_model)."""

  def __init__(
    self,
    vocab_size: int,
    d_model: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ) -> None:
    # weight[vc_size, d_model]  
    super().__init__()
    weight = torch.randn(vocab_size, d_model, device=device, dtype=dtype).clamp(-3, 3)
    self.weight = nn.Parameter(weight)

  def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
    return self.weight[token_ids]
