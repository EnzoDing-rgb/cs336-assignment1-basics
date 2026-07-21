"""Full transformer language model.

本文件除整网组装外，还放仅在此处使用的：
- SwiGLU feed-forward
- Transformer block（pre_norm / post_norm / none_norm，由 config 选择）
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.model.attention import MultiheadSelfAttention
from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear
from cs336_basics.model.normalization import RMSNorm

NormPlacement = Literal["pre_norm", "post_norm", "none_norm"]
NORM_PLACEMENTS: tuple[str, ...] = ("pre_norm", "post_norm", "none_norm")


def compute_d_ff(d_model: int) -> int:
  """d_ff ≈ 8/3 * d_model，向上取到 64 的倍数。"""
  d_ff = int(8 / 3 * d_model)
  return ((d_ff + 63) // 64) * 64


def _validate_norm_placement(norm_placement: str) -> NormPlacement:
  if norm_placement not in NORM_PLACEMENTS:
    raise ValueError(
      f"norm_placement must be one of {NORM_PLACEMENTS}, got {norm_placement!r}"
    )
  return norm_placement  # type: ignore[return-value]


def _make_norm(
  d_model: int,
  *,
  norm_placement: NormPlacement,
  device: torch.device | None,
  dtype: torch.dtype | None,
) -> nn.Module:
  """none_norm → Identity；pre_norm / post_norm → RMSNorm。"""
  if norm_placement == "none_norm":
    return nn.Identity()
  return RMSNorm(d_model, device=device, dtype=dtype)


class SwiGLU(nn.Module):
  """Position-wise feed-forward with SwiGLU: W2(SiLU(W1 x) ⊙ W3 x).

  「position-wise」= 每个大格子自己算，不看别的大格子；
  「SwiGLU」= 具体门控公式（SiLU 门 × 另一路线性）。
  """

  def __init__(
    self,
    d_model: int,
    d_ff: int | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ) -> None:
    super().__init__()
    if d_ff is None:
      d_ff = compute_d_ff(d_model)
    self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
    self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

  def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
    w1_out = self.w1(x)
    w3_out = self.w3(x)
    hidden = (w1_out * torch.sigmoid(w1_out)) * w3_out  # SiLU(W1x) ⊙ W3x
    return self.w2(hidden)


class TransformerBlock(nn.Module):
  """Transformer block：注意力 sub-layer + 前馈 sub-layer。

  norm_placement:
    pre_norm:  x + f(RMSNorm(x))
    post_norm: RMSNorm(x + f(x))
    none_norm: x + f(x)
  """

  def __init__(
    self,
    d_model: int,
    num_heads: int,
    d_ff: int,
    *,
    max_seq_len: int,
    theta: float,
    norm_placement: NormPlacement = "pre_norm",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ) -> None:
    super().__init__()
    self.norm_placement = _validate_norm_placement(norm_placement)

    self.attn_rms_norm = _make_norm(
      d_model, norm_placement=self.norm_placement, device=device, dtype=dtype
    )
    self.attn = MultiheadSelfAttention(
      d_model,
      num_heads,
      max_seq_len=max_seq_len,
      theta=theta,
      device=device,
      dtype=dtype,
    )

    self.ffn_rms_norm = _make_norm(
      d_model, norm_placement=self.norm_placement, device=device, dtype=dtype
    )
    self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

  def forward(
    self,
    x: Float[Tensor, "batch seq d_model"],
    token_positions: Int[Tensor, "batch seq"] | None = None,
  ) -> Float[Tensor, "batch seq d_model"]:
    batch, seq_len, _ = x.shape

    if token_positions is None:
      token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)

    if self.norm_placement == "pre_norm":
      x = x + self.attn(self.attn_rms_norm(x), token_positions=token_positions)
      x = x + self.ffn(self.ffn_rms_norm(x))
    elif self.norm_placement == "post_norm":
      x = self.attn_rms_norm(x + self.attn(x, token_positions=token_positions))
      x = self.ffn_rms_norm(x + self.ffn(x))
    else:  # none_norm
      x = x + self.attn(x, token_positions=token_positions)
      x = x + self.ffn(x)

    return x


class TransformerLM(nn.Module):
  """整网：Embedding → N×Block → (可选 Final RMSNorm) → LM Head。

  none_norm：去掉 block 内与出口全部 RMSNorm。
  pre_norm / post_norm：保留出口 ln_final。
  """

  def __init__(
    self,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    *,
    norm_placement: NormPlacement = "pre_norm",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ) -> None:
    super().__init__()
    self.norm_placement = _validate_norm_placement(norm_placement)
    self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
    self.layers = nn.ModuleList(
      [
        TransformerBlock(
          d_model,
          num_heads,
          d_ff,
          max_seq_len=context_length,
          theta=rope_theta,
          norm_placement=self.norm_placement,
          device=device,
          dtype=dtype,
        )
        for _ in range(num_layers)
      ]
    )
    self.ln_final = _make_norm(
      d_model, norm_placement=self.norm_placement, device=device, dtype=dtype
    )
    self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

  def forward(
    self,
    token_ids: Int[Tensor, "batch seq"],
  ) -> Float[Tensor, "batch seq vocab_size"]:
    x = self.token_embeddings(token_ids)
    for layer in self.layers:
      x = layer(x)
    x = self.ln_final(x)
    return self.lm_head(x)
