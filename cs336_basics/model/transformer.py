"""Full transformer language model.

本文件除整网组装外，还放仅在此处使用的：
- SwiGLU feed-forward
- Transformer block（pre_norm / post_norm / none_norm，由 config 选择）
- 位置编码（rope / no_rope，由 config 选择；no_rope → attention theta=None）
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

PosEncoding = Literal["rope", "no_rope"]
POS_ENCODINGS: tuple[str, ...] = ("rope", "no_rope")

FfnType = Literal["swiglu", "silu"]
FFN_TYPES: tuple[str, ...] = ("swiglu", "silu")


def compute_d_ff(d_model: int) -> int:
  """SwiGLU 默认：d_ff ≈ 8/3 * d_model，向上取到 64 的倍数。"""
  d_ff = int(8 / 3 * d_model)
  return ((d_ff + 63) // 64) * 64


def resolve_d_ff(
  d_model: int,
  *,
  ffn_type: FfnType,
  d_ff: int | None = None,
) -> int:
  """显式 d_ff 优先；否则 swiglu→compute_d_ff，silu→4*d_model。"""
  if d_ff is not None:
    return d_ff
  if ffn_type == "silu":
    return 4 * d_model
  return compute_d_ff(d_model)


def _validate_norm_placement(norm_placement: str) -> NormPlacement:
  if norm_placement not in NORM_PLACEMENTS:
    raise ValueError(
      f"norm_placement must be one of {NORM_PLACEMENTS}, got {norm_placement!r}"
    )
  return norm_placement  # type: ignore[return-value]


def _validate_pos_encoding(pos_encoding: str) -> PosEncoding:
  if pos_encoding not in POS_ENCODINGS:
    raise ValueError(
      f"pos_encoding must be one of {POS_ENCODINGS}, got {pos_encoding!r}"
    )
  return pos_encoding  # type: ignore[return-value]


def _validate_ffn_type(ffn_type: str) -> FfnType:
  if ffn_type not in FFN_TYPES:
    raise ValueError(f"ffn_type must be one of {FFN_TYPES}, got {ffn_type!r}")
  return ffn_type  # type: ignore[return-value]


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


class SiLUFFN(nn.Module):
  """Position-wise FFN without gating: W2 SiLU(W1 x). 消融对照用。"""

  def __init__(
    self,
    d_model: int,
    d_ff: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ) -> None:
    super().__init__()
    self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
    self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

  def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
    h = self.w1(x)
    return self.w2(h * torch.sigmoid(h))


def _make_ffn(
  d_model: int,
  d_ff: int,
  *,
  ffn_type: FfnType,
  device: torch.device | None,
  dtype: torch.dtype | None,
) -> nn.Module:
  if ffn_type == "silu":
    return SiLUFFN(d_model, d_ff, device=device, dtype=dtype)
  return SwiGLU(d_model, d_ff, device=device, dtype=dtype)


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
    theta: float | None,
    norm_placement: NormPlacement = "pre_norm",
    ffn_type: FfnType = "swiglu",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ) -> None:
    super().__init__()
    self.norm_placement = _validate_norm_placement(norm_placement)
    self.ffn_type = _validate_ffn_type(ffn_type)

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
    self.ffn = _make_ffn(
      d_model, d_ff, ffn_type=self.ffn_type, device=device, dtype=dtype
    )

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

  pos_encoding:
    rope:    各层 attention 使用 RoPE（theta=rope_theta）
    no_rope: 不注入位置编码（theta=None；仍保留 causal mask）

  ffn_type:
    swiglu: W2(SiLU(W1x) ⊙ W3x)，默认 d_ff≈8/3 d_model
    silu:   W2 SiLU(W1x)，默认 d_ff=4 d_model（对齐参数量）
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
    pos_encoding: PosEncoding = "rope",
    ffn_type: FfnType = "swiglu",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ) -> None:
    super().__init__()
    self.norm_placement = _validate_norm_placement(norm_placement)
    self.pos_encoding = _validate_pos_encoding(pos_encoding)
    self.ffn_type = _validate_ffn_type(ffn_type)
    theta: float | None = rope_theta if self.pos_encoding == "rope" else None
    self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
    self.layers = nn.ModuleList(
      [
        TransformerBlock(
          d_model,
          num_heads,
          d_ff,
          max_seq_len=context_length,
          theta=theta,
          norm_placement=self.norm_placement,
          ffn_type=self.ffn_type,
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
