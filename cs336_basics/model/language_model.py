"""Full transformer language model.

本文件除整网组装外，还放仅在此处使用的：
- SwiGLU feed-forward
- Pre-norm transformer block（调用 linear / attention / normalization）
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from cs336_basics.model.attention import MultiheadSelfAttention
from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear
from cs336_basics.model.normalization import RMSNorm


def compute_d_ff(d_model: int) -> int:
  """d_ff ≈ 8/3 * d_model，向上取到 64 的倍数。"""
  d_ff = int(8 / 3 * d_model)
  return ((d_ff + 63) // 64) * 64


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
  """Pre-norm Transformer block：注意力 sub-layer + 前馈 sub-layer。

  全文统一数字例子：
    d_model=768, num_heads=8 → 每头 64；d_ff=2048
    batch=2, seq=3（"I love cats"），x.shape = (2, 3, 768)

  每个 sub-layer 都是：先 RMSNorm → 再主运算 → 再残差加回去。
  """

  def __init__(
    self,
    d_model: int,
    num_heads: int,
    d_ff: int,
    *,
    max_seq_len: int,
    theta: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ) -> None:
    super().__init__()

    # 注意力前的归一化（每个大格子 768 维自己归一）
    self.attn_rms_norm = RMSNorm(d_model, device=device, dtype=dtype)
    # 带 RoPE 的因果多头自注意力
    self.attn = MultiheadSelfAttention(
      d_model,
      num_heads,
      max_seq_len=max_seq_len,
      theta=theta,
      device=device,
      dtype=dtype,
    )

    # 前馈前的归一化（另一套可学习 weight，不和上面共用）
    self.ffn_rms_norm = RMSNorm(d_model, device=device, dtype=dtype)
    # 每个大格子自己做的 SwiGLU：768 → 2048 门控 → 768
    self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

  def forward(
    self,
    x: Float[Tensor, "batch seq d_model"],
    token_positions: Int[Tensor, "batch seq"] | None = None,
  ) -> Float[Tensor, "batch seq d_model"]:
    # x 例: (2, 3, 768)
    batch, seq_len, _ = x.shape

    # RoPE 需要每个大格子的座位号；没传入就按 0..seq-1 填
    # 例: token_positions = [[0,1,2],[0,1,2]]  shape (2, 3)
    if token_positions is None:
      token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)

    # ── sub-layer 1：注意力 ──
    # y1 = x + MultiHeadSelfAttention(RMSNorm(x))
    #
    # attn_rms_norm(x): (2, 3, 768) → (2, 3, 768)  幅度稳住
    # attn(...):        大格子之间换信息，输出仍 (2, 3, 768)
    # 残差 +x:          把原始 x 加回去，信息高速公路
    x = x + self.attn(self.attn_rms_norm(x), token_positions=token_positions)

    # ── sub-layer 2：前馈 ──
    # y = y1 + SwiGLU(RMSNorm(y1))
    #
    # ffn_rms_norm(x): (2, 3, 768) → (2, 3, 768)
    # ffn(...):        每个大格子独自 768→2048 门控→768，互不 attend
    # 残差:            再加回进入本 sub-layer 前的 x
    x = x + self.ffn(self.ffn_rms_norm(x))

    # 输出仍是 (2, 3, 768)，交给下一层 block 或最终归一化
    return x


class TransformerLM(nn.Module):
  """整网：Embedding → N×Block → Final RMSNorm → LM Head。

  例子：vocab=10000, d_model=768, num_layers=2, batch=2, seq=3
    token_ids (2, 3) → logits (2, 3, 10000)
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
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ) -> None:
    super().__init__()
    # embedding_weight 就挂在这里：形状 (vocab_size, d_model)，前向按 id 取行
    self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
    self.layers = nn.ModuleList(
      [
        TransformerBlock(
          d_model,
          num_heads,
          d_ff,
          max_seq_len=context_length,
          theta=rope_theta,
          device=device,
          dtype=dtype,
        )
        for _ in range(num_layers)
      ]
    )
    self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
    # LM Head：就是一层线性，768 → vocab_size；weight 形状 (vocab_size, 768)
    self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

  def forward(
    self,
    token_ids: Int[Tensor, "batch seq"],
  ) -> Float[Tensor, "batch seq vocab_size"]:
    # ① 查表：token_ids (2, 3) → x (2, 3, 768)
    x = self.token_embeddings(token_ids)
    # ② 叠 N 层 block，形状始终 (2, 3, 768)
    for layer in self.layers:
      x = layer(x)
    # ③ 出口 RMSNorm
    x = self.ln_final(x)
    # ④ LM Head → logits (2, 3, vocab_size)
    return self.lm_head(x)
