"""Full transformer language model.

本文件除整网组装外，还放仅在此处使用的：
- SwiGLU feed-forward
- Pre-norm transformer block（调用 linear / attention / normalization）
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from cs336_basics.model.linear import Linear


def compute_d_ff(d_model: int) -> int:
  """d_ff ≈ 8/3 * d_model，向上取到 64 的倍数。"""
  d_ff = int(8 / 3 * d_model)
  return ((d_ff + 63) // 64) * 64


class PointWise_FFN(nn.Module):
  """Position-wise SwiGLU FFN: W2(SiLU(W1 x) ⊙ W3 x)."""

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


class RoPE(nn.Module):
  """按 token 在句中的 pos，对 Q/K（已是 d_k 维）做逐对 2D 旋转。

  全文例子（用小数字演示，真实模型 d_k 通常是 64）：
    句子:  I    love  this  cute  cat
    pos:   0     1     2     3     4
    theta=10000, d_k=4, max_seq_len=5
  """

  def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
    super().__init__()
    self.theta = theta
    self.d_k = d_k

    # pair_idx：告诉公式「第 k 对」用哪个指数。长度 = d_k/2
    # 例子 d_k=4 → pair_idx = tensor([0., 2.])，对应 k=0 和 k=1 两对
    pair_idx = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)

    # inv_freq[k] = 1 / Θ^(2k/d_k)，第 k 对转多「快」
    # 例子 theta=10000, d_k=4:
    #   inv_freq = tensor([1.0000, 0.0100])
    #   k=0: 1/10000^0 = 1
    #   k=1: 1/10000^0.5 = 0.01
    # shape: (d_k/2,) → 例子里是 (2,)
    inv_freq = 1.0 / (theta ** (pair_idx / d_k))

    # pos：句子里可能出现的位置编号，从 0 到 max_seq_len-1
    # 例子 max_seq_len=5 → pos = tensor([0., 1., 2., 3., 4.])
    # shape: (max_seq_len,) → 例子里是 (5,)
    pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)

    # angles[pos, k] = pos * inv_freq[k] = θ(pos, k)
    # shape: (max_seq_len, d_k/2) → 例子里是 (5, 2)
    # 完整表（例子）:
    #   pos=0 → [0.00,  0.000]
    #   pos=1 → [1.00,  0.010]
    #   pos=2 → [2.00,  0.020]
    #   pos=3 → [3.00,  0.030]
    #   pos=4 → [4.00,  0.040]   ← "cat" 那一行
    angles = pos[:, None] * inv_freq[None, :]

    # cos_cached / sin_cached：把上面每个角度先算好，forward 里只查表
    # shape 同 angles → 例子 (5, 2)
    # cos_cached[4] = tensor([cos(4.0), cos(0.04)]) ≈ tensor([-0.6536,  0.9992])
    # sin_cached[4] = tensor([sin(4.0), sin(0.04)]) ≈ tensor([-0.7568,  0.0400])
    self.register_buffer("cos_cached", angles.cos(), persistent=False)
    self.register_buffer("sin_cached", angles.sin(), persistent=False)

  def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    # ── 输入 x ──
    # 某一个 head 的 Q 或 K，最后一维已经是 d_k（W_Q/W_K 在别处算完了）
    # 例子 batch=2, 5 个词, d_k=4 → x.shape = (2, 5, 4)
    #
    # x[0, 4, :] = "cat"(pos=4) 在 batch0 里的 q 向量，比如 tensor([3., 4., 5., 6.])
    #               拆开就是 (a₀,b₀)=(3,4), (a₁,b₁)=(5,6)

    # ── 输入 token_positions ──
    # 每个 seq 格子填「这个词是句中第几个」
    # 例子 → tensor([[0, 1, 2, 3, 4],
    #                [0, 1, 2, 3, 4]])   shape (2, 5)

    # ── 查 cos / sin 表 ──
    # cos_cached[token_positions]：按每个格子的 pos 取对应行
    # 例子 cos.shape = (2, 5, 2)
    #   cos[0, 4, :] = cos_cached[4] ≈ tensor([-0.6536,  0.9992])  ← "cat" 两个 k 的 cos
    #   cos[0, 1, :] = cos_cached[1] ≈ tensor([ 0.5403,  0.9999])  ← "love"
    cos = self.cos_cached[token_positions]
    sin = self.sin_cached[token_positions]  # shape 同 cos，例子 (2, 5, 2)

    # ── 拆成 a_k, b_k ──
    # x[..., 0::2] 取偶数位 → 所有 a_k
    # x[..., 1::2] 取奇数位 → 所有 b_k
    # 例子 x[0,4,:]=[3,4,5,6] → a=tensor([3.,5.]), b=tensor([4.,6.])
    # shape: (..., seq, d_k/2) → 例子 (2, 5, 2)
    a = x[..., 0::2]
    b = x[..., 1::2]

    # ── 2D 旋转（减号来自旋转矩阵，不是 pos 为负）──
    # 对 "cat" (pos=4, k=0): a₀'=cos(4)*3 - sin(4)*4
    # 对 "cat" (pos=4, k=1): a₁'=cos(0.04)*5 - sin(0.04)*6
    # shape 不变 → 例子 a_rot.shape = (2, 5, 2)
    a_rot = a * cos - b * sin
    b_rot = a * sin + b * cos

    # ── 拼回 d_k 维 ──
    # out[0, 4, :] = [a₀', b₀', a₁', b₁']，还是 4 个数
    # shape 与 x 完全相同 → 例子 (2, 5, 4)
    out = torch.empty_like(x)
    out[..., 0::2] = a_rot
    out[..., 1::2] = b_rot
    return out
