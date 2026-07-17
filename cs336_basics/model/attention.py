"""Scaled dot-product attention, multi-head attention, RoPE."""

from __future__ import annotations

import math

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from cs336_basics.model.normalization import softmax


def scaled_dot_product_attention(
  Q: Float[Tensor, "..."],
  K: Float[Tensor, "..."],
  V: Float[Tensor, "..."],
  mask: Bool[Tensor, "..."] | None = None,
) -> Float[Tensor, "..."]:
  """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V，可选 mask。

  形状（前缀维是 batch，或 batch×num_heads，原样保留）：
    Q:    (batch, n, d_k) 或 (batch, heads, n, d_k)
    K:    (batch, m, d_k) 或 (batch, heads, m, d_k)
    V:    (batch, m, d_v) 或 (batch, heads, m, d_v)
    mask: (batch, n, m)   或 (batch, heads, n, m)；True=可看，False=不可看
    out:  (batch, n, d_v) 或 (batch, heads, n, d_v)

  全文例子（三维，数字小一点好盯）：
    batch=1, n_queries=2, n_keys=3, d_k=4, d_v=4
    Q.shape = (1, 2, 4)   # 2 个大格子，每个 4 维
    K.shape = (1, 3, 4)   # 3 个大格子，每个 4 维
    V.shape = (1, 3, 4)
    mask 与 scores 同形 (1, 2, 3)，每个格子 mask[b,i,j] 回答：
      「batch b 的 query i 能不能看 key j？」True=能，False=不能。
    例如 mask[0, 0, 2]=False → 禁止 query0 看 key2 → 会把 scores[0, 0, 2] 改成 -inf
  """
  # ── d_k：小格子个数，用来做缩放 ──
  # 例子 Q.shape=(1,2,4) → d_k=4，scale=2.0
  d_k = Q.shape[-1]

  # ── scores = Q @ K^T ──
  # K 最后两维从 (..., m, d_k) 转成 (..., d_k, m)，才能「横乘纵」做点积
  # 例子: (1,2,4) @ (1,4,3) → (1,2,3)
  # scores[0, i, j] = 大格子 i（query）与大格子 j（key）的点积
  scores = Q @ K.transpose(-2, -1)

  # ── 除以 sqrt(d_k)，避免点积过大把 softmax 挤成 one-hot ──
  scores = scores / math.sqrt(d_k)

  # ── mask：决定「哪个 query 大格子可以看哪个 key 大格子」──
  #
  # mask 和 scores 形状相同，都是 (batch, n_queries, n_keys)，例子里是 (1, 2, 3)。
  # 三个下标分别是：
  #   mask[b, i, j]  ↔  scores[b, i, j]
  #   b = 第几个 batch
  #   i = 第几个 query 大格子（行）
  #   j = 第几个 key 大格子（列）
  #
  # 约定：True = 允许 attend；False = 禁止 attend。
  #
  # 例子（先看二维表，再想外面还有 batch 维）：
  #              key0   key1   key2
  #   query0    True   True   False     ← query0 只能看 key0、key1
  #   query1    True   False  True      ← query1 只能看 key0、key2
  #
  # 写成张量（再包一层 batch）:
  #   mask.shape = (1, 2, 3)
  #   mask[0, 0, :] = [True,  True,  False]   # 第 0 个 batch，第 0 行
  #   mask[0, 1, :] = [True,  False, True ]   # 第 0 个 batch，第 1 行
  #
  # 和 scores 一一对齐，例如：
  #   mask[0, 0, 2] == False
  #   表示：batch0 里，query0 不能看 key2
  #   对应分数格子就是 scores[0, 0, 2]（同一个 b,i,j）
  #
  # 做法：把 False 的位置改成 -inf。softmax 时 e^{-inf}=0，
  # 所以这些位置权重变成 0；同一行里剩下的 True 位置会重新归一化，和仍为 1。
  #
  # ~mask：把 True/False 取反，得到「哪些位置要填 -inf」。
  # masked_fill(~mask, -inf)：只在 ~mask 为 True（即原 mask 为 False）的格子写入 -inf。
  # 例子：mask[0,0,2]=False → ~mask[0,0,2]=True → scores[0,0,2] = -inf
  if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))

  # ── 沿 key 维（最后一维 j）做 softmax → 每行变成对所有 key 的概率 ──
  # 例子只看 query0 那一行 scores[0, 0, :]，假设缩放后是 [1.0, 2.0, -inf]：
  #   softmax → 大约 [0.27, 0.73, 0.0]，第三个是 0（被 mask 掉），前两个和为 1
  attn = softmax(scores, dim=-1)

  # ── 用权重混合 V ──
  # (1,2,3) @ (1,3,4) → (1,2,4)：每个 query 大格子得到一个 d_v 维输出
  return attn @ V
