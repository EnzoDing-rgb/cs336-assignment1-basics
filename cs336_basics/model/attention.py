"""Scaled dot-product attention, multi-head attention, RoPE."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.model.linear import Linear
from cs336_basics.model.language_model import RoPE
from cs336_basics.model.normalization import softmax


def scaled_dot_product_attention(
  Q: Float[Tensor, "..."],
  K: Float[Tensor, "..."],
  V: Float[Tensor, "..."],
  mask: Bool[Tensor, "..."] | None = None,
) -> Float[Tensor, "..."]:
  """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V，可选 mask。

  术语（全文统一）：
    大格子 = 一个 token / 一个序列座位。例："I love cats" → 3 个大格子。
    小格子 = 某个大格子里那条向量的一格数字。
      Q/K 每个大格子有 d_k 个小格子；V 每个大格子有 d_v 个小格子。

  形状（前缀维 batch / heads 原样保留）：
    Q:    (batch, n, d_k)   # n 个 query 大格子，每个 d_k 个小格子
    K:    (batch, m, d_k)   # m 个 key 大格子，每个也必须是 d_k 个小格子
    V:    (batch, m, d_v)   # 与 K 同为 m 个大格子；每个 d_v 个小格子（可 ≠ d_k）
    mask: (batch, n, m)     # 大格子×大格子的开关表；True=可看，False=不可看
    out:  (batch, n, d_v)

  哪些必须相等 / 可以不等：
    必须相等:
      - Q 与 K 的小格子个数都是 d_k（要对齐逐格相乘再相加，见下）
      - K 与 V 的大格子个数都是 m（同一批座位：权重按 key 大格子分，V 也是这批）
    可以不等:
      - n 与 m（query 大格子数 vs key/value 大格子数）
      - d_k 与 d_v（「像不像」用几格 vs 「货物本身」用几格；常设相等只是工程方便）

  三个容易混的词（不是一回事）：
    逐格相乘 (element-wise)：小格子配对相乘，结果仍是一串数字。
    点积 (dot product)：先逐格相乘，再把所有乘积加总 → 只剩 1 个数（相似度分数）。
    转置 (transpose)：把 K 从「大格子横着放」翻成「大格子竖着放」，
      好让矩阵乘法对每一对 (query大格子, key大格子) 自动做一次点积。
      转置本身不乘法；它只是改摆放，好让「横乘纵」= 点积。

  数字例子（可不等的维用不同数）：
    batch=1, n=2, m=3, d_k=4, d_v=5
    Q=(1,2,4), K=(1,3,4), V=(1,3,5), mask=(1,2,3), out=(1,2,5)
  """
  # ── d_k = 每个 Q/K 大格子里有多少个小格子 ──
  # 例子 Q=(1,2,4) → d_k=4
  d_k = Q.shape[-1]

  # ── scores = Q @ K^T ：对每一对大格子算一次点积 ──
  #
  # 点积像什么（十岁版）：
  #   两个大格子各有 4 个小格子，像两排抽屉。
  #   把对应抽屉里的数相乘，再把 4 个乘积加起来 → 得到 1 个分数。
  #   分数大 = 这两排抽屉「合拍」。
  #   例：q=[1,2,3,4], k=[1,0,1,0] → 1*1+2*0+3*1+4*0 = 4
  #
  # 为什么要 transpose K？
  #   Q 里每个大格子是横着的一行：(n, d_k)
  #   K 原本也是横着的一行：(m, d_k)
  #   矩阵乘法规定：左边一行 × 右边一列 = 点积。
  #   所以要把 K 翻成列摆放：K^T 形状 (d_k, m)
  #   例子: (1,2,4) @ (1,4,3) → (1,2,3)
  #   scores[0,i,j] = 第 i 个 query 大格子 与 第 j 个 key 大格子 的点积
  scores = Q @ K.transpose(-2, -1)

  # ── 分数除以 sqrt(d_k)，小格子多时点积容易偏大，压一压 ──
  scores = scores / math.sqrt(d_k)

  # ── mask：大格子×大格子的「能不能看」开关表 ──
  #
  # mask 与 scores 同形 (1,2,3)。下标：
  #   mask[b, i, j] ↔ scores[b, i, j]
  #   i = query 大格子；j = key 大格子
  #   True=能看，False=不能看
  #
  # 通用例子 (n=2, m=3)：
  #              key大格子0  key大格子1  key大格子2
  #   query大格子0   True        True         False
  #   query大格子1   True        False        True
  #   mask[0,0,2]=False → 禁止 query0 看 key2 → scores[0,0,2]=-inf
  #
  # 因果自注意力特例（常有 n=m）：大格子=句子里的座位。
  # "I love cats" → 大格子0="I", 1="love", 2="cats"
  # 站在大格子 i 只能看 j<=i（过去+自己），看未来=作弊。
  # 下三角（含对角线），不是只留对角线：
  #              key0("I")  key1("love")  key2("cats")
  #   query0("I")    True       False         False
  #   query1("love") True       True          False
  #   query2("cats") True       True          True
  #
  # False → -inf → softmax 后权重 0；同行 True 位置重新归一化，和为 1。
  if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))

  # ── 沿 key 大格子维做 softmax：每个 query 大格子得到对 m 个 key 的概率 ──
  # 例：[1.0, 2.0, -inf] → 约 [0.27, 0.73, 0.0]
  attn = softmax(scores, dim=-1)

  # ── 用概率去混合 V 的大格子（货物）──
  # (1,2,3) @ (1,3,5) → (1,2,5)
  # 每个 query 大格子输出一条有 d_v=5 个小格子的向量；与 d_k 无关。
  return attn @ V


class MultiheadSelfAttention(nn.Module):
  """因果多头自注意力。

  全文统一数字例子：
    d_model = 768, num_heads = 8 → 每头 d_k = d_v = 768/8 = 64
    句子大格子数 seq 先当 3："I love cats"（自注意力里 n = m = 3）
    batch 先当 2，好看形状（两句并行，互不相干）

  和「8×3 个 768×64 小矩阵」的关系：
    实现上用 3 个大矩阵 W_Q/W_K/W_V，每个 768×768，一次乘完再切成 8 段×64。
    这和「每个头各有一套 768×64 的 Q/K/V，共 8×3 个」数学上等价；只是算得更省。
  """

  def __init__(
    self,
    d_model: int,
    num_heads: int,
    *,
    max_seq_len: int | None = None,
    theta: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
  ) -> None:
    super().__init__()
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    self.d_model = d_model
    self.num_heads = num_heads
    # 例子：768 / 8 = 64 —— 每个头里，一个大格子有 64 个小格子
    self.d_k = d_model // num_heads

    factory = {"device": device, "dtype": dtype}
    # 你心里的「8×3 个 768×64」就落在这三个 Linear 上（再加后面的切头）：
    #
    #   看法 A（直观）：每个头各有 W_Qᵢ、W_Kᵢ、W_Vᵢ，每个是 768→64，共 8×3 = 24 个小矩阵。
    #   看法 B（本代码）：三个大矩阵，每个 768→768（Linear 的 weight 形状是 (768, 768)）。
    #                     一次乘完后，768 = [头0的64 | 头1的64 | … | 头7的64]，
    #                     再在 _split_heads 里切开 —— 和看法 A 数学等价，少做几次 kernel。
    #
    # 例：x 某个大格子是 768 维 → q_proj → 仍是 768 维 Q（里面已经并排装着 8 份 64）。
    self.q_proj = Linear(d_model, d_model, bias=False, **factory)
    self.k_proj = Linear(d_model, d_model, bias=False, **factory)
    self.v_proj = Linear(d_model, d_model, bias=False, **factory)
    # 输出投影 W_O：不算进上面的 8×3。那是「8 头算完、拼回 768」之后的第四个 768×768，
    # 用来混合各头结果；不是再生成一套 Q/K/V。
    self.o_proj = Linear(d_model, d_model, bias=False, **factory)

    # RoPE 可选。挂上时作用在「每头的 64 个小格子」上，不是整条 768。
    if theta is not None:
      assert max_seq_len is not None, "RoPE needs max_seq_len"
      self.rope: RoPE | None = RoPE(theta, self.d_k, max_seq_len, device=device)
    else:
      self.rope = None

  def _split_heads(self, x: Tensor) -> Tensor:
    # 把「8 段 64 并排放着的 768」切开，变成 8 个独立的头。
    # 例: (2, 3, 768) → (2, 8, 3, 64)
    #   ... seq (heads d_k)  = batch 维保留，最后一维 768 拆成 heads×d_k
    #   ... heads seq d_k    = 把头维挪到 seq 前面，方便每头自己算 attention
    return rearrange(
      x,
      "... seq (heads d_k) -> ... heads seq d_k",
      heads=self.num_heads,
      d_k=self.d_k,
    )

  def _merge_heads(self, x: Tensor) -> Tensor:
    # 把 8 个头的输出再拼回一条 768（concat，还没做输出投影）。
    # 例: (2, 8, 3, 64) → (2, 3, 768)
    #   ... heads seq d_k → ... seq (heads d_k)
    return rearrange(x, "... heads seq d_k -> ... seq (heads d_k)")

  def forward(
    self,
    x: Float[Tensor, "..."],
    token_positions: Int[Tensor, "..."] | None = None,
  ) -> Float[Tensor, "..."]:
    # x 例: (2, 3, 768) —— 2 个 batch，每个 3 个大格子，每个大格子 768 维
    seq_len = x.shape[-2]

    # ① 投影 + 切头
    # q_proj(x): (2, 3, 768)  —— 每个大格子一条 Q（8×64 并排）
    # split 后:  (2, 8, 3, 64) —— 8 个头，每个头里 3 个大格子×64 小格子
    # K、V 同理。这就是「8 份互相独立的 Q/K/V」，每份 64 维。
    Q = self._split_heads(self.q_proj(x))
    K = self._split_heads(self.k_proj(x))
    V = self._split_heads(self.v_proj(x))

    # ② RoPE（可选）：只转每个头内部的 Q、K，不转 V
    # 在 64 个小格子上按对旋转（32 对）；角度看大格子座位 pos + 小格子对编号 k
    # Q/K 形状 (2, 8, 3, 64)；token_positions 例 (2, 3) → unsqueeze 成 (2, 1, 3)
    # 好让 RoPE 的 cos/sin 广播过「8 个头」这一维
    if self.rope is not None:
      assert token_positions is not None, "RoPE requires token_positions"
      pos = token_positions.unsqueeze(-2)  # (batch, 1, seq) 例 (2, 1, 3)
      Q = self.rope(Q, pos)
      K = self.rope(K, pos)

    # ③ 因果 mask：大格子×大格子的下三角
    # seq=3 时：
    #              key0  key1  key2
    #   query0     True  False False
    #   query1     True  True  False
    #   query2     True  True  True
    # shape (3, 3)，广播到 (2, 8, 3, 3) —— 每个 batch、每个头共用同一张因果表
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

    # ④ 8 个头各自做：softmax(Q K^T / √64) V（带上面的因果 mask）
    # 输入/输出都是 (2, 8, 3, 64)；batch 与 heads 维都当 batch-like，互不往来
    # 每个头留下的是「按注意力权重混合后的 V」，不是裸 V，也不是只打分
    attn_out = scaled_dot_product_attention(Q, K, V, mask=causal)

    # ⑤ 拼头 → 输出投影
    # merge: (2, 8, 3, 64) → (2, 3, 768)  只是并排放着，头与头还没线性混合
    # o_proj: (2, 3, 768) → (2, 3, 768)  W_O 让最终每个小格子可以是 8 头结果的加权和
    return self.o_proj(self._merge_heads(attn_out))
