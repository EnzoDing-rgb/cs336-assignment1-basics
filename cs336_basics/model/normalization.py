"""RMSNorm, softmax, and cross-entropy."""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
  """Numerically stable softmax along `dim`.

  Subtracts the max along `dim` before exp so large logits do not overflow.
  Softmax is invariant to adding a constant to all entries along that dim.
  """
  x_max = torch.amax(x, dim=dim, keepdim=True)
  exp_x = torch.exp(x - x_max)
  return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def cross_entropy(
  inputs: Float[Tensor, "batch_size vocab_size"],
  targets: Int[Tensor, "batch_size"],
) -> Float[Tensor, ""]:
  """对一批 logits 计算平均交叉熵损失（cross-entropy）。

  ---------------------------------------------------------------------------
  一、损失的定义（先记住这一条）

    每个样本有一份得分向量 o（长度等于词表大小 V），以及真实类别下标 y。
    用 softmax 把 o 变成概率分布 q；q(y) 是模型分给真实类别的概率。

    单条损失定义为：
        ℓ = -log q(y) = -log softmax(o)[y]

    下面代码里的「减最大值、拆成加减」，只是同一条公式的数值稳定写法，
    并不改变这个定义。

  ---------------------------------------------------------------------------
  二、为什么要把损失定义成 -log q(y)？

    我们希望真实类别的概率 q(y) 尽可能大。优化器做的是「最小化某个数」，
    所以把「让 q(y) 变大」改写成「让 -log q(y) 变小」：
      - q(y) = 1 时，ℓ = 0；
      - q(y) 接近 0 时，ℓ 变得非常大。

    这和下面两个经典说法是同一件事（知道名字和含义即可）：

    (1) 最大似然（maximum likelihood）
        似然（likelihood）指的是：在当前参数下，模型认为「数据里那个
        真实答案」出现的概率有多大。最大似然就是调整参数，让这个概率
        尽可能大。对概率取负对数再最小化，就是在做最大似然。

    (2) 独热目标下的交叉熵（cross-entropy with a one-hot target）
        独热（one-hot）指的是：只在真实类别位置为 1、其余位置全为 0
        的分布。例如三类且真实类是 B 时，真分布就是 [0, 1, 0]。
        交叉熵衡量「用模型分布 q 去描述这个真分布」有多费劲；
        当真分布是独热时，交叉熵正好化简成 -log q(y)。

  ---------------------------------------------------------------------------
  三、用数字把计算过程看一遍（真实类故意不是最高分）

    三类 A、B、C，得分 o = [100, 200, 300]。
    真实类是 B（下标 1），因此 o[y] = 200。
    三个数里的最大值 max = 300，来自 C，不是真实类。

    按定义：
      q(B) = exp(200) / (exp(100) + exp(200) + exp(300))
      ℓ    = -log q(B)
           = -200 + log(exp(100) + exp(200) + exp(300))

    为了避免 exp(300) 这类大数溢出，先对所有得分同减 max = 300
   （这样做不会改变概率），再写成：
      ℓ = -(o[y] - max) + log( Σ_a exp(o[a] - max) )
        = -(200 - 300) + log(exp(-200) + exp(-100) + exp(0))
        = 100 + log(很小的数 + 很小的数 + 1)
        ≈ 100

    要点：真实类不是最高分时，-(o[y] - max) 这一项必须保留。
    只有真实类碰巧就是最高分时，这一项才等于 0。

  ---------------------------------------------------------------------------
  四、张量形状，以及 adapter 里 batch_size 的含义

    inputs:  [N, V]  —— 每一行是一个样本的得分向量 o
    targets: [N]     —— 每一行是真实类别下标 y（词表编号，不是序列位置）
    返回值:  标量    —— 对 N 条样本的 ℓ 取平均

    语言模型里 TransformerLM 输出常是 [B, S, V]。测试里会先写成：
      inputs.view(-1, V)、targets.view(-1)
    再调用本函数（见 tests/test_nn_utils.py）。
    因此这里的 N 常常等于 B×S；adapter 文档把这个 N 叫做 batch_size，
    意思是「独立分类样本的条数」，不是「一次喂了几条完整序列」。

  ---------------------------------------------------------------------------
  五、为什么不用 einsum

    einsum 适合按维度做乘加合并。这里需要按 y 从每一行取出 o[y]，
    属于按索引取值，用 gather 更直接。
  """
  # inputs 形状 [N, V]，targets 形状 [N]。下面按「一行 = 一个样本的 o」理解。

  # 稳定实现对应的公式：
  #   ℓ = -(o[y] - max) + log Σ_a exp(o[a] - max)

  # ① 对每个样本，在词表维上取最大得分。
  #    dim=-1：沿着最后一维（长度 V 的词表维）计算。
  #    keepdim=True：结果保留成 [N, 1]，才能和 [N, V] 做逐行相减。
  #    例子：o=[100,200,300] → max=300 → 减去后得到 [-200,-100,0]
  max_logits = torch.amax(inputs, dim=-1, keepdim=True)  # [N, 1]
  shifted = inputs - max_logits  # [N, V]

  # ② 计算 log Σ_a exp(o[a] - max)。对最后一维求和后形状为 [N]。
  #    例子：log(exp(-200)+exp(-100)+exp(0)) ≈ log(1) = 0
  log_sum_exp_shifted = torch.log(torch.sum(torch.exp(shifted), dim=-1))  # [N]

  # ③ 取出真实类的得分 o[y]。
  #    例子：真实类是 B → o[y]=200（注意不是最大值 300）。
  #    unsqueeze(-1)：把 [N] 变成 [N, 1]，以符合 gather 对 index 形状的要求。
  #    gather(dim=-1, ...)：第 n 行取出 inputs[n, targets[n]]，得到 [N, 1]。
  #    squeeze(-1)：再把 [N, 1] 变回 [N]。
  correct_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [N]

  # ④ 拼成每条样本的损失。
  #    例子：-(200 - 300) + 0 ≈ 100
  #    这与 -log softmax(o)[y] 恒等，只是先减了 max 再算。
  max_per_example = max_logits.squeeze(-1)  # [N, 1] → [N]
  per_example_loss = -(correct_logits - max_per_example) + log_sum_exp_shifted  # [N]

  # ⑤ 对全部 N 条样本取平均，得到 0 维标量
  return torch.mean(per_example_loss)


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
