"""Autoregressive decoding with temperature scaling and top-p (nucleus) sampling.

本模块分三层（每层只干一件事，正交可测）：

  logits  ──► temperature_softmax ──► top_p_filter ──► sample_from_probs ──► next_id
                （温度）                  （砍长尾）         （加权抽签）

  decode() 负责主循环：encode prompt → 反复 decode_one_step → decode 成字符串。

贯穿例子（玩具词表 V=5，真实 GPT-2 只是 V 变成 50257，逻辑一样）：

  token id:   0      1      2      3      4
  含义:      the     a     cat    dog    xyz
  probs:     0.40  0.25  0.15  0.12  0.08
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor

from cs336_basics.model.normalization import softmax
from cs336_basics.model.transformer import TransformerLM
from cs336_basics.tokenization.tokenizer import Tokenizer

# 作业规定的序列结束符；decode 循环里遇到它的 id 就停。
EOS_TOKEN = "<|endoftext|>"


# ---------------------------------------------------------------------------
# Layer 1：纯张量运算（不碰 model / tokenizer）
# ---------------------------------------------------------------------------


def temperature_softmax(
    logits: Float[Tensor, " vocab"],
    temperature: float,
) -> Float[Tensor, " vocab"]:
    """把最后位置的 logits 变成概率分布 q。

    输入 logits 例（V=3）: tensor([3.0, 1.0, 0.0])  —— 还没归一化，只是分数。
    输出 probs 例:         tensor([0.84, 0.11, 0.04]) —— 非负，沿最后一维求和 ≈ 1。

    temperature 直觉（同一组 logits）：
      τ=1.0  普通 softmax
      τ=0.5  更尖，最大那个 id 概率接近 1（更像总选 argmax）
      τ=2.0  更平，小概率 id 也更容易被抽到
      τ=0    greedy：100% 压在 argmax 那个 id 上（one-hot）
    """
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")

    if temperature == 0:
        # greedy：例 logits=[3,1,0] → probs=[1,0,0]
        best_id = torch.argmax(logits)
        probs = torch.zeros_like(logits)
        probs[best_id] = 1.0
        return probs

    # τ>0：先除以温度再 softmax。例 τ=2 → [3,1,0]/2 = [1.5,0.5,0] 再 softmax
    return softmax(logits / temperature, dim=-1)


def top_p_filter(
    probs: Float[Tensor, " vocab"],
    top_p: float,
) -> Float[Tensor, " vocab"]:
    """Nucleus / top-p：砍掉长尾，只留「累计概率刚满 p」的最高频词，再归一化。

    讲义玩具分布（已按概率从大到小）：
      the=0.40, a=0.25, cat=0.15, dog=0.12, xyz=0.08

    top_p=0.5：
      累加 0.40 < 0.5，继续；0.40+0.25=0.65 ≥ 0.5 → 只留 {the, a}
      新分布: the≈0.615, a≈0.385，其余 id 概率为 0

    top_p=1.0：
      第一个词就使累加 ≥ 1？不，要保留全部词直到累加满 1 → 等价于不截断。
    """
    if not (0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    # sort 返回 (values, indices)。例 probs=[0.15,0.40,0.25,0.12,0.08]
    #   → sorted_probs=[0.40,0.25,0.15,0.12,0.08], sorted_idx=[1,2,0,3,4]
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # 找「第一个让前缀和 ≥ top_p」的位置 r，保留排序后的 0..r。
    # cumulative > top_p 为 True 的位置及之后都要丢掉；
    # 右移一位：刚好把「第一个 ≥ top_p」的那个词留在核里。
    #
    # 例 top_p=0.5, cumulative=[0.4,0.65,0.8,...]
    #   cumulative > 0.5 → [F,T,T,...] → 右移+强制第0个保留 → [F,F,T,...]
    #   → 保留 sorted 的前 2 个（the, a）
    remove_mask = cumulative > top_p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)

    # 把截断后的概率 scatter 回原 token id 顺序
    filtered = torch.zeros_like(probs)
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)

    total = filtered.sum(dim=-1, keepdim=True)
    # 理论上 total>0（至少保留了概率最大的那个词）
    return filtered / total


def sample_from_probs(
    probs: Float[Tensor, " vocab"],
    generator: torch.Generator | None = None,
) -> int:
    """按 probs 加权随机抽 1 个 token id（轮盘赌 / multinomial）。

    例 probs=[0.6, 0.3, 0.1, 0, 0]（V=5，后两个被 top-p 砍掉）：
      在 [0,1) 上掷骰子 u；u<0.6 选 0，u<0.9 选 1，否则选 2。
      torch.multinomial 做的就是这件事，只是用 C++ 实现，大词表更快。

    返回 Python int，例如 318，表示「下一个 token 的 id」。
    """
    # multinomial 要 2D 输入：(batch=1, vocab)；num_samples=1 表示只抽 1 次。
    # 例返回 tensor([2]) → .item() → 2
    drawn = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(drawn.item())


# ---------------------------------------------------------------------------
# Layer 2：单步解码（logits → 一个 next_id）
# ---------------------------------------------------------------------------


def decode_one_step(
    logits: Float[Tensor, " vocab"],
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
) -> int:
    """模型吐出的最后位置 logits → 采样得到下一个 token id。

    管线（每步固定走这三步，没有「可选开关」）：
      logits  --temperature_softmax-->  q  --top_p_filter-->  q'  --sample-->  id
    """
    probs = temperature_softmax(logits, temperature)
    probs = top_p_filter(probs, top_p)
    return sample_from_probs(probs, generator=generator)


# ---------------------------------------------------------------------------
# Layer 3：自回归主循环
# ---------------------------------------------------------------------------


@torch.inference_mode()
def decode(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_tokens: int,
    context_length: int,
    temperature: float,
    top_p: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> str:
    """用已训练好的 LM 续写 prompt，直到 EOS 或生成满 max_tokens 个新 token。

    参数（作业要求的四项都在，外加跑模型必需的 context_length / device）：
      prompt:        用户输入字符串，例 "Once upon a time"
      max_tokens:    最多**新生成**几个 token（不含 prompt 里的 token）
      temperature:   温度 τ（见 temperature_softmax）
      top_p:         nucleus 概率质量 p ∈ (0, 1]
      context_length: 模型能看的最大序列长度；超长时从左侧截断旧 token
      device:        例 "cpu" 或 "cuda:0"

    返回完整字符串 tokenizer.decode(prompt_ids + generated_ids)。

    循环里只维护 list[int]，不在中间拼字符串；BPE 的空格在 bytes 里，
    最后 decode 时 tokenizer 会拼 bytes 再 utf-8 解码。
    """
    if max_tokens < 0:
        raise ValueError(f"max_tokens must be >= 0, got {max_tokens}")
    if context_length <= 0:
        raise ValueError(f"context_length must be > 0, got {context_length}")

    model.eval()
    torch_device = torch.device(device)

    # EOS id：例 encode("<|endoftext|>") → [50256]（GPT-2 词表）
    eos_id = tokenizer.encode(EOS_TOKEN)[0]
    prompt_ids: list[int] = tokenizer.encode(prompt)
    generated_ids: list[int] = []

    while len(generated_ids) < max_tokens:
        # 当前喂给模型的上下文 = prompt + 已生成部分
        # 例 prompt_ids=[1,2], generated=[3] → context=[1,2,3]
        context_ids = prompt_ids + generated_ids
        if len(context_ids) > context_length:
            # 只保留最后 context_length 个（丢掉最左边的旧 token）
            # 例 context_length=4, context=[10,11,12,13,14] → [11,12,13,14]
            context_ids = context_ids[-context_length:]

        # list[int] → tensor，加 batch 维：shape (1, seq)
        # 例 context=[1,2,3] → input_ids=tensor([[1,2,3]], device=cpu, dtype=long)
        input_ids = torch.tensor(
            [context_ids], device=torch_device, dtype=torch.long
        )

        # 前向：input (1, seq) → logits (1, seq, V)；我们只要最后一个位置
        # 例 logits[0,-1,:] shape (V,) = 词表上每个 token 的下一词分数
        logits = model(input_ids)[0, -1, :]

        next_id = decode_one_step(
            logits,
            temperature,
            top_p,
            generator=generator,
        )
        generated_ids.append(next_id)

        if next_id == eos_id:
            break

    return tokenizer.decode(prompt_ids + generated_ids)


__all__ = [
    "EOS_TOKEN",
    "decode",
    "decode_one_step",
    "sample_from_probs",
    "temperature_softmax",
    "top_p_filter",
]
