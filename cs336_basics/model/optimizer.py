from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine LR schedule with linear warmup (LLaMA-style).

    - Warm-up (t < Tw): α_t = (t / Tw) * α_max
    - Cosine (Tw ≤ t ≤ Tc): α_t = α_min + ½(1 + cos(π u))(α_max − α_min),
      where u = (t − Tw) / (Tc − Tw)
    - Post-annealing (t > Tc): α_t = α_min
    """
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    if it <= cosine_cycle_iters:
        u = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1.0 + math.cos(math.pi * u)) * ( max_learning_rate - min_learning_rate)
    return min_learning_rate


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """把全体参数的梯度原地缩小（如果合在一起范数超过 M）。

    本质：不返回新梯度，也不新建一份 grad。
    直接改每个 parameter.grad 里的数（in-place），函数返回 None。
    优化器后面 step() 读到的就是裁过之后的梯度。
    """
    # grads 里每个 g 就是某个 p.grad 本身（同一个张量对象），不是拷贝。
    # 所以后面 g.mul_(...) 改的就是 parameter.grad。
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # 先量一下：所有 grad 摊平后的总长度 ‖g‖₂。
    # detach()：量尺寸时先「断开」自动求导。
    # 通俗讲：我们只是拿尺子量一下梯度有多大，用来决定要不要缩小；
    # 量尺子这个动作本身不需要再被反传。detach 之后，PyTorch 就不会
    # 把后面的 norm / sqrt 记进计算图里。
    # （真正要改的梯度还是下面的 g；detach 只影响这次量范数用的那份视图。）
    total_norm = torch.sqrt(
        sum(g.detach().float().norm(2).square() for g in grads)
    )
    if total_norm <= max_l2_norm:
        return

    # 超了上限：每个 grad 乘同一个 scale。mul_ 末尾的下划线 = 原地改。
    # g 就是 p.grad，所以这就是 in-place 更新梯度。
    scale = max_l2_norm / (total_norm + eps)
    for g in grads:
        g.mul_(scale)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")

        # m / v / t 不在这里初始化：每个 p 第一次进 step() 时再往 self.state[p] 里写。
        # 讲义伪代码 for t = 1..T：state 里先存 t=0，每次 step 开头 t ← t+1，第一步用的就是 t=1。
        # m、v 初值都是与 p 同形状的全 0。
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        # params + defaults 交给基类后，基类会填好 self.param_groups 和空的 self.state。
        #
        # 日常用法（零分组负担）：
        #   AdamW(model.parameters(), lr=1e-3, ...)
        # 基类自动做成「一个 group」，所有参数共用这份 defaults。你不用手写分组。
        #
        # 进阶用法（需要时才拆组）：
        #   AdamW([{"params": emb.parameters(), "lr": 1e-4},
        #          {"params": rest.parameters(), "lr": 1e-3}], ...)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        # self.param_groups：基类在 __init__ 里建好的列表，不是我们手写赋值的字段。
        # 每个 group 是 dict，至少有 "params"，以及从 defaults 合并来的 lr/betas/eps/weight_decay。
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 第一次见到这个 p：m、v 全 0，t = 0
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["t"] = 0

                # m、v 是对 state["m"] / state["v"] 的引用（同一个张量对象）。
                # 下面用 mul_ / add_ 原地改 m、v，就是在改 state 里存的那份，不必再写
                #   state["m"] = m
                # t 是 Python int（不可变），所以必须显式写回 state["t"] = t。
                m = state["m"]
                v = state["v"]
                t = state["t"] + 1
                state["t"] = t

                # α_t ← α * sqrt(1 - β2^t) / (1 - β1^t)
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # PyTorch：带「_」的方法原地改张量。p.mul_(c) ≡ p←p*c；m.add_(g, alpha=a) ≡ m←m+a*g。

                # --- Weight decay（本配置 α=lr=3e-4，λ=weight_decay=0.1）---
                # 公式：θ ← θ * (1 - α λ)。代入数：α λ = 3e-4 * 0.1 = 3e-5，故乘子 = 0.99997。
                # 例子：某权重 θ=2.0，仅做本行、不算后面 Adam 时：
                #   λ=0  →  2.0 * 1       = 2.0      （参数原样）
                #   λ=0.1 → 2.0 * 0.99997 = 1.99994  （每步往 0 收一丁点）
                # 一万步量级、若梯度长期推不动它：2.0 * (0.99997)**10000 ≈ 1.48，幅度被压下去。
                # 要解决的事：参数越长越大、模型死记训练集；每步固定收缩，减轻过拟合。
                # AdamW 解耦：收缩直接乘在 θ 上。若改成 L2（grad ← grad + λθ 再进 m/v），
                #   同一 θ=2、λ=0.1 会往 grad 里加 0.2；再被 √v 除掉，实际衰减强弱跟自适应步长缠在一起。
                # 容易忘：本行用的是 α=lr，下面 Adam 那步用的才是带 bias correction 的 lr_t。
                p.mul_(1 - lr * weight_decay)

                # m ← β1 m + (1-β1) g
                # v ← β2 v + (1-β2) g²   （addcmul_: v ← v + value * grad * grad）
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # θ ← θ - α_t * m / (√v + ε)    （本配置 ε=1e-8）
                # 例子：某维长期 grad=0，则 v 一直是 0，m 也是 0 → 商为 0，没事。
                # 更危险：m=1e-3（还有一点动量），v=0（二阶还没攒起来）
                #   ε=0    →  1e-3 / 0      → Inf/NaN，参数直接坏掉
                #   ε=1e-8 →  1e-3 / 1e-8   = 1e5，步子大但有限
                #   正常 √v=0.1 时：1e-3/(0.1+1e-8) ≈ 0.01，ε 可以当不存在
                p.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)

        return loss
