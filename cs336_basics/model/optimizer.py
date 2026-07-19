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
    """Clip combined parameter gradients to have ℓ₂-norm at most max_l2_norm.

    If ‖g‖₂ ≤ M, leave grads unchanged; otherwise scale every grad in-place by
    M / (‖g‖₂ + ε). Matches the assignment (ε = 10⁻⁶, PyTorch default).
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    # ‖g‖₂ = sqrt(sum over all grad tensors of ‖grad‖₂²)
    # detach()：返回与 g 共享数据、但不挂在 autograd 图上的张量。
    # 量范数只是读数做判断/算 scale，不该也不需要再对「范数本身」反传；
    # 不 detach 的话，后面的 sqrt/sum 可能把计算图接上去，既浪费又可能干扰梯度。
    total_norm = torch.sqrt(
        sum(g.detach().float().norm(2).square() for g in grads)
    )
    if total_norm <= max_l2_norm:
        return

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

                # PyTorch 约定：名字以「_」结尾的方法 = 原地（in-place）修改该张量，不新建一份。
                # 例如 p.mul_(c) 等价于 p ← p * c；m.add_(g, alpha=a) 等价于 m ← m + a*g。
                # 官方优化器实现里大量用这种写法（省分配、直接改参数/状态），属于常见实践。
                # 写成 p.data = p.data * c、m = beta1*m + ... 再赋回 state 也对，只是更啰嗦、多临时张量。

                # 解耦 weight decay：θ ← θ - α λ θ  ≡  θ ← θ * (1 - α λ)
                p.mul_(1 - lr * weight_decay)

                # m ← β1 m + (1-β1) g   （原地写回 state["m"]）
                # v ← β2 v + (1-β2) g²  （addcmul_: v ← v + value * grad * grad）
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # θ ← θ - α_t * m / (√v + ε)
                # addcdiv_: p ← p + value * m / denom
                p.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)

        return loss
