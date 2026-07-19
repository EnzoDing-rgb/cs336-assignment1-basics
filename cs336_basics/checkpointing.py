"""Save / load training checkpoints (model + optimizer + iteration)."""

from __future__ import annotations

import os
from typing import BinaryIO, IO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """把 model / optimizer / iteration 打成一个 dict，torch.save 出去。

    例（训到第 10 步后）：
        model:      torch.nn.Module（如 Linear / TransformerLM）
        optimizer:  torch.optim.Optimizer（如 AdamW）
        iteration:  int = 10
        out:        str | Path | 二进制文件对象，如 "checkpoint.pt"

        model.state_dict()
            → dict[str, torch.Tensor]
              例：{"fc1.weight": Tensor 形状 (200, 100), "fc1.bias": Tensor 形状 (200,), ...}

        optimizer.state_dict()
            → dict[str, Any]
              例：{"state": {0: {"m": Tensor(...), "v": Tensor(...), "t": 10}, ...},
                   "param_groups": [{"lr": 0.001, "betas": (0.9, 0.999), ...}]}

        写入的 obj: dict[str, Any] =
            {
              "model":      <上面那份 model state_dict>,      # dict[str, Tensor]
              "optimizer":  <上面那份 optimizer state_dict>,  # dict
              "iteration":  10,                               # int
            }
    """
    obj: dict[str, object] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """从 src 读回 checkpoint，原地写进 model / optimizer，返回 iteration。

    例：
        src:        str | Path | 文件对象，如 Path("checkpoint.pt")
        model:      新的、同结构的 torch.nn.Module（参数还是随机的）
        optimizer:  绑在这个 model.parameters() 上的新 Optimizer

        torch.load(src)
            → obj: dict[str, object]
              obj["model"]:      dict[str, torch.Tensor]
              obj["optimizer"]:  dict（含 state / param_groups）
              obj["iteration"]:  int，例 10

        model.load_state_dict(...)      # 原地改 model 里各 Parameter，返回值不用
        optimizer.load_state_dict(...)  # 原地改 optimizer.state，返回值不用
        return 10                       # 类型 int
    """
    obj: dict = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    iteration: int = obj["iteration"]
    return iteration
