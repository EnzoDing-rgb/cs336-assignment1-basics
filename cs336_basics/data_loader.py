"""Sample language-modeling batches from a 1D token array."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从一长串 token ID 里随机切 (inputs, targets)。

    贯穿本函数的例子（跟下面注释同一套数）：
        dataset: np.ndarray[int]，形状 (10,) = [10,11,12,13,14,15,16,17,18,19]
        batch_size B=2: int
        context_length m=4: int
        碰巧抽到起点 starts: np.ndarray[int]，形状 (2,) = [2, 5]
    """
    # dataset 形状 (n,)；例 n=10, m=4 → 合法起点 0..5，共 6 个
    # np.random.randint 的 high 开区间：randint(0, 6) → 可能抽到 0..5
    # 例：starts: np.ndarray，dtype=int64，形状 (2,) = [2, 5]
    starts = np.random.randint(
        0, len(dataset) - context_length, size=batch_size
    )

    # 循环前：两个空 list[np.ndarray]
    # 循环后（本例）：
    #   inputs_list:  list[np.ndarray]，长度 2
    #                 [ array([12,13,14,15], shape=(4,)), array([15,16,17,18], shape=(4,)) ]
    #   targets_list: list[np.ndarray]，长度 2
    #                 [ array([13,14,15,16], shape=(4,)), array([16,17,18,19], shape=(4,)) ]
    inputs_list: list[npt.NDArray] = []
    targets_list: list[npt.NDArray] = []
    for i in starts:
        # 例 i=2 (Python int / numpy 标量), m=4：
        #   window: np.ndarray，形状 (5,) = dataset[2:7] = [12,13,14,15,16]
        #   window[:-1] → np.ndarray 形状 (4,) = [12,13,14,15]  进 inputs_list
        #   window[1:]  → np.ndarray 形状 (4,) = [13,14,15,16]  进 targets_list
        window = dataset[i : i + context_length + 1]
        inputs_list.append(window[:-1])
        targets_list.append(window[1:])

    # np.stack：list 里 B 个形状 (m,) 的数组 → 一个形状 (B, m) 的 np.ndarray
    #
    # 例 inputs_list：
    #   行0: np.ndarray (4,) = [12, 13, 14, 15]
    #   行1: np.ndarray (4,) = [15, 16, 17, 18]
    # stack 后：
    #   np.ndarray，dtype 随原数组，形状 (2, 4)：
    #   [[12, 13, 14, 15],
    #    [15, 16, 17, 18]]
    #
    # torch.tensor(..., dtype=torch.long, device=device)：
    #   → torch.Tensor，dtype=torch.int64，形状 (2, 4)，在 device 上
    # targets 同理：torch.Tensor (2, 4) int64
    inputs: torch.Tensor = torch.tensor(
        np.stack(inputs_list), dtype=torch.long, device=device
    )
    targets: torch.Tensor = torch.tensor(
        np.stack(targets_list), dtype=torch.long, device=device
    )
    return inputs, targets
