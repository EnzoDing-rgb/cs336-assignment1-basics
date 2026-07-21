# Learning rate experiment（TinyStories）

**问题：** 在固定数据、模型与训练步数时，把 `lr_max` 从偏小扫到偏大，最终 valid loss 怎么变？有没有「过大就崩」的边？

**交付图：**  
[`artifacts/plots/lr_sweep/final_valid_vs_lr.png`](../artifacts/plots/lr_sweep/final_valid_vs_lr.png)  
（曲线叠加见同目录 `valid_curves_overlay.png`）

---

## 1. 设置

| 固定 | 取值 |
|------|------|
| 底座配置 | `configs/tinystories_small.yaml`（数据、模型、`B=32`、`max_iters=20000`） |
| 日程形状 | warmup → cosine；`lr_min = lr_max / 10` |
| 自变量 | 只改 `lr_max`（以及成比例的 `lr_min`） |

九档网格：`1e-4`、`1.8e-4`、`3e-4`、`5.6e-4`、`1e-3`、`1.8e-3`、`3.2e-3`、`5.6e-3`、`1e-2`。  
其中 `3e-4` 复用早期长跑 `tinystories_small/20260720_1436`（该次训练的 `lr_max` 就是 `3e-4`）。

脚本：`scripts/sweep_lr.py`、`scripts/plot_lr_sweep.py`。

---

## 2. 结果总表

按 `lr_max` 从小到大排列；`final valid` 取各 run 接近收尾的验证损失（step ≈ 19800）。

| `lr_max` | final valid | 备注 |
|---------:|------------:|------|
| 1e-4 | 1.752 | 明显偏小，学得慢 |
| 1.8e-4 | 1.609 | 仍偏小 |
| 3e-4 | 1.527 | 来自 `tinystories_small` 长跑 |
| 5.6e-4 | 1.465 | 继续下降 |
| 1e-3 | 1.436 | 已进入较好区间 |
| **1.8e-3** | **1.425** | **本网格最低** |
| 3.2e-3 | 1.436 | 开始回升 |
| 5.6e-3 | 1.459 | 继续变差 |
| 1e-2 | 1.516 | 右侧最差一档；训练仍能跑完，没有 NaN 炸掉 |

---

## 3. 发现

1. **曲线呈 U 形，而不是「越大越好」。**  
   从 `1e-4` 到 `1.8e-3`，final valid 一路下降；再往右到 `3.2e-3`、`5.6e-3`、`1e-2`，损失重新抬高。最佳点落在 **`lr_max=1.8e-3`（约 1.425）**，已经压进作业常见的 ≤1.45 目标附近。

2. **本网格右侧是「变差」，不是「瞬间崩盘」。**  
   即便扫到 `1e-2`，run 也能正常写完 metrics，没有出现 loss 变 NaN、训练中断那种灾难性不稳定。这里的「edge」更像是最优点右侧的性能悬崖：步子过大，最终收敛变差，但过程仍可观测。

3. **给后面 batch-size 实验定锚。**  
   后续同预算扫 `B` 时，以 `B=32`、`lr_max=1.8e-3` 为中心做线性缩放，再在每档做短跑三选一。batch-size 报告见 [`batch_size.md`](batch_size.md)。

---

## 4. 一句话

固定 TinyStories 小模型与 2 万步训练，只扫 `lr_max` 时，最优大约在 **`1.8e-3`**；再大 valid 会回升，但本网格里仍是「变差」而不是「炸掉」。
