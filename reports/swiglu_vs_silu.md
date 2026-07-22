# SwiGLU vs SiLU（TinyStories）

**问题：** 去掉 FFN 门控（GLU），改用无门控 `FFN_SiLU(x)=W₂ SiLU(W₁x)`，在近似相同参数量下，valid loss 比默认 SwiGLU 差多少？

**交付图：**  
[`artifacts/plots/ffn/swiglu_vs_silu_valid.png`](../artifacts/plots/ffn/swiglu_vs_silu_valid.png)

---

## 1. 设置

| 固定 | 取值 |
|------|------|
| 底座 | `configs/tinystories_small.yaml`（`d_model=512`，`pre_norm`，`pos_encoding=rope`） |
| `batch_size` | **64** |
| `max_iters` | 20000 |
| `lr_max` / `lr_min` | **1.8e-3** / 1.8e-4 |
| 自变量 | 只改 `model.ffn_type` |
| 提前停 | 非有限 loss，或 `train_loss > 200` |

| `ffn_type` | 公式 | 自动 `d_ff` | 单层 FFN 参数量 |
|------------|------|-------------|----------------:|
| `swiglu` | \(W_2(\mathrm{SiLU}(W_1x)\odot W_3x)\) | 1408（`≈8/3·512`，对齐 64） | 2,162,688 |
| `silu` | \(W_2\,\mathrm{SiLU}(W_1x)\) | 2048（`4·512`） | 2,097,152 |

参数量差约 **3%**（作业要求 approximately matched）。

**复用：** SwiGLU 臂复用 RoPE 消融同超参长跑  
`tinystories_rope_b64_lr1.8e-3` → symlink `tinystories_swiglu_b64_lr1.8e-3`。  
SiLU 从零新跑 `tinystories_silu_b64_lr1.8e-3`。

脚本：`scripts/sweep_ffn.py`、`scripts/plot_ffn.py`。

---

## 2. 结果

`final valid` 取 step ≈ 19800。

| 变体 | final valid | 备注 |
|------|------------:|------|
| **SwiGLU** | **1.353** | 全程略低 |
| SiLU | 1.368 | 能训通；未 abort |

差值约 **+0.015** valid CE（SiLU 更差）。

---

## 3. 发现

1. **门控有帮助，但幅度不大。**  
   同 B、同 LR、同步数下，SwiGLU 最终 valid 1.353，无门控 SiLU 1.368；曲线形状接近，SwiGLU 全程略优。

2. **SiLU@`d_ff=4d` 训练稳定。**  
   开局 CE≈9.27，平滑降到 ~1.37，没有数值炸裂；参数量对齐策略可用。

3. **默认仍用 SwiGLU。**  
   在本 TinyStories 小模型设定下，门控带来可复现的小幅收益；没有理由换成无门控 SiLU。

---

## 4. 一句话

B=64、`lr_max=1.8e-3`、20k step：SwiGLU valid **1.353**，SiLU **1.368**；门控略好，基线保留 SwiGLU。
