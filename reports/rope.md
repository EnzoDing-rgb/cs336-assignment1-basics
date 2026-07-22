# RoPE vs NoPE（TinyStories）

**问题：** 在因果 decoder-only Transformer 上，去掉显式位置编码（NoPE）相对基线 RoPE，valid loss 差多少？理论说 causal mask 或许已隐含位置信息；这里用同预算实证比一下。

**交付图：**  
[`artifacts/plots/pos_encoding/rope_vs_nope_valid.png`](../artifacts/plots/pos_encoding/rope_vs_nope_valid.png)

---

## 1. 设置

| 固定 | 取值 |
|------|------|
| 底座 | `configs/tinystories_small.yaml`（数据、模型宽深、`pre_norm`、warmup→cosine、`grad_clip=1.0`） |
| `batch_size` | **64** |
| `max_iters` | 20000（两边同 step、同 token） |
| `lr_max` / `lr_min` | **1.8e-3** / 1.8e-4（与本仓库 B=64 长跑赢家一致） |
| 自变量 | 只改 `model.pos_encoding` |
| 提前停 | 非有限 loss，或 `train_loss > 200` |

| `pos_encoding` | 含义 |
|----------------|------|
| `rope` | 各层 attention 对 Q/K 做 RoPE（`rope_theta=10000`） |
| `no_rope` | 不注入位置编码（`theta=None`）；**仍保留 causal mask** |

两边都从零训练（不复用 B=32 旧 run）。  
脚本：`scripts/sweep_pos_encoding.py`、`scripts/plot_pos_encoding.py`。

---

## 2. 结果

`final valid` 取 step ≈ 19800。

| 变体 | final valid | 备注 |
|------|------------:|------|
| **RoPE** | **1.353** | 全程更低 |
| NoPE | 1.414 | 能学，但全程高一截；未 abort |

差值约 **+0.06** valid CE（NoPE 更差）。两条曲线都平滑下降，没有炸。

---

## 3. 发现

1. **NoPE 在本设定下能训通。**  
   开局 CE≈9.25（与 RoPE 同量级），20k step 内收到 ~1.41，说明单靠 causal attention + 内容，模型仍能学到可用的 TinyStories 语言模型。

2. **RoPE 仍然明显更好。**  
   同 B、同 LR、同步数下，RoPE 最终 valid **1.353 vs 1.414**；曲线从早期评估点起就持续低于 NoPE。显式相对位置编码在这里不是可有可无的装饰。

3. **和「理论可行」不矛盾。**  
   文献指出 decoder-only 可以不显式喂位置；本实验支持的是「能跑」，不是「一样好」。默认训练应继续用 **RoPE**。

---

## 4. 一句话

B=64、`lr_max=1.8e-3`、20k step：RoPE valid **1.353**，NoPE **1.414**；NoPE 可训但系统性更差，基线保留 RoPE。
