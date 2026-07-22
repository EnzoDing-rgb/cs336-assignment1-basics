# Norm placement ablation（TinyStories）

**问题：** 在同一套 TinyStories 小模型与 LR 网格下，把 RMSNorm 放在残差前（`pre_norm`）、残差后（`post_norm`）、或干脆去掉（`none_norm`），训练稳不稳、最终 loss 差多少？

**交付图：**  
[`artifacts/plots/norm_ablation/norm_ablation_summary.png`](../artifacts/plots/norm_ablation/norm_ablation_summary.png)  
（**色系=placement**：蓝 `pre_norm` / 橙 `post_norm` / 红 `none_norm`；**深浅 + 末端 `lr=` 标签** = learning rate。实线=收得还行，虚线=跑满但差，▲+标注框=炸。y：train loss，截断 10.5。）

---

## 1. 设置

| 固定 | 取值 |
|------|------|
| 底座 | `configs/tinystories_small.yaml`（`B=32`，`max_iters=20000`，`grad_clip=1.0`，warmup 1000 → cosine） |
| 模型 | 同底座；只改 `model.norm_placement` |
| 日程 | `lr_min = lr_max / 10` |
| 提前停 | 非有限 loss，或 `train_loss > 200`（见下） |

**三种 placement（API 名固定写全名）：**

| `norm_placement` | block | 出口 `ln_final` |
|------------------|--------|-----------------|
| `pre_norm` | \(x + f(\mathrm{RMSNorm}(x))\) | 保留 |
| `post_norm` | \(\mathrm{RMSNorm}(x + f(x))\) | 保留 |
| `none_norm` | \(x + f(x)\) | 去掉（`Identity`） |

**4×3 网格：** `lr_max ∈ {1.8e-4, 1.8e-3, 1.8e-2, 9e-2}` × 上表三种。  
`pre_norm @ 1.8e-4 / 1.8e-3` 复用 LR 扫里的同配置长跑；其余从零训。

脚本：`scripts/sweep_norm_ablation.py`、`scripts/plot_norm_ablation.py`。

**为何阈值是 200 而不是 20：**  
`none_norm` 在随机初始化、尚未更新时，step-0 CE 已约 **21**（`pre`/`post` 约 **9.25** ≈ \(\log 10000\)）。阈值 20 会把「开局尺度差」误判成「已炸」并在 iter 0 杀掉；200 才让小 LR 的 `none_norm` 有机会往下爬，同时仍能截住真正失控的尖峰。

---

## 2. 结果总表

`final` 取收尾附近（step ≈ 19800）的 train / valid；炸的 run 记 abort step 与最后一次记下的 train loss。

| `lr_max` | `pre_norm` train / valid | `post_norm` train / valid | `none_norm` |
|---------:|-------------------------:|--------------------------:|-------------|
| 1.8e-4 | 1.612 / **1.609** | 1.613 / **1.603** | ✓ 跑满：1.640 / **1.648**（step-0 从 ~21 爬回） |
| **1.8e-3** | **1.426 / 1.425**（网格最优） | 1.478 / 1.469 | ✗ abort @ 833，train ≈ 6007 |
| 1.8e-2 | 1.641 / 1.647 | 1.757 / 1.764 | ✗ abort @ 189，train ≈ 1200 |
| 9e-2 | 2.195 / 2.235 | ✓ 跑满但蹲死：~**5.80** / **5.79** | ✗ abort @ 47，train ≈ \(1.7\times 10^7\) |

---

## 3. 发现

1. **`pre_norm` 最稳、LR 最宽。**  
   四档全部跑满；最佳仍在 **`1.8e-3`（valid ≈ 1.425）**，与 [learning_rate.md](learning_rate.md) 锚点一致。即便 `9e-2` 也能收在 ~2.2，没有熔断。

2. **`post_norm` 中低 LR 可用，高 LR 脆。**  
   `1.8e-4` 与 `pre` 几乎打平；`1.8e-3` 略差一截（1.47 vs 1.43）；`1.8e-2` 再拉开。到 **`9e-2`**：训练没越过 200 的 abort 线，但 loss 从早期就卡在 **~5.8**，等于「活着却没学会」——比 `pre` 同档差一大截。

3. **`none_norm` 不是「不能训」，而是「几乎没有 LR 容错」。**  
   开局 CE≈21 来自缺 norm 后的激活/logit 尺度，不是某个大 LR 独有的现象。  
   - 只有 **`1.8e-4`** 能从 ~21 压回 ~1.65 并跑满；  
   - 从 `1.8e-3` 起，更新一猛就真炸，且 LR 越大炸得越早、尖峰越高。  
   有 norm 时，同网格里 `1.8e-3` 反而是甜点——去掉 norm 以后，甜点直接左移到「极小步长」一侧。

4. **一句话取舍：**  
   默认应继续用 **`pre_norm`**；`post_norm` 在温和 LR 下可作对照，但不要当高 LR 的安全带；`none_norm` 仅在刻意做稳定性消融时有意义，且必须用明显更小的 LR。

---

## 4. 一句话

同预算 4×3 扫下来：**`pre_norm` 全面更稳更好**；`post_norm` 在中低 LR 接近 `pre`，在 `9e-2` 学废；`none_norm` 开局 CE≈21，仅 `1.8e-4` 能救回来，再大必炸。
