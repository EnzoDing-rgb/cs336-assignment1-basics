# Experiment log（CS336 A1）

每次 `uv run python -m cs336_basics.train --config ...` 结束时，`cs336_basics/logging.py` 会自动往本文件追加一小节事实（路径、墙钟、最后 loss、曲线路径）。  
**Notes** 行留给手写观察。学习率 / batch size 的分析报告分别在 `reports/learning_rate.md` 与 `reports/batch_size.md`。

主入口：

```bash
uv run python -m cs336_basics.train --config configs/tinystories_smoke.yaml
# 正式长跑（以后）：
# uv run python -m cs336_basics.train --config configs/tinystories_small.yaml
```

产物布局：

```text
artifacts/checkpoints/<experiment_name>/<YYYYMMDD_HHMM>/
  run_config.yaml
  metrics.csv
  curves/loss_vs_steps.png
  curves/loss_vs_wallclock.png
  ckpt_*.pt
```

---

## Runs

### 2026-07-20 · tinystories_smoke · 20260720_1424

- **Run dir:** `artifacts/checkpoints/tinystories_smoke/20260720_1424/`
- **Wall (train loop):** 1.16 s
- **Last train_loss / valid_loss:** 7.059415 / 7.244363
- **Curves:** `artifacts/checkpoints/tinystories_smoke/20260720_1424/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_small · 20260720_1433

- **Run dir:** `artifacts/checkpoints/tinystories_small/20260720_1433/`
- **Wall (train loop):** 73.61 s
- **Last train_loss / valid_loss:** 2.905658 / 3.350754
- **Curves:** `artifacts/checkpoints/tinystories_small/20260720_1433/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_small · 20260720_1436

- **Run dir:** `artifacts/checkpoints/tinystories_small/20260720_1436/`
- **Wall (train loop):** 1990.97 s
- **Last train_loss / valid_loss:** 1.524046 / 1.527257
- **Curves:** `artifacts/checkpoints/tinystories_small/20260720_1436/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_lr1e-4 · 20260720_1814

- **Run dir:** `artifacts/checkpoints/tinystories_lr1e-4/20260720_1814/`
- **Wall (train loop):** 1988.21 s
- **Last train_loss / valid_loss:** 1.737860 / 1.752369
- **Curves:** `artifacts/checkpoints/tinystories_lr1e-4/20260720_1814/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_lr1.8e-4 · 20260720_1848

- **Run dir:** `artifacts/checkpoints/tinystories_lr1.8e-4/20260720_1848/`
- **Wall (train loop):** 1991.71 s
- **Last train_loss / valid_loss:** 1.611512 / 1.609277
- **Curves:** `artifacts/checkpoints/tinystories_lr1.8e-4/20260720_1848/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_lr5.6e-4 · 20260720_1921

- **Run dir:** `artifacts/checkpoints/tinystories_lr5.6e-4/20260720_1921/`
- **Wall (train loop):** 1989.10 s
- **Last train_loss / valid_loss:** 1.466707 / 1.464548
- **Curves:** `artifacts/checkpoints/tinystories_lr5.6e-4/20260720_1921/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_lr1e-3 · 20260720_1955

- **Run dir:** `artifacts/checkpoints/tinystories_lr1e-3/20260720_1955/`
- **Wall (train loop):** 1991.03 s
- **Last train_loss / valid_loss:** 1.451617 / 1.435667
- **Curves:** `artifacts/checkpoints/tinystories_lr1e-3/20260720_1955/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_lr1.8e-3 · 20260720_2028

- **Run dir:** `artifacts/checkpoints/tinystories_lr1.8e-3/20260720_2028/`
- **Wall (train loop):** 1989.50 s
- **Last train_loss / valid_loss:** 1.425766 / 1.425355
- **Curves:** `artifacts/checkpoints/tinystories_lr1.8e-3/20260720_2028/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_lr3.2e-3 · 20260720_2102

- **Run dir:** `artifacts/checkpoints/tinystories_lr3.2e-3/20260720_2102/`
- **Wall (train loop):** 1992.94 s
- **Last train_loss / valid_loss:** 1.441573 / 1.436217
- **Curves:** `artifacts/checkpoints/tinystories_lr3.2e-3/20260720_2102/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_lr5.6e-3 · 20260720_2145

- **Run dir:** `artifacts/checkpoints/tinystories_lr5.6e-3/20260720_2145/`
- **Wall (train loop):** 1994.73 s
- **Last train_loss / valid_loss:** 1.449469 / 1.459307
- **Curves:** `artifacts/checkpoints/tinystories_lr5.6e-3/20260720_2145/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_lr1e-2 · 20260720_2219

- **Run dir:** `artifacts/checkpoints/tinystories_lr1e-2/20260720_2219/`
- **Wall (train loop):** 1995.64 s
- **Last train_loss / valid_loss:** 1.513649 / 1.515470
- **Curves:** `artifacts/checkpoints/tinystories_lr1e-2/20260720_2219/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_bs8_memprobe · 20260720_2329

- **Run dir:** `artifacts/checkpoints/tinystories_bs8_memprobe/20260720_2329/`
- **Wall (train loop):** 0.60 s
- **Last train_loss / valid_loss:** 9.241344 / —
- **Curves:** `artifacts/checkpoints/tinystories_bs8_memprobe/20260720_2329/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_bs16_memprobe · 20260720_2329

- **Run dir:** `artifacts/checkpoints/tinystories_bs16_memprobe/20260720_2329/`
- **Wall (train loop):** 0.59 s
- **Last train_loss / valid_loss:** 9.245067 / —
- **Curves:** `artifacts/checkpoints/tinystories_bs16_memprobe/20260720_2329/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_bs32_memprobe · 20260720_2329

- **Run dir:** `artifacts/checkpoints/tinystories_bs32_memprobe/20260720_2329/`
- **Wall (train loop):** 0.67 s
- **Last train_loss / valid_loss:** 9.253831 / —
- **Curves:** `artifacts/checkpoints/tinystories_bs32_memprobe/20260720_2329/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_bs64_memprobe · 20260720_2330

- **Run dir:** `artifacts/checkpoints/tinystories_bs64_memprobe/20260720_2330/`
- **Wall (train loop):** 0.62 s
- **Last train_loss / valid_loss:** 9.250597 / —
- **Curves:** `artifacts/checkpoints/tinystories_bs64_memprobe/20260720_2330/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_bs128_memprobe · 20260720_2330

- **Run dir:** `artifacts/checkpoints/tinystories_bs128_memprobe/20260720_2330/`
- **Wall (train loop):** 1.01 s
- **Last train_loss / valid_loss:** 9.248495 / —
- **Curves:** `artifacts/checkpoints/tinystories_bs128_memprobe/20260720_2330/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_bs8_probe_0p5x · 20260720_2330

- **Run dir:** `artifacts/checkpoints/tinystories_bs8_probe_0p5x/20260720_2330/`
- **Wall (train loop):** 464.65 s
- **Last train_loss / valid_loss:** 2.198824 / 1.936992
- **Curves:** `artifacts/checkpoints/tinystories_bs8_probe_0p5x/20260720_2330/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_bs8_probe_1x · 20260720_2339

- **Run dir:** `artifacts/checkpoints/tinystories_bs8_probe_1x/20260720_2339/`
- **Wall (train loop):** 455.49 s
- **Last train_loss / valid_loss:** 2.110917 / 1.827727
- **Curves:** `artifacts/checkpoints/tinystories_bs8_probe_1x/20260720_2339/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-20 · tinystories_bs8_probe_2x · 20260720_2347

- **Run dir:** `artifacts/checkpoints/tinystories_bs8_probe_2x/20260720_2347/`
- **Wall (train loop):** 461.03 s
- **Last train_loss / valid_loss:** 2.077428 / 1.802903
- **Curves:** `artifacts/checkpoints/tinystories_bs8_probe_2x/20260720_2347/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs8 · 20260720_2356

- **Run dir:** `artifacts/checkpoints/tinystories_bs8/20260720_2356/`
- **Wall (train loop):** 2545.57 s
- **Last train_loss / valid_loss:** 1.455485 / 1.536667
- **Curves:** `artifacts/checkpoints/tinystories_bs8/20260720_2356/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs16_probe_0p5x · 20260721_0039

- **Run dir:** `artifacts/checkpoints/tinystories_bs16_probe_0p5x/20260721_0039/`
- **Wall (train loop):** 387.11 s
- **Last train_loss / valid_loss:** 1.693516 / 1.779790
- **Curves:** `artifacts/checkpoints/tinystories_bs16_probe_0p5x/20260721_0039/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs16_probe_1x · 20260721_0046

- **Run dir:** `artifacts/checkpoints/tinystories_bs16_probe_1x/20260721_0046/`
- **Wall (train loop):** 390.10 s
- **Last train_loss / valid_loss:** 1.629172 / 1.714236
- **Curves:** `artifacts/checkpoints/tinystories_bs16_probe_1x/20260721_0046/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs16_probe_2x · 20260721_0053

- **Run dir:** `artifacts/checkpoints/tinystories_bs16_probe_2x/20260721_0053/`
- **Wall (train loop):** 389.16 s
- **Last train_loss / valid_loss:** 1.612352 / 1.704074
- **Curves:** `artifacts/checkpoints/tinystories_bs16_probe_2x/20260721_0053/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs16 · 20260721_0101

- **Run dir:** `artifacts/checkpoints/tinystories_bs16/20260721_0101/`
- **Wall (train loop):** 2122.78 s
- **Last train_loss / valid_loss:** 1.662210 / 1.447309
- **Curves:** `artifacts/checkpoints/tinystories_bs16/20260721_0101/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs32_probe_0p5x · 20260721_0136

- **Run dir:** `artifacts/checkpoints/tinystories_bs32_probe_0p5x/20260721_0136/`
- **Wall (train loop):** 363.03 s
- **Last train_loss / valid_loss:** 1.829679 / 1.741065
- **Curves:** `artifacts/checkpoints/tinystories_bs32_probe_0p5x/20260721_0136/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs32_probe_1x · 20260721_0143

- **Run dir:** `artifacts/checkpoints/tinystories_bs32_probe_1x/20260721_0143/`
- **Wall (train loop):** 362.93 s
- **Last train_loss / valid_loss:** 1.793127 / 1.703201
- **Curves:** `artifacts/checkpoints/tinystories_bs32_probe_1x/20260721_0143/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs32_probe_2x · 20260721_0150

- **Run dir:** `artifacts/checkpoints/tinystories_bs32_probe_2x/20260721_0150/`
- **Wall (train loop):** 363.99 s
- **Last train_loss / valid_loss:** 1.854735 / 1.760772
- **Curves:** `artifacts/checkpoints/tinystories_bs32_probe_2x/20260721_0150/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs32 · 20260721_0157

- **Run dir:** `artifacts/checkpoints/tinystories_bs32/20260721_0157/`
- **Wall (train loop):** 1991.90 s
- **Last train_loss / valid_loss:** 1.425766 / 1.425355
- **Curves:** `artifacts/checkpoints/tinystories_bs32/20260721_0157/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs64_probe_0p5x · 20260721_0230

- **Run dir:** `artifacts/checkpoints/tinystories_bs64_probe_0p5x/20260721_0230/`
- **Wall (train loop):** 354.13 s
- **Last train_loss / valid_loss:** 1.653550 / 1.702425
- **Curves:** `artifacts/checkpoints/tinystories_bs64_probe_0p5x/20260721_0230/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs64_probe_1x · 20260721_0237

- **Run dir:** `artifacts/checkpoints/tinystories_bs64_probe_1x/20260721_0237/`
- **Wall (train loop):** 354.48 s
- **Last train_loss / valid_loss:** 1.657266 / 1.704473
- **Curves:** `artifacts/checkpoints/tinystories_bs64_probe_1x/20260721_0237/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs64_probe_2x · 20260721_0244

- **Run dir:** `artifacts/checkpoints/tinystories_bs64_probe_2x/20260721_0244/`
- **Wall (train loop):** 354.81 s
- **Last train_loss / valid_loss:** 2.014684 / 2.047404
- **Curves:** `artifacts/checkpoints/tinystories_bs64_probe_2x/20260721_0244/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs64 · 20260721_0250

- **Run dir:** `artifacts/checkpoints/tinystories_bs64/20260721_0250/`
- **Wall (train loop):** 1943.93 s
- **Last train_loss / valid_loss:** 1.440831 / 1.428722
- **Curves:** `artifacts/checkpoints/tinystories_bs64/20260721_0250/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs128_probe_0p5x · 20260721_0323

- **Run dir:** `artifacts/checkpoints/tinystories_bs128_probe_0p5x/20260721_0323/`
- **Wall (train loop):** 349.45 s
- **Last train_loss / valid_loss:** 1.701405 / 1.716366
- **Curves:** `artifacts/checkpoints/tinystories_bs128_probe_0p5x/20260721_0323/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs128_probe_1x · 20260721_0329

- **Run dir:** `artifacts/checkpoints/tinystories_bs128_probe_1x/20260721_0329/`
- **Wall (train loop):** 350.82 s
- **Last train_loss / valid_loss:** 2.036001 / 2.043405
- **Curves:** `artifacts/checkpoints/tinystories_bs128_probe_1x/20260721_0329/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs128_probe_2x · 20260721_0336

- **Run dir:** `artifacts/checkpoints/tinystories_bs128_probe_2x/20260721_0336/`
- **Wall (train loop):** 350.06 s
- **Last train_loss / valid_loss:** 2.298668 / 2.307792
- **Curves:** `artifacts/checkpoints/tinystories_bs128_probe_2x/20260721_0336/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs128 · 20260721_0343

- **Run dir:** `artifacts/checkpoints/tinystories_bs128/20260721_0343/`
- **Wall (train loop):** 1916.27 s
- **Last train_loss / valid_loss:** 1.440197 / 1.411149
- **Curves:** `artifacts/checkpoints/tinystories_bs128/20260721_0343/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs256_memprobe · 20260721_0416

- **Run dir:** `artifacts/checkpoints/tinystories_bs256_memprobe/20260721_0416/`
- **Wall (train loop):** 2.41 s
- **Last train_loss / valid_loss:** 9.249886 / —
- **Curves:** `artifacts/checkpoints/tinystories_bs256_memprobe/20260721_0416/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs256_probe_1x · 20260721_0416

- **Run dir:** `artifacts/checkpoints/tinystories_bs256_probe_1x/20260721_0416/`
- **Wall (train loop):** 363.33 s
- **Last train_loss / valid_loss:** 2.491154 / 2.464243
- **Curves:** `artifacts/checkpoints/tinystories_bs256_probe_1x/20260721_0416/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs256_probe_2x · 20260721_0423

- **Run dir:** `artifacts/checkpoints/tinystories_bs256_probe_2x/20260721_0423/`
- **Wall (train loop):** 362.92 s
- **Last train_loss / valid_loss:** 2.977844 / 2.955323
- **Curves:** `artifacts/checkpoints/tinystories_bs256_probe_2x/20260721_0423/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs256 · 20260721_0429

- **Run dir:** `artifacts/checkpoints/tinystories_bs256/20260721_0429/`
- **Wall (train loop):** 1994.82 s
- **Last train_loss / valid_loss:** 1.652409 / 1.633987
- **Curves:** `artifacts/checkpoints/tinystories_bs256/20260721_0429/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs512_probe_0p5x · 20260721_1231

- **Run dir:** `artifacts/checkpoints/tinystories_bs512_probe_0p5x/20260721_1231/`
- **Wall (train loop):** 393.12 s
- **Last train_loss / valid_loss:** 2.924090 / 2.900208
- **Curves:** `artifacts/checkpoints/tinystories_bs512_probe_0p5x/20260721_1231/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs512_probe_1x · 20260721_1238

- **Run dir:** `artifacts/checkpoints/tinystories_bs512_probe_1x/20260721_1238/`
- **Wall (train loop):** 392.94 s
- **Last train_loss / valid_loss:** 3.534798 / 3.509894
- **Curves:** `artifacts/checkpoints/tinystories_bs512_probe_1x/20260721_1238/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs512_probe_2x · 20260721_1246

- **Run dir:** `artifacts/checkpoints/tinystories_bs512_probe_2x/20260721_1246/`
- **Wall (train loop):** 391.61 s
- **Last train_loss / valid_loss:** 3.551396 / 3.521709
- **Curves:** `artifacts/checkpoints/tinystories_bs512_probe_2x/20260721_1246/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_bs512 · 20260721_1253

- **Run dir:** `artifacts/checkpoints/tinystories_bs512/20260721_1253/`
- **Wall (train loop):** 2081.92 s
- **Last train_loss / valid_loss:** 1.871347 / 1.888500
- **Curves:** `artifacts/checkpoints/tinystories_bs512/20260721_1253/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_post_norm_lr1.8e-3 · 20260721_1703

- **Run dir:** `artifacts/checkpoints/tinystories_post_norm_lr1.8e-3/20260721_1703/`
- **Wall (train loop):** — s
- **Last train_loss / valid_loss:** — / —
- **Curves:** `artifacts/checkpoints/tinystories_post_norm_lr1.8e-3/20260721_1703/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_post_norm_lr1e-3 · 20260721_1703

- **Run dir:** `artifacts/checkpoints/tinystories_post_norm_lr1e-3/20260721_1703/`
- **Wall (train loop):** — s
- **Last train_loss / valid_loss:** — / —
- **Curves:** `artifacts/checkpoints/tinystories_post_norm_lr1e-3/20260721_1703/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_post_norm_lr5.6e-4 · 20260721_1703

- **Run dir:** `artifacts/checkpoints/tinystories_post_norm_lr5.6e-4/20260721_1703/`
- **Wall (train loop):** — s
- **Last train_loss / valid_loss:** — / —
- **Curves:** `artifacts/checkpoints/tinystories_post_norm_lr5.6e-4/20260721_1703/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_post_norm_lr1.8e-4 · 20260721_1703

- **Run dir:** `artifacts/checkpoints/tinystories_post_norm_lr1.8e-4/20260721_1703/`
- **Wall (train loop):** — s
- **Last train_loss / valid_loss:** — / —
- **Curves:** `artifacts/checkpoints/tinystories_post_norm_lr1.8e-4/20260721_1703/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_none_norm_lr1.8e-3 · 20260721_1703

- **Run dir:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-3/20260721_1703/`
- **Wall (train loop):** — s
- **Last train_loss / valid_loss:** — / —
- **Curves:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-3/20260721_1703/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_none_norm_lr1e-3 · 20260721_1703

- **Run dir:** `artifacts/checkpoints/tinystories_none_norm_lr1e-3/20260721_1703/`
- **Wall (train loop):** — s
- **Last train_loss / valid_loss:** — / —
- **Curves:** `artifacts/checkpoints/tinystories_none_norm_lr1e-3/20260721_1703/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_none_norm_lr5.6e-4 · 20260721_1703

- **Run dir:** `artifacts/checkpoints/tinystories_none_norm_lr5.6e-4/20260721_1703/`
- **Wall (train loop):** — s
- **Last train_loss / valid_loss:** — / —
- **Curves:** `artifacts/checkpoints/tinystories_none_norm_lr5.6e-4/20260721_1703/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_none_norm_lr1.8e-4 · 20260721_1703

- **Run dir:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-4/20260721_1703/`
- **Wall (train loop):** — s
- **Last train_loss / valid_loss:** — / —
- **Curves:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-4/20260721_1703/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_post_norm_lr1.8e-3 · 20260721_1707

- **Run dir:** `artifacts/checkpoints/tinystories_post_norm_lr1.8e-3/20260721_1707/`
- **Wall (train loop):** 1992.15 s
- **Last train_loss / valid_loss:** 1.478081 / 1.468839
- **Curves:** `artifacts/checkpoints/tinystories_post_norm_lr1.8e-3/20260721_1707/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_post_norm_lr1e-3 · 20260721_1741

- **Run dir:** `artifacts/checkpoints/tinystories_post_norm_lr1e-3/20260721_1741/`
- **Wall (train loop):** 1992.00 s
- **Last train_loss / valid_loss:** 1.456050 / 1.449351
- **Curves:** `artifacts/checkpoints/tinystories_post_norm_lr1e-3/20260721_1741/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_post_norm_lr5.6e-4 · 20260721_1814

- **Run dir:** `artifacts/checkpoints/tinystories_post_norm_lr5.6e-4/20260721_1814/`
- **Wall (train loop):** 1990.57 s
- **Last train_loss / valid_loss:** 1.469007 / 1.462197
- **Curves:** `artifacts/checkpoints/tinystories_post_norm_lr5.6e-4/20260721_1814/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_post_norm_lr1.8e-4 · 20260721_1848

- **Run dir:** `artifacts/checkpoints/tinystories_post_norm_lr1.8e-4/20260721_1848/`
- **Wall (train loop):** 1988.92 s
- **Last train_loss / valid_loss:** 1.613479 / 1.603410
- **Curves:** `artifacts/checkpoints/tinystories_post_norm_lr1.8e-4/20260721_1848/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_pre_norm_lr1.8e-2 · 20260721_1922

- **Run dir:** `artifacts/checkpoints/tinystories_pre_norm_lr1.8e-2/20260721_1922/`
- **Wall (train loop):** 1993.87 s
- **Last train_loss / valid_loss:** 1.641321 / 1.647284
- **Curves:** `artifacts/checkpoints/tinystories_pre_norm_lr1.8e-2/20260721_1922/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_pre_norm_lr9e-2 · 20260721_1955

- **Run dir:** `artifacts/checkpoints/tinystories_pre_norm_lr9e-2/20260721_1955/`
- **Wall (train loop):** 2002.02 s
- **Last train_loss / valid_loss:** 2.194704 / 2.235170
- **Curves:** `artifacts/checkpoints/tinystories_pre_norm_lr9e-2/20260721_1955/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_post_norm_lr1.8e-2 · 20260721_2029

- **Run dir:** `artifacts/checkpoints/tinystories_post_norm_lr1.8e-2/20260721_2029/`
- **Wall (train loop):** 1989.92 s
- **Last train_loss / valid_loss:** 1.756684 / 1.764359
- **Curves:** `artifacts/checkpoints/tinystories_post_norm_lr1.8e-2/20260721_2029/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_post_norm_lr9e-2 · 20260721_2103

- **Run dir:** `artifacts/checkpoints/tinystories_post_norm_lr9e-2/20260721_2103/`
- **Wall (train loop):** 621.92 s
- **Last train_loss / valid_loss:** 22.099289 / 5.836689
- **Curves:** `artifacts/checkpoints/tinystories_post_norm_lr9e-2/20260721_2103/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_none_norm_lr1.8e-4 · 20260721_2113

- **Run dir:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-4/20260721_2113/`
- **Wall (train loop):** 0.60 s
- **Last train_loss / valid_loss:** 21.182518 / —
- **Curves:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-4/20260721_2113/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_none_norm_lr1.8e-3 · 20260721_2113

- **Run dir:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-3/20260721_2113/`
- **Wall (train loop):** 0.56 s
- **Last train_loss / valid_loss:** 21.182518 / —
- **Curves:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-3/20260721_2113/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_none_norm_lr1.8e-2 · 20260721_2113

- **Run dir:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-2/20260721_2113/`
- **Wall (train loop):** 0.56 s
- **Last train_loss / valid_loss:** 21.182518 / —
- **Curves:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-2/20260721_2113/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_none_norm_lr9e-2 · 20260721_2113

- **Run dir:** `artifacts/checkpoints/tinystories_none_norm_lr9e-2/20260721_2113/`
- **Wall (train loop):** 0.53 s
- **Last train_loss / valid_loss:** 21.182518 / —
- **Curves:** `artifacts/checkpoints/tinystories_none_norm_lr9e-2/20260721_2113/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_post_norm_lr9e-2 · 20260721_2118

- **Run dir:** `artifacts/checkpoints/tinystories_post_norm_lr9e-2/20260721_2118/`
- **Wall (train loop):** 1995.60 s
- **Last train_loss / valid_loss:** 5.796884 / 5.791701
- **Curves:** `artifacts/checkpoints/tinystories_post_norm_lr9e-2/20260721_2118/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_none_norm_lr1.8e-4 · 20260721_2152

- **Run dir:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-4/20260721_2152/`
- **Wall (train loop):** 1900.71 s
- **Last train_loss / valid_loss:** 1.640495 / 1.648220
- **Curves:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-4/20260721_2152/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_none_norm_lr1.8e-3 · 20260721_2224

- **Run dir:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-3/20260721_2224/`
- **Wall (train loop):** 79.99 s
- **Last train_loss / valid_loss:** 6007.055664 / 2.682920
- **Curves:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-3/20260721_2224/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_none_norm_lr1.8e-2 · 20260721_2225

- **Run dir:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-2/20260721_2225/`
- **Wall (train loop):** 18.56 s
- **Last train_loss / valid_loss:** 1199.653320 / —
- **Curves:** `artifacts/checkpoints/tinystories_none_norm_lr1.8e-2/20260721_2225/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-21 · tinystories_none_norm_lr9e-2 · 20260721_2226

- **Run dir:** `artifacts/checkpoints/tinystories_none_norm_lr9e-2/20260721_2226/`
- **Wall (train loop):** 5.18 s
- **Last train_loss / valid_loss:** 16996166.000000 / —
- **Curves:** `artifacts/checkpoints/tinystories_none_norm_lr9e-2/20260721_2226/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-22 · tinystories_rope_b64_lr1.8e-3 · 20260721_2344

- **Run dir:** `artifacts/checkpoints/tinystories_rope_b64_lr1.8e-3/20260721_2344/`
- **Wall (train loop):** 3836.98 s
- **Last train_loss / valid_loss:** 1.395041 / 1.353412
- **Curves:** `artifacts/checkpoints/tinystories_rope_b64_lr1.8e-3/20260721_2344/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-22 · tinystories_no_rope_b64_lr1.8e-3 · 20260722_0049

- **Run dir:** `artifacts/checkpoints/tinystories_no_rope_b64_lr1.8e-3/20260722_0049/`
- **Wall (train loop):** 3684.82 s
- **Last train_loss / valid_loss:** 1.454648 / 1.414034
- **Curves:** `artifacts/checkpoints/tinystories_no_rope_b64_lr1.8e-3/20260722_0049/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_


### 2026-07-22 · tinystories_silu_b64_lr1.8e-3 · 20260722_1152

- **Run dir:** `artifacts/checkpoints/tinystories_silu_b64_lr1.8e-3/20260722_1152/`
- **Wall (train loop):** 3765.03 s
- **Last train_loss / valid_loss:** 1.408039 / 1.367624
- **Curves:** `artifacts/checkpoints/tinystories_silu_b64_lr1.8e-3/20260722_1152/curves/loss_vs_steps.png`, `.../loss_vs_wallclock.png`
- **Notes:** _(fill in: what you changed / what you saw)_

