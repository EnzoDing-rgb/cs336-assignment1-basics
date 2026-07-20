# Experiment log（CS336 A1 · experiment_log）

每次 `uv run python -m cs336_basics.train --config ...` 结束时，`cs336_basics/logging.py` 会 **自动追加** 一小节事实（路径、墙钟、最后 loss、曲线路径）。  
**Notes** 行留给你手写观察；学习性分析仍由人/对话完成，不在这里堆生成器。

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

