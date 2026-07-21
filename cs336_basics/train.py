"""Training entrypoint: config file is the source of truth; CLI only picks config (+ light overrides).

Usage:
  uv run python -m cs336_basics.train --config configs/tinystories_small.yaml
  uv run python -m cs336_basics.train --config configs/tinystories_small.yaml --override train.device=cpu
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.data_loader import get_batch
from cs336_basics.logging import RunLogger, make_run_dir
from cs336_basics.model.normalization import cross_entropy
from cs336_basics.model.optimizer import AdamW, clip_gradients, get_lr_cosine_schedule
from cs336_basics.model.transformer import TransformerLM, compute_d_ff


# ---------------------------------------------------------------------------
# Config dataclasses (YAML nested dict → typed objects)
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    train_path: str
    valid_path: str


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    rope_theta: float
    d_ff: int | None = None  # None → compute_d_ff(d_model)
    # pre_norm | post_norm | none_norm（默认与作业基线一致）
    norm_placement: str = "pre_norm"


@dataclass
class OptimConfig:
    lr_max: float
    lr_min: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    warmup_iters: int
    cosine_cycle_iters: int
    grad_clip: float


@dataclass
class TrainLoopConfig:
    batch_size: int
    max_iters: int
    eval_interval: int
    eval_batches: int
    checkpoint_interval: int
    checkpoint_dir: str
    device: str
    seed: int
    resume_path: str | None = None


@dataclass
class LoggingConfig:
    log_interval: int
    wandb: bool
    wandb_project: str


@dataclass
class TrainConfig:
    experiment_name: str
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainLoopConfig
    logging: LoggingConfig


def _as_tuple2(x: Any) -> tuple[float, float]:
    a, b = x
    return (float(a), float(b))


def dict_to_train_config(d: dict[str, Any]) -> TrainConfig:
    """例：YAML 顶层 dict → TrainConfig（嵌套 dataclass）。"""
    optim = d["optim"]
    train = d["train"]
    model = d["model"]
    return TrainConfig(
        experiment_name=str(d["experiment_name"]),
        data=DataConfig(**d["data"]),
        model=ModelConfig(
            vocab_size=int(model["vocab_size"]),
            context_length=int(model["context_length"]),
            d_model=int(model["d_model"]),
            num_layers=int(model["num_layers"]),
            num_heads=int(model["num_heads"]),
            rope_theta=float(model["rope_theta"]),
            d_ff=int(model["d_ff"]) if model.get("d_ff") is not None else None,
            norm_placement=str(model.get("norm_placement", "pre_norm")),
        ),
        optim=OptimConfig(
            lr_max=float(optim["lr_max"]),
            lr_min=float(optim["lr_min"]),
            weight_decay=float(optim["weight_decay"]),
            betas=_as_tuple2(optim["betas"]),
            eps=float(optim["eps"]),
            warmup_iters=int(optim["warmup_iters"]),
            cosine_cycle_iters=int(optim["cosine_cycle_iters"]),
            grad_clip=float(optim["grad_clip"]),
        ),
        train=TrainLoopConfig(
            batch_size=int(train["batch_size"]),
            max_iters=int(train["max_iters"]),
            eval_interval=int(train["eval_interval"]),
            eval_batches=int(train["eval_batches"]),
            checkpoint_interval=int(train["checkpoint_interval"]),
            checkpoint_dir=str(train["checkpoint_dir"]),
            device=str(train["device"]),
            seed=int(train["seed"]),
            resume_path=train.get("resume_path"),
        ),
        logging=LoggingConfig(**d["logging"]),
    )


def _set_by_dotted_key(d: dict[str, Any], dotted: str, value: Any) -> None:
    """例：dotted='train.device', value='cpu' → d['train']['device'] = 'cpu'."""
    keys = dotted.split(".")
    cur: dict[str, Any] = d
    for k in keys[:-1]:
        cur = cur[k]
    # YAML 标量：尝试 json 解析，否则当字符串
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    cur[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> TrainConfig:
    """读 YAML，可选 key=value 覆盖，再转成 TrainConfig。

    例：
      load_config("configs/tinystories_small.yaml")
      load_config(..., overrides=["train.device=cpu", "train.batch_size=8"])
    """
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    for item in overrides or []:
        key, _, val = item.partition("=")
        if not key or _ == "":
            raise ValueError(f"override must look like key=value, got {item!r}")
        _set_by_dotted_key(raw, key, val)

    return dict_to_train_config(raw)


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


# ---------------------------------------------------------------------------
# Data / model / optimizer builders
# ---------------------------------------------------------------------------


def open_memmap_dataset(path: str | Path) -> np.memmap:
    """用 mmap 打开 token 数组，假装整库在内存，按需读盘。

    例：path → artifacts/.../tinystories_train.npy
        返回 np.memmap，1D int，len = 语料 token 数
    """
    path = Path(path)
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 1:
        raise ValueError(f"expected 1D token array, got shape {arr.shape} from {path}")
    return arr


def build_model(cfg: TrainConfig) -> TransformerLM:
    m = cfg.model
    d_ff = m.d_ff if m.d_ff is not None else compute_d_ff(m.d_model)
    device = torch.device(cfg.train.device)
    return TransformerLM(
        vocab_size=m.vocab_size,
        context_length=m.context_length,
        d_model=m.d_model,
        num_layers=m.num_layers,
        num_heads=m.num_heads,
        d_ff=d_ff,
        rope_theta=m.rope_theta,
        norm_placement=m.norm_placement,  # type: ignore[arg-type]
        device=device,
    )


def build_optimizer(cfg: TrainConfig, model: torch.nn.Module) -> AdamW:
    o = cfg.optim
    return AdamW(
        model.parameters(),
        lr=o.lr_max,
        betas=o.betas,
        eps=o.eps,
        weight_decay=o.weight_decay,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Train / eval steps
# ---------------------------------------------------------------------------


def train_step(
    model: TransformerLM,
    optimizer: AdamW,
    dataset: np.ndarray,
    cfg: TrainConfig,
    iteration: int,
) -> float:
    """一次参数更新（这才是「训练一步」的实现）。

    流程：get_batch → 设 lr → forward → CE → backward → clip → optimizer.step
    返回：Python float，本 step 的标量 loss。

    注意下面的 model.train()：那是 PyTorch nn.Module 自带的开关，不是空函数、
    也不需要我们实现。它只做一件事：把模块标成「训练模式」(training=True)。
    例：有 Dropout / BatchNorm 时，train 模式与 eval 模式行为不同；
        我们这套 TransformerLM 目前主要靠它表达意图，和 model.eval() 成对使用。
    """
    # PyTorch 内置：model.train() ≡ 设置 self.training = True（递归到子模块）
    # 对比：model.eval() ≡ self.training = False。都不是「跑训练循环」。
    model.train()
    device = cfg.train.device
    x, y = get_batch(
        dataset,
        batch_size=cfg.train.batch_size,
        context_length=cfg.model.context_length,
        device=device,
    )
    # 本步学习率写入每个 param group（cosine schedule）
    # 例：iteration=0 → lr 常为 0（warm-up 起点）；iteration 增大后爬到 lr_max
    lr = get_lr_cosine_schedule(
        it=iteration,
        max_learning_rate=cfg.optim.lr_max,
        min_learning_rate=cfg.optim.lr_min,
        warmup_iters=cfg.optim.warmup_iters,
        cosine_cycle_iters=cfg.optim.cosine_cycle_iters,
    )
    for group in optimizer.param_groups:
        group["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    logits = model(x)  # torch.Tensor，形状 (B, m, V)，dtype float
    # cross_entropy 要 (N, V) 与 (N,)：把 batch×seq 摊平
    # 例：B=2, m=4, V=10000 → logits.view(8, 10000), y.view(8,)
    loss = cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
    )
    loss.backward()
    clip_gradients(model.parameters(), cfg.optim.grad_clip)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    cfg: TrainConfig,
) -> float:
    """验证集上估平均 CE（不反传、不改参数）。

    model.eval() 同样是 PyTorch 内置模式开关（training=False），不是待实现函数。
    """
    model.eval()
    total = 0.0
    n = cfg.train.eval_batches
    for _ in range(n):
        x, y = get_batch(
            dataset,
            batch_size=cfg.train.batch_size,
            context_length=cfg.model.context_length,
            device=cfg.train.device,
        )
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total += float(loss.item())
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CS336 A1 training (config-file driven)")
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config, e.g. configs/tinystories_small.yaml",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted override, e.g. train.device=cpu (repeatable)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg.train.seed)

    # 每次 run：checkpoint_dir / YYYYMMDD_HHMM/ （到分钟；同分钟冲突则 _2）
    # 例：artifacts/checkpoints/tinystories_smoke/20260720_1419/
    run_dir = make_run_dir(cfg.train.checkpoint_dir)
    logger = RunLogger.create(run_dir)
    # 入口仍是 --config configs/...；run_config.yaml 是快照，不要用来启动训练。
    with open(run_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)
    print(f"[run] dir={run_dir}")

    train_data = open_memmap_dataset(cfg.data.train_path)
    valid_data = open_memmap_dataset(cfg.data.valid_path)
    print(
        f"[data] train_len={len(train_data):,} valid_len={len(valid_data):,} "
        f"dtype={train_data.dtype}"
    )

    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    start_iter = 0
    if cfg.train.resume_path:
        start_iter = load_checkpoint(cfg.train.resume_path, model, optimizer)
        print(f"[resume] loaded iteration={start_iter} from {cfg.train.resume_path}")

    use_wandb = cfg.logging.wandb
    if use_wandb:
        import wandb

        wandb.init(project=cfg.logging.wandb_project, name=cfg.experiment_name, config=config_to_dict(cfg))

    model.to(cfg.train.device)
    print(f"[train] device={cfg.train.device} max_iters={cfg.train.max_iters} start={start_iter}")
    # 墙钟从训练循环起算（不含建模型/读数据），曲线横轴才有意义
    logger.t0 = time.perf_counter()

    try:
        for it in range(start_iter, cfg.train.max_iters):
            # loss: Python float，本 step 的 CE。例：9.2413
            loss = train_step(model, optimizer, train_data, cfg, iteration=it)

            if it % cfg.logging.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"iter {it:6d}  train_loss={loss:.4f}  lr={lr:.6e}  wall={logger.wall_s():.1f}s")
                # → metrics.csv 一行 train；例 step=0, train_loss=9.24, lr=0
                logger.log_train(it, loss, lr)
                if use_wandb:
                    import wandb

                    wandb.log({"train/loss": loss, "train/lr": lr, "iter": it}, step=it)

            # 提前停：数值崩坏，或 train loss 已坏到不值得继续空转
            if not math.isfinite(loss) or loss > 20.0:
                lr = optimizer.param_groups[0]["lr"]
                if it % cfg.logging.log_interval != 0:
                    logger.log_train(it, loss, lr)
                reason = "non-finite loss" if not math.isfinite(loss) else f"train_loss={loss:.4f}>20"
                print(f"[abort] iter={it} {reason}  wall={logger.wall_s():.1f}s", flush=True)
                break

            if it > 0 and it % cfg.train.eval_interval == 0:
                # val_loss: 验证集若干 batch 平均 CE。例：8.4361
                val_loss = evaluate(model, valid_data, cfg)
                print(f"iter {it:6d}  valid_loss={val_loss:.4f}  wall={logger.wall_s():.1f}s")
                # → metrics.csv 一行 valid（同 step 可有 train+valid 两行）
                logger.log_valid(it, val_loss)
                if use_wandb:
                    import wandb

                    wandb.log({"valid/loss": val_loss, "iter": it}, step=it)

            if it > 0 and it % cfg.train.checkpoint_interval == 0:
                out = run_dir / f"ckpt_iter{it}.pt"
                # 存「已完成」的步数：下一轮从 it+1 接着训
                save_checkpoint(model, optimizer, iteration=it + 1, out=out)
                print(f"[ckpt] wrote {out}")

        else:
            # final checkpoint（正常跑满才写）
            final_path = run_dir / f"ckpt_iter{cfg.train.max_iters}.pt"
            save_checkpoint(model, optimizer, iteration=cfg.train.max_iters, out=final_path)
            print(f"[done] final checkpoint {final_path}")
    finally:
        # 画 curves/*.png，并往 misc/experiment_log.md 追加一小节事实
        logger.finalize(experiment_name=cfg.experiment_name)
        logger.close()
        if use_wandb:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
