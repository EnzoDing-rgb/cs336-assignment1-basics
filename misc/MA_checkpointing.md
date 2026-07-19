# Checkpointing（保存 / 恢复训练状态）

对应 CS336 Assignment 1：`Problem (checkpointing)`。

测试：`uv run pytest -k test_checkpointing`  
适配器：`adapters.run_save_checkpoint` / `adapters.run_load_checkpoint`

目标：你自己写两个函数。本文把 **为啥存、存什么、PyTorch 怎么调用** 直接写清楚，少查文档。

---

## 0. 这题在干什么（不绕）

训练会中断（超时、机器挂、你自己 Ctrl+C）。  
以后也可能想拿「第 5000 步的模型」出来采样。

所以要定期把「接着训所需的一切」写到磁盘；之后能读回来，从断点继续。

一份 checkpoint **至少**要有：

| 东西 | 为啥 |
|------|------|
| 模型权重 | 不然网络参数没了 |
| 优化器状态 | AdamW 有 $m$、$v$、步数 $t$；不存的话动量从头算，续训行为会变 |
| iteration（当前步数） | LR schedule（cosine / warm-up）靠 $t$；不知道停在第几步就接不上 |

本题 **不要** 你发明复杂格式；用 PyTorch 自带的三板斧即可。

---

## 1. 你要用的 PyTorch API（直接抄用法）

### 1.1 模型：取出 / 写回权重

```python
# 取出：一个 dict，key 是参数名，value 是 Tensor
# 例：{"fc1.weight": tensor(...), "fc1.bias": tensor(...), ...}
sd = model.state_dict()

# 写回：把 dict 里的值装进当前这个 model 对象
model.load_state_dict(sd)
```

注意：`load_state_dict` 改的是 **你传入的那个 model 实例**，原地更新参数。  
新模型必须先 `new_model = MyNet(...)` 建好结构，再 `load_state_dict`。

### 1.2 优化器：取出 / 写回状态

```python
# 取出：通常含 "state"（每个参数的 m/v/t 等）和 "param_groups"（lr、betas…）
opt_sd = optimizer.state_dict()

# 写回
optimizer.load_state_dict(opt_sd)
```

`new_optimizer` 也要先用 **同一个模型** 的 `new_model.parameters()` 建好，再 load。  
（参数对象要对上号；测试里就是先建 `new_model`，再用它的 parameters 建 `new_optimizer`。）

### 1.3 整包存盘 / 读盘

```python
import torch

# 存：out 可以是路径字符串，也可以是打开的二进制文件对象
torch.save(obj, out)

# 读：src 同样可以是路径或文件对象
obj = torch.load(src)
```

`obj` 你自己定。最常见、也最省事：一个 **普通 Python dict**，里面既有 tensor，也有 int。

例：

```python
obj = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "iteration": 10,   # 普通 int，torch.save 也能存
}
torch.save(obj, "checkpoint.pt")

ckpt = torch.load("checkpoint.pt")
# ckpt["model"]、ckpt["optimizer"]、ckpt["iteration"]
```

键名你自己定，只要 **save 和 load 用同一套键** 就行。  
测试不检查键名，只检查：load 之后 model / optimizer 的 `state_dict` 和原来一致，且返回的 iteration 对。

---

## 2. 两个函数的接口（作业原文）

```python
def save_checkpoint(model, optimizer, iteration, out) -> None:
    ...

def load_checkpoint(src, model, optimizer) -> int:
    ...
```

| 参数 | 类型 | 含义 |
|------|------|------|
| `model` | `torch.nn.Module` | 要保存 / 要恢复的模型 |
| `optimizer` | `torch.optim.Optimizer` | 要保存 / 要恢复的优化器 |
| `iteration` | `int` | 已经完成的训练步数（测例里训完 10 步后 `it == 10`） |
| `out` / `src` | `str` \| `Path` \| 二进制 file-like | 写入目标 / 读取来源 |

`load_checkpoint` **返回值**：checkpoint 里存的那个 `iteration`（`int`）。

`save_checkpoint` **没有有用返回值**（存完就行）。

---

## 3. 实现时脑子里的数据流

### 3.1 save

```text
model.state_dict()  ──┐
optimizer.state_dict()├──►  装进一个 dict  ──►  torch.save(dict, out)
iteration (int)    ──┘
```

伪代码级别（你自己落成真代码）：

```python
def save_checkpoint(model, optimizer, iteration, out):
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)
```

### 3.2 load

```text
torch.load(src) ──► dict
                      ├── load_state_dict 进 model
                      ├── load_state_dict 进 optimizer
                      └── return iteration
```

伪代码：

```python
def load_checkpoint(src, model, optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]
```

就这点事。难的是「知道要存这三样」；API 就是上面几行。

---

## 4. 玩具例子（形状直觉）

假设：

```python
model = nn.Linear(3, 2)
optimizer = AdamW(model.parameters(), lr=1e-3)
# ... 训练若干步 ...
iteration = 7
```

`model.state_dict()` 大概长这样（示意）：

```text
{
  "weight": Tensor 形状 (2, 3),
  "bias":   Tensor 形状 (2,),
}
```

`optimizer.state_dict()` 大概长这样（示意，AdamW）：

```text
{
  "state": {
      0: {"m": Tensor(...), "v": Tensor(...), "t": 7},  # 第 0 个参数
      1: {"m": ..., "v": ..., "t": 7},
  },
  "param_groups": [ {"lr": 0.001, "betas": (0.9, 0.999), ...} ],
}
```

你 `torch.save` 的大 dict：

```text
{
  "model":      <上面那份 model state_dict>,
  "optimizer":  <上面那份 optimizer state_dict>,
  "iteration":  7,
}
```

load 到 **另一个** `new_model` / `new_optimizer` 后：

- `new_model` 的 weight/bias 数值 = 原来的  
- `new_optimizer` 的 m/v/t = 原来的  
- 函数返回 `7`

---

## 5. 测试在查什么（`test_checkpointing`）

文件：`tests/test_serialization.py`

流程：

1. 建 `_TestNet` + 你的 `AdamW`，训 `num_iters=10` 步，`it` 变成 `10`
2. `run_save_checkpoint(model, optimizer, iteration=it, out=tmp_path/"checkpoint.pt")`
3. **重新**建一个同结构的 `new_model` + `new_optimizer`（随机初始化，和训完的不同）
4. `loaded_iterations = run_load_checkpoint(src=..., model=new_model, optimizer=new_optimizer)`
5. 断言：
   - `loaded_iterations == it`（也就是 `10`）
   - `new_model.state_dict()` 各 tensor ≈ 原 `model`
   - `new_optimizer.state_dict()` 与原 optimizer 一致（含 Adam 的 state）

所以你 load 时必须真的调用了两边的 `load_state_dict`，不能只返回 iteration。

---

## 6. 适配器怎么接

`tests/adapters.py` 里现在是 `raise NotImplementedError`。  
你实现函数后，改成例如：

```python
def run_save_checkpoint(model, optimizer, iteration, out):
    from cs336_basics.... import save_checkpoint  # 你的模块路径
    save_checkpoint(model, optimizer, iteration, out)

def run_load_checkpoint(src, model, optimizer):
    from cs336_basics.... import load_checkpoint
    return load_checkpoint(src, model, optimizer)
```

文件放哪随你，常见选择：

- `cs336_basics/checkpointing.py`（两个函数放一起，名字直白）
- 或塞进已有的 `optimizer.py` 旁边（不太建议，职责混了）

---

## 7. 小坑（写的时候留意）

1. **`torch.load` 的路径 / 文件对象**  
   测例传的是 `pathlib.Path`。`torch.save` / `torch.load` 都认，直接传即可。

2. **新模型必须同结构**  
   `load_state_dict` 按名字对齐；层名对不上会报错或缺 key。

3. **优化器要绑在新模型的 parameters 上再 load**  
   测例已经这样做了；你自己写训练脚本时也要：先建 model，再 `Optimizer(model.parameters())`，再 `load_checkpoint`。

4. **PyTorch 新版本**  
   有的环境 `torch.load` 会提示 `weights_only`。作业测例一般 `torch.load(src)` 默认就能过；若你本地报警告/报错，再查当前版本要求（本题测试不要求你传花活）。

5. **dict 键名**  
   用 `"model"` / `"optimizer"` / `"iteration"` 或 `"model_state_dict"` 都行，**save/load 一致**即可。

---

## 8. 建议你怎么写（自己动手的顺序）

1. 新建模块，写 `save_checkpoint` / `load_checkpoint`（§3 伪代码落地）  
2. 接上两个 adapter  
3. 跑：`uv run pytest -k test_checkpointing`  
4. 过了就把模块路径记进训练脚本：每隔 N step `save_checkpoint(...)`；重启时 `it = load_checkpoint(...)` 再接着训

---

## 9. 一句话

Checkpoint = `torch.save` 一个 dict，里面是 `model.state_dict()` + `optimizer.state_dict()` + `iteration`；  
恢复 = `torch.load` 再两边 `load_state_dict`，并 `return iteration`。
