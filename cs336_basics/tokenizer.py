from __future__ import annotations

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 返回值是「一个二元组」
    #   第 1 项 vocab:  dict[int, bytes]
    #   第 2 项 merges: list[tuple[bytes, bytes]]

    vocab = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    corpus = read_corpus(input_path)
    # segments = split_special_tokens(corpus, special_tokens)
    # pretokens = pretokenize(segments)
    # merges = merge_loop(pretokens, vocab, vocab_size, special_tokens)

    return vocab, merges


def split_special_tokens(corpus: str, special_tokens: list[str]) -> list[str]:
    """
    按 special token 切开大字符串，返回「多段普通文本」。

    ── 输入 ──
    corpus: str
        read_corpus 读出来的整份语料，一个巨大的字符串。
        里面可能夹着很多篇文档，文档之间用 <|endoftext|> 之类标记分隔。

    special_tokens: list[str]
        调用方传进来的「特殊字符串」列表，不是写死在函数里的。
        测试里常见: special_tokens=["<|endoftext|>"]
        以后也可以有多个，比如 ["<|endoftext|>", "<|pad|>"]

    ── 输出 ──
    list[str]
        每一段是一个「不含 special token 本身」的文本片段。
        每一段会单独做后面的 pre-tokenize 和 BPE 统计。
        相邻两段之间原来有一个 special token，但 special token 不会出现在结果里。

    ── 用什么切？不是空格！──
    切分符 = special_tokens 里的那些字符串（精确匹配整个子串）。
    例如 special_tokens=["<|endoftext|>"] 时，切分符就是字面量 "<|endoftext|>"。

    不是:
      - 不是按空格 split（空格留给后面的 GPT-2 pre-tokenizer 处理）
      - 不是按字节切
      - 不是按单个字符切

    是:
      - 在语料里找到 "<|endoftext|>" 这个完整子串，在它出现的位置下刀
      - 多个 special token 时，任意一个出现都算边界

  ── special token 到底是什么？──
    你的理解对: 本作业里主要就是 <|endoftext|>，表示「一篇文档结束」。
    它不是空格。空格是普通字符，会留在 segment 里，后面 pre-tokenize 时处理。

    special token 的两重身份:
      1. 在 init_vocab 里: 作为一个完整 token 放进词表（有固定 ID）
      2. 在训练统计时: 作为切分边界，且不参与 merge 计数（所以要先 split 掉）

  ── 举例 ──
    输入 corpus:
        "Doc A text<|endoftext|>Doc B text<|endoftext|>Doc C"

    输出 segments（示意）:
        ["Doc A text", "Doc B text", "Doc C"]

    注意:
      - "<|endoftext|>" 本身不在输出里
      - "Doc A text" 里的空格都还在
      - 三段彼此独立，后面统计 pair 时不能跨段 merge

    边界情况（实现时要处理）:
      - 开头就是 <|endoftext|>  → 可能产生空字符串 ""
      - 连续两个 <|endoftext|>  → 中间也会产生 ""
      - 没有 special token 的语料   → 返回 ["整份 corpus"] 一项
      - 空字符串 segment 通常应跳过，不参与统计

  ── 实现思路（单线程版，先想清再写）──
    1. 用 re.split，分隔 pattern 由 special_tokens 拼出来
    2. 每个 token 要先 re.escape()，因为 token 里可能有 | < > 等正则特殊字符
    3. 多个 token 用 "|" 连接成一个大 pattern（讲义原话）
    4. 对 split 得到的 list 过滤掉 "" 空段
    5. return 剩下的 list[str]
    """
    raise NotImplementedError


def read_corpus(input_path: str) -> str:
    """
    读训练语料，返回整个文件内容作为一个 Python 字符串 (str)。

    返回的是什么？
      - 类型: str（Unicode 文本），不是 bytes
      - 内容: 文件从头到尾的全部文字，可能几百万字符
      - 例子: "iron cement is a ready for use paste ...\\ntranslator Internet ..."

    路径不要硬编码：测试会传入不同的 input_path。
    """
    with open(input_path, encoding="utf-8") as f:
        text: str = f.read()
    return text


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
  """
  初始化词表的前 256 + len(special_tokens) 个格子。

  「500 个格子」怎么理解？
    vocab_size=500 是最终目标；一开始只有 256 个单字节 + special tokens。
    剩下 500 - 256 - len(special_tokens) 个格子，靠后面的 merge 一个一个填进去。

  ID 怎么分配？（要和 reference 对齐，否则 test 挂）
    1. special_tokens[0] -> ID 0, special_tokens[1] -> ID 1, ...
    2. 256 个单字节 token 接在后面，但顺序不是 0,1,2,...,255，
       而是 GPT-2 规定的 bs 顺序（见 _gpt2_byte_order 和 tests/common.py）。
       所以 ID 1 是 b'!'（字节 33），不是 b'\\x00'（字节 0）。

  返回什么类型？
    dict[int, bytes]：键是 token ID，值是该 token 对应的字节序列。
  """
  vocab: dict[int, bytes] = {}

  # 1) special tokens 占最低的 ID
  next_id = 0
  for token in special_tokens:
    vocab[next_id] = token.encode("utf-8")  # str -> bytes；special token 用 UTF-8 编码
    next_id += 1

  # 2) 256 个单字节 token，按 GPT-2 规定的顺序依次分配 ID
  for byte_val in _gpt2_byte_order():
    vocab[next_id] = bytes([byte_val])  # 单个字节的 bytes 长度为 1
    next_id += 1

  return vocab


def _gpt2_byte_order() -> list[int]:
  """
  返回「256 个单字节 token 应该按什么顺序分配 ID」。

  返回值
  ------
  list[int]，长度 256，每个元素是 0..255 之间的一个字节值，且互不重复。
  例如（配合 init_vocab 且只有一个 special token 时）：
      result[0]  == 33   →  vocab[1] == b'!'
      result[1]  == 34   →  vocab[2] == b'"'
      ...
      result[32] == 32   →  某个 ID 对应 b' '（空格；显示为 Ġ）
      ...
      result[255] == 某个剩余字节

  为什么顺序不是 0,1,2,...,255？
  ------------------------------
  测试要求与 GPT-2 / tiktoken 一致。OpenAI 在构造词表时，
  先把「打印出来好看」的字节排在前面，剩下的控制字符、空格等排在后面。
  tests/common.py 里的 gpt2_bytes_to_unicode() 用同一套 bs 顺序；
  本函数只复刻「顺序」，不管每个字节显示成什么 Unicode 字符。

  与 init_vocab 的关系
  --------------------
  init_vocab 会遍历本函数的返回值，依次执行：
      vocab[next_id] = bytes([byte_val])
  所以这里的顺序直接决定「哪个字节拿哪个 ID」。
  """
  # 第 1 段：ASCII 可打印字符  ! " # ... ~  （字节 33..126，共 94 个）
  # 这就是为什么 reference vocab 里 ID 1 是 b'!' 而不是 b'\\x00'
  bs: list[int] = list(range(ord("!"), ord("~") + 1))

  # 第 2 段：拉丁文补充区  ¡ ¢ £ ... ¬  （字节 161..172，共 12 个）
  bs += list(range(ord("¡"), ord("¬") + 1))

  # 第 3 段：拉丁文扩展  ® ¯ ° ... ÿ  （字节 174..255，共 82 个）
  # 三段合计 94 + 12 + 82 = 188 个「视觉上能直接显示」的字节
  bs += list(range(ord("®"), ord("ÿ") + 1))

  # 第 4 段：剩下 256 - 188 = 68 个字节（含空格 32、换行 10、\\x00 等）
  # 按数值 0,1,2,... 依次追加到末尾；空格 byte 32 会落在这里，而不是最前面
  for b in range(256):
    if b not in bs:
      bs.append(b)

  # 最终：长度恰好 256 的排列，是 [0..255] 的一个重排（permutation）
  assert len(bs) == 256
  assert len(set(bs)) == 256
  return bs
