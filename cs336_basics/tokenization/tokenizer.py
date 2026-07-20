"""Byte-level BPE tokenizer: str ↔ list[int]。

══════════════════════════════════════════════════════════════════════════════
整个文件在干什么（一句话）
══════════════════════════════════════════════════════════════════════════════
  encode:  "Hi there"  →  [123, 456, ...]     # 给 LM 吃的整数序列
  decode:  [123, 456]  →  "Hi there"         # 给人看的字符串

  中间靠两样东西：
    vocab  {0: b'!', 257: b' the', ...}       # id ↔ 这个 token 的原始 bytes
    merges [(b'h',b'i'), (b' t',b'he'), ...]   # BPE 训练时学到的「相邻 bytes 怎么并」

══════════════════════════════════════════════════════════════════════════════
调用关系（谁调谁）
══════════════════════════════════════════════════════════════════════════════

  构造阶段（只做一次）：
    Tokenizer.from_files("vocab.json", "merges.txt")
      → _load_vocab_from_json   # json → dict[int, bytes]
      → _load_merges_from_text  # merges.txt → list[tuple[bytes, bytes]]
      → Tokenizer.__init__      # 建 _bytes_to_id、_merge_rank 查表

  encode 方向（text → ids）：
    encode("Hi<|endoftext|>there")
      → split_text_on_special_tokens     # 按 special 切开
      → 普通片段: pre_tokenize (utils)   # 正则切成 pretoken
      → 每个 pretoken: _encode_pretoken  # 单字节 → BPE merge → 查 id
      → special 片段: 直接 _bytes_to_id

  decode 方向（ids → text）：
    decode([123, 456])
      → vocab[123], vocab[456] 拼成 b'...'
      → .decode("utf-8")

  大文件（边读边吐 id，不一次 load 全文）：
    encode_iterable(open("corpus.txt"))  → 内部反复调 encode(一行)

══════════════════════════════════════════════════════════════════════════════
数据流例子：encode("Hi there")（数字是示意，逻辑不变）
══════════════════════════════════════════════════════════════════════════════

  "Hi there"
       │ split_text_on_special_tokens（无 special 时整段是一个 piece）
       ▼
  piece = "Hi there"
       │ pre_tokenize
       ▼
  pretokens = ["Hi", " there"]     # 空格粘在下一个词前面，GPT-2 惯例
       │
       ├─ "Hi"  → _encode_pretoken
       │            [b'H',b'i'] → merge → [b'Hi'] → id 17250
       │
       └─ " there" → _encode_pretoken
                     [b' ',b't',b'h',b'e',b'r',b'e'] → merge → [b' there'] → id 994
       ▼
  ids = [17250, 994]

  decode([17250, 994]):
    vocab[17250]=b'Hi'  +  vocab[994]=b' there'  →  b'Hi there'  →  "Hi there"
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

from cs336_basics.tokenization.utils import (
    gpt2_display_to_bytes,
    merge_bytes_at_index,
    pre_tokenize,
    split_text_on_special_tokens,
)


class Tokenizer:
    """Byte-level BPE tokenizer: encode text with a fixed vocab + merge list."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """构造查表，之后 encode/decode 都走这些表。

        输入例:
          vocab  = {0: b'<|endoftext|>', 1: b'!', 257: b' the'}
          merges = [(b'h', b'i'), (b' t', b'he')]
          special_tokens = ['<|endoftext|>']

        输出（挂在 self 上）:
          self.vocab         同上，若 special 不在 vocab 里会 append 新 id
          self._bytes_to_id  {b'!': 1, b' the': 257, b'<|endoftext|>': 0}
          self._merge_rank   {(b'h', b'i'): 0, (b' t', b'he'): 1}  # 越小越先 merge
        """
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = list(special_tokens or [])

        self._bytes_to_id: dict[bytes, int] = {
            token_bytes: token_id for token_id, token_bytes in self.vocab.items()
        }

        for special in self.special_tokens:
            special_bytes = special.encode("utf-8")
            if special_bytes not in self._bytes_to_id:
                next_id = max(self.vocab.keys(), default=-1) + 1
                self.vocab[next_id] = special_bytes
                self._bytes_to_id[special_bytes] = next_id

        self._special_token_set = set(self.special_tokens)

        self._merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

    @classmethod
    def from_files(
        tokenizer_class,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """从磁盘文件构造 Tokenizer。

        输入例:
          vocab_filepath  = "artifacts/vocab.json"
          merges_filepath = "artifacts/merges.txt"
          special_tokens  = ["<|endoftext|>"]

        输出例:
          Tokenizer 实例（内部 vocab/merges 已填好，可直接 .encode/.decode）

        数据流:
          vocab.json  {"257": "Ġthe", "!": 1, ...}
            → _load_vocab_from_json → {257: b' the', 1: b'!', ...}
          merges.txt  两行 "h e" / "Ġt he"
            → _load_merges_from_text → [(b'h', b'e'), (b' t', b'he')]
          → Tokenizer.__init__(vocab, merges, special_tokens)
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = _load_vocab_from_json(raw_vocab)

        merges = _load_merges_from_text(merges_filepath)

        return tokenizer_class(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """整段文本 → token id 列表。

        输入例:
          text = "Hi<|endoftext|>there"
          special_tokens = ["<|endoftext|>"]

        输出例:
          [17250, 50256, 258]    # Hi | <|endoftext|> | there（数字仅示意）

        数据流:
          "Hi<|endoftext|>there"
            → split → ["Hi", "<|endoftext|>", "there"]
            → "Hi"     → pre_tokenize → ["Hi"]     → _encode_pretoken → [17250]
            → "<|eot|>" → 整段查 _bytes_to_id      → [50256]
            → "there"  → pre_tokenize → ["there"]  → _encode_pretoken → [258]
            → 拼起来 [17250, 50256, 258]
        """
        ids: list[int] = []
        for piece in split_text_on_special_tokens(text, self.special_tokens):
            if not piece:
                continue
            if piece in self._special_token_set:
                ids.append(self._bytes_to_id[piece.encode("utf-8")])
                continue

            for pretoken in pre_tokenize([piece], workers=1):
                ids.extend(self._encode_pretoken(pretoken))

        return ids

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """单个 pretoken（不能再跨词切开）→ 若干 token id。

        输入例:
          pretoken = "there"    # 无空格版，3 个字符，方便手算
          self._merge_rank = {
            (b't', b'h'): 0,
            (b'h', b'e'): 1,
            (b'th', b'e'): 2,
          }

        输出例:
          [258, 302]   # b'the' 和 b're' 各一个 id（未必合成一个 token）

        为什么单独一个函数：merge 不能跨 pretoken 边界；
        "the cat" 必须先切成 "the" 和 " cat" 再分别进这里。
        """
        # pretoken="there" → pretoken.encode("utf-8") = b'there'（5 个 byte）
        # 每个 byte 单独包成一个 list 元素：
        byte_seq = [bytes([byte_val]) for byte_val in pretoken.encode("utf-8")]
        # byte_seq = [b't', b'h', b'e', b'r', b'e']
        #            ^index0 ^1    ^2    ^3    ^4
        # 类型：list[bytes]，长度 = pretoken 的 UTF-8 字节数

        while True:
            best_rank = None      # 本轮找到的「最小 merge 优先级」；None = 还没找到
            best_index = None     # 这个最小 rank 的 pair 在 byte_seq 里的左端下标

            # len(byte_seq)=5 → index 走 0,1,2,3（共 4 个相邻 pair）
            for index in range(len(byte_seq) - 1):
                pair = (byte_seq[index], byte_seq[index + 1])
                # index=0: pair=(b't', b'h')
                # index=1: pair=(b'h', b'e')
                # index=2: pair=(b'e', b'r')
                # index=3: pair=(b'r', b'e')

                rank = self._merge_rank.get(pair)
                # (t,h)→0, (h,e)→1；(e,r)和(r,e)不在表里 → rank=None

                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
                # 扫完后：best_rank=0, best_index=0（(t,h) 比 (h,e) 优先）

            if best_index is None:
                # 所有相邻 pair 都不在 merge 表里 → 再也合不动了
                break

            left = byte_seq[best_index]           # b't'
            right = byte_seq[best_index + 1]      # b'h'
            merge_bytes_at_index(byte_seq, best_index, left, right)
            # ── 第 1 轮 merge 后 ──
            # byte_seq = [b'th', b'e', b'r', b'e']   长度 5→4
            #
            # ── 第 2 轮 while：best_index=0，pair (th,e) rank=2 ──
            # byte_seq = [b'the', b'r', b'e']          长度 4→3
            #
            # ── 第 3 轮 while：(the,r)、(r,e) 都不在 merge 表 → best_index=None → break ──

        # byte_seq 停住时 = [b'the', b're']   （2 个 token，不是 1 个——合不动了就停）
        return [self._bytes_to_id[token_bytes] for token_bytes in byte_seq]
        # → [_bytes_to_id[b'the'], _bytes_to_id[b're']]  例：[258, 302]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """大文件边读边 encode，每次只处理 iterable 吐出的一小块文本。

        输入例:
          iterable = open("tinystories.txt")   # 每次 for 循环给一行 "Once upon...\n"
          或 iterable = ["Hello", " world"]

        输出例（逐个 yield，不是一次性 list）:
          123, 456, 789, ...   # 等价于把全文 encode 后按顺序吐 id

        和 encode 的区别:
          encode("整本书")        → list[int]，全文同时在内存
          encode_iterable(file)   → 读一行 encode 一行，内存只占一行

        为什么按行切仍正确：GPT-2 pre-tokenize 本来就在空白/换行处切开，
        行尾的 '\\n' 会留在 pretoken 里，和一次 encode 整文件一致。
        """
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def decode(self, ids: list[int]) -> str:
        """token id 列表 → 字符串（encode 的逆，但不保证逐 token 可逆）。

        输入例:
          ids = [17250, 994]    # 对应 b'Hi' + b' there'

        输出例:
          "Hi there"

        数据流:
          [17250, 994]
            → vocab[17250]=b'Hi', vocab[994]=b' there'
            → b"".join(...) = b'Hi there'
            → .decode("utf-8") = "Hi there"

        注意：不在 token 之间加空格；空格已经编码在某个 token 的 bytes 里（如 b' there'）。
        """
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")


def _load_vocab_from_json(raw_vocab: dict) -> dict[int, bytes]:
    """json 里的 vocab → 内存里统一的 dict[int, bytes]。

    输入例 A（训练脚本格式，key 是 id 字符串）:
      {"0": "<|endoftext|>", "257": "Ġthe"}

    输入例 B（GPT-2 fixture，key 是 display 字符串）:
      {"<|endoftext|>": 0, "Ġthe": 257}

    输出例（两种输入都得到同一种结构）:
      {0: b'<|endoftext|>', 257: b' the'}
      # "Ġ" 是 GPT-2 文件里表示前导空格的 display 字符 → 真实 bytes 是 b' the'
    """
    sample_key = next(iter(raw_vocab))
    if sample_key.isdigit():
        return {int(token_id): gpt2_display_to_bytes(display) for token_id, display in raw_vocab.items()}
    return {int(token_id): gpt2_display_to_bytes(display) for display, token_id in raw_vocab.items()}


def _load_merges_from_text(merges_filepath: str) -> list[tuple[bytes, bytes]]:
    """merges.txt → BPE 合并规则列表（顺序 = 训练时学到的优先级）。

    输入例（文件内容两行）:
      Ġ t
      h e

    输出例:
      [(b' ', b't'), (b'h', b'e')]
      # rank 0 = (b' ', b't') 最先被考虑合并
      # rank 1 = (b'h', b'e')

    为什么用 bytes 不用 display 字符串：encode 时 byte_seq 全是 bytes，
    查 _merge_rank 的 key 也是 (bytes, bytes)。
    """
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_filepath, encoding="utf-8") as f:
        for line in f:
            cleaned = line.rstrip()
            if not cleaned:
                continue
            parts = cleaned.split(" ")
            if len(parts) != 2:
                continue
            left_display, right_display = parts
            merges.append((gpt2_display_to_bytes(left_display), gpt2_display_to_bytes(right_display)))
    return merges
