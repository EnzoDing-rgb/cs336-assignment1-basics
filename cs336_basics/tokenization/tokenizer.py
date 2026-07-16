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
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = list(special_tokens or [])

        # bytes -> id，encode 最后查表用
        self._bytes_to_id: dict[bytes, int] = {
            token_bytes: token_id for token_id, token_bytes in self.vocab.items()
        }

        # 作业要求：special token 不在 vocab 里就 append
        for special in self.special_tokens:
            special_bytes = special.encode("utf-8")
            if special_bytes not in self._bytes_to_id:
                next_id = max(self.vocab.keys(), default=-1) + 1
                self.vocab[next_id] = special_bytes
                self._bytes_to_id[special_bytes] = next_id

        self._special_token_set = set(self.special_tokens)

        # merge_rank: pair -> 它在 merges 列表里的下标（越小 = 越早学到 = 越优先合并）
        # 例: merges = [(b't', b'h'), (b'th', b'e'), (b'a', b'b')]
        #   -> merge_rank = {(b't', b'h'): 0, (b'th', b'e'): 1, (b'a', b'b'): 2}
        # 后面 encode 时用这个表，避免每次把 5 万条 merges 全扫一遍
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
        # @classmethod：第一个参数是类本身（这里就是 Tokenizer），不是实例
        # return tokenizer_class(...) 等价于 return Tokenizer(...)
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = _load_vocab_from_json(raw_vocab)

        merges = _load_merges_from_text(merges_filepath)

        return tokenizer_class(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # 例: text='Hello<|endoftext|>world' + special_tokens=['<|endoftext|>']
        #   -> pieces=['Hello', '<|endoftext|>', 'world']
        #   -> Hello/world 走 pre-tokenize + BPE；special 整段查 id
        ids: list[int] = []
        for piece in split_text_on_special_tokens(text, self.special_tokens):
            if not piece:
                continue
            if piece in self._special_token_set:
                ids.append(self._bytes_to_id[piece.encode("utf-8")])
                continue

            # 先 pre-tokenize，再对每个 pretoken 单独做 BPE merge。
            # 这不是“巧妙优化”，而是 GPT-2 BPE 的正确语义：merge 只发生在同一个
            # pretoken 内部，绝不能跨词边界。
            # 例: piece='the cat ate'
            #   -> pretokens=['the', ' cat', ' ate']
            #   -> 分别 encode；不会出现把 'e' 和 ' ' 合成一个 token 这种跨词 merge
            for pretoken in pre_tokenize([piece], workers=1):
                ids.extend(self._encode_pretoken(pretoken))

        return ids

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        # Step 1: pre-token 拆成单字节 list[bytes]
        #   例: 'the' -> [b't', b'h', b'e']
        byte_seq = [bytes([byte_val]) for byte_val in pretoken.encode("utf-8")]

        # Step 2: 每轮扫「当前所有相邻 pair」，只合并 rank 最小的那一对。
        # rank = merges 下标，越小越优先。和 pair 在序列里靠左/靠右无关：
        #   例 [a,b,c,d]，若 (c,d) rank=0、(a,b) rank=9 → 先合 (c,d)，不会因为 (a,b) 靠左就先合它。
        while True:
            best_rank = None
            best_index = None
            for index in range(len(byte_seq) - 1):
                pair = (byte_seq[index], byte_seq[index + 1])
                rank = self._merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index

            # 所有相邻 pair 都不在 merge_rank 里 → 没法再合了
            if best_index is None:
                break

            left = byte_seq[best_index]
            right = byte_seq[best_index + 1]
            merge_bytes_at_index(byte_seq, best_index, left, right)

        # Step 3: 每个 bytes token 查 vocab 得 id
        #   例: [b'the'] -> [9]
        return [self._bytes_to_id[token_bytes] for token_bytes in byte_seq]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # --- iterable vs generator，用这个函数记一下 ---
        #
        # iterable：任何能被 for 循环的东西。
        #   例: open("big.txt") 每次给出一行（通常带 '\n'）
        #       ["Hello", " world"]
        #
        # 本函数带 yield，所以调用时返回的是 generator（一种 iterator）：
        #   ids = tokenizer.encode_iterable(f)   # 这里几乎还没开始 encode
        #   for token_id in ids:                 # 每要一个 id，才继续往下跑一点
        #       ...
        #
        # 对比 encode()：
        #   encode(text) -> list[int]   # 一次算完，整份结果都在内存里
        #   encode_iterable(...)        # 边读边 yield，适合大文件（测记忆体时只有 ~1MB 限额）
        #
        # 为什么可以对“每一行”分别 encode？
        #   文件按行迭代时，边界落在 '\n' 上；GPT-2 pre-tokenize 本来就会在空白处切开，
        #   所以 line-by-line 的结果应和整文件一次 encode 一致（至少对本作业语料成立）
        for text_chunk in iterable:
            # yield from：把 encode 得到的那一小段 list，逐个 id 往外递
            # 等价于: for token_id in self.encode(text_chunk): yield token_id
            yield from self.encode(text_chunk)

    def decode(self, ids: list[int]) -> str:
        # id -> vocab 里的 bytes，拼成一条 bytes，再 decode 成 str
        # errors="replace"：非法 utf-8 字节用 � 代替，避免直接抛错
        # 例: ids=[72, 101, 108, 108, 111] -> b'Hello' -> 'Hello'
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")


def _load_vocab_from_json(raw_vocab: dict) -> dict[int, bytes]:
    """支持两种落盘格式：训练脚本 {id: display}，GPT-2 fixture {display: id}。"""
    # 例 A（训练脚本 artifacts/vocab.json）:
    #   {"0": "<|endoftext|>", "1": "!", "257": "Ġt", ...}
    #   -> {0: b'<|endoftext|>', 1: b'!', 257: b' t', ...}
    #
    # 例 B（GPT-2 fixture gpt2_vocab.json / train-bpe-reference-vocab.json）:
    #   {"<|endoftext|>": 0, "!": 1, "Ġt": 257, ...}
    #   -> {0: b'<|endoftext|>', 1: b'!', 257: b' t', ...}
    # 两种输入方向不同，最终都统一成 id -> bytes
    sample_key = next(iter(raw_vocab))
    if sample_key.isdigit():
        return {int(token_id): gpt2_display_to_bytes(display) for token_id, display in raw_vocab.items()}
    return {int(token_id): gpt2_display_to_bytes(display) for display, token_id in raw_vocab.items()}


def _load_merges_from_text(merges_filepath: str) -> list[tuple[bytes, bytes]]:
    # 例 merges.txt 每行一个 pair（GPT-2 display，空格分隔）:
    #   Ġ t
    #   h e
    #   Ġt he
    # -> [(b' ', b't'), (b'h', b'e'), (b' t', b'he')]
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
