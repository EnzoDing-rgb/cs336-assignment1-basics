"""Transformer model (5 modules).

linear / embedding / normalization / attention  →  原子算子
transformer  →  组装层：SwiGLU、transformer block、整网 TransformerLM

FFN 和 block 不是同一个东西：block 里包含 attention + FFN + norm + 残差。
但 FFN 只服务于 block，所以和 block 一起放在 transformer.py，不单独开文件。
"""
