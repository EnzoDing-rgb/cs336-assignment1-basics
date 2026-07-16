"""Transformer model (5 modules).

linear / embedding / normalization / attention  →  原子算子
language_model  →  组装层：SwiGLU FFN、transformer block、整网 forward

FFN 和 block 不是同一个东西：block 里包含 attention + FFN + norm + 残差。
但 FFN 只服务于 block，所以和 block 一起放在 language_model.py，不单独开文件。
"""
