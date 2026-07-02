"""T5EncoderModel: a faithful MLX port of the T5 v1.1 XXL text encoder used by FLUX.

Mirrors transformers' ``T5EncoderModel`` so the official ``t5-v1_1-xxl`` weights load
via the converter. T5 v1.1 differs from umT5 (see :mod:`~mlx_diffuser.models.umt5`)
in one place only: the relative-position bias is computed **once** in the first block
and shared across every layer (umT5 recomputes it per block). It is otherwise the same
encoder — bidirectional, T5 RMSNorm (scale-only, fp32 variance), gated-GELU
feed-forward, and unscaled attention (the position bias replaces ``1/sqrt(d)``).

Like umT5, this is designed to be loaded weight-quantized so the 4.7B-parameter
encoder fits in ~2.5 GB instead of ~9.5 GB.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..configuration import Config
from ..modeling import ModelMixin
from .umt5 import T5RMSNorm, _relative_position_bucket


@dataclasses.dataclass
class T5Config(Config):
    vocab_size: int = 32128
    d_model: int = 4096
    d_kv: int = 64
    d_ff: int = 10240
    num_layers: int = 24
    num_heads: int = 64
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_epsilon: float = 1e-6


class T5Attention(nn.Module):
    """T5 self-attention. Only the first layer owns ``relative_attention_bias``."""

    def __init__(self, cfg: T5Config, has_relative_attention_bias: bool):
        super().__init__()
        inner = cfg.num_heads * cfg.d_kv
        self.n_heads = cfg.num_heads
        self.d_kv = cfg.d_kv
        self.q = nn.Linear(cfg.d_model, inner, bias=False)
        self.k = nn.Linear(cfg.d_model, inner, bias=False)
        self.v = nn.Linear(cfg.d_model, inner, bias=False)
        self.o = nn.Linear(inner, cfg.d_model, bias=False)
        self.has_relative_attention_bias = has_relative_attention_bias
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                cfg.relative_attention_num_buckets, cfg.num_heads
            )
        self._num_buckets = cfg.relative_attention_num_buckets
        self._max_distance = cfg.relative_attention_max_distance

    def position_bias(self, q_len: int, k_len: int) -> mx.array:
        ctx = np.arange(q_len)[:, None]
        mem = np.arange(k_len)[None, :]
        buckets = _relative_position_bucket(mem - ctx, self._num_buckets, self._max_distance)
        vals = self.relative_attention_bias(mx.array(buckets))  # (q, k, heads)
        return vals.transpose(2, 0, 1)[None]  # (1, heads, q, k)

    def __call__(self, x: mx.array, bias: mx.array) -> mx.array:
        b, lq, _ = x.shape
        q = self.q(x).reshape(b, lq, self.n_heads, self.d_kv).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(b, lq, self.n_heads, self.d_kv).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(b, lq, self.n_heads, self.d_kv).transpose(0, 2, 1, 3)
        # T5 uses no 1/sqrt(d) scaling; the relative-position bias is the additive mask.
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=bias.astype(q.dtype))
        out = out.transpose(0, 2, 1, 3).reshape(b, lq, self.n_heads * self.d_kv)
        return self.o(out)


class T5LayerSelfAttention(nn.Module):
    def __init__(self, cfg: T5Config, has_relative_attention_bias: bool):
        super().__init__()
        self.SelfAttention = T5Attention(cfg, has_relative_attention_bias)
        self.layer_norm = T5RMSNorm(cfg.d_model, cfg.layer_norm_epsilon)

    def __call__(self, x: mx.array, bias: mx.array) -> mx.array:
        return x + self.SelfAttention(self.layer_norm(x), bias)


class T5DenseGatedActDense(nn.Module):
    def __init__(self, cfg: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wi_1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wo = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.wo(nn.gelu_approx(self.wi_0(x)) * self.wi_1(x))


class T5LayerFF(nn.Module):
    def __init__(self, cfg: T5Config):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(cfg)
        self.layer_norm = T5RMSNorm(cfg.d_model, cfg.layer_norm_epsilon)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.DenseReluDense(self.layer_norm(x))


class T5Block(nn.Module):
    def __init__(self, cfg: T5Config, has_relative_attention_bias: bool):
        super().__init__()
        self.layer = [T5LayerSelfAttention(cfg, has_relative_attention_bias), T5LayerFF(cfg)]

    def __call__(self, x: mx.array, bias: mx.array) -> mx.array:
        x = self.layer[0](x, bias)
        return self.layer[1](x)


class _Stack(nn.Module):
    def __init__(self, cfg: T5Config):
        super().__init__()
        # Only block 0 owns the relative-position bias; it is shared across all blocks.
        self.block = [
            T5Block(cfg, has_relative_attention_bias=(i == 0)) for i in range(cfg.num_layers)
        ]
        self.final_layer_norm = T5RMSNorm(cfg.d_model, cfg.layer_norm_epsilon)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        bias = self.block[0].layer[0].SelfAttention.position_bias(x.shape[1], x.shape[1])
        if mask is not None:
            bias = bias + mask
        for block in self.block:
            x = block(x, bias)
        return self.final_layer_norm(x)


class T5EncoderModel(ModelMixin[T5Config]):
    """T5 v1.1 text encoder. Produces per-token embeddings for FLUX joint attention."""

    config_class = T5Config

    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = _Stack(config)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        """``input_ids`` ``(B, L)`` -> embeddings ``(B, L, d_model)``.

        Activations run in fp32 (T5 routinely exceeds the fp16 range); only the
        (optionally quantized) weights stay small.
        """
        x = self.shared(input_ids).astype(mx.float32)
        mask = None
        if attention_mask is not None:
            neg = (1.0 - attention_mask.astype(mx.float32)) * -1e9
            mask = neg[:, None, None, :]  # (B, 1, 1, L) additive
        return self.encoder(x, mask)
