"""UMT5EncoderModel: a faithful MLX port of the umT5-xxl text encoder used by WAN.

Mirrors transformers' ``UMT5EncoderModel`` so the official weights load via the
converter. umT5 differs from vanilla T5 in that *every* block computes its own
relative-position bias (T5 shares layer 0's). The encoder is bidirectional, uses
T5 RMSNorm (scale-only, fp32 variance), gated-GELU feed-forward, and unscaled
attention (the relative-position bias replaces the usual ``1/sqrt(d)`` scaling).

Designed to be loaded weight-quantized (``from_pretrained(..., quantize=4)``) so
the 5.6B-parameter encoder fits in ~3 GB instead of ~11 GB.
"""

from __future__ import annotations

import dataclasses
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..configuration import Config
from ..modeling import ModelMixin


@dataclasses.dataclass
class UMT5Config(Config):
    vocab_size: int = 256384
    d_model: int = 4096
    d_kv: int = 64
    d_ff: int = 10240
    num_layers: int = 24
    num_heads: int = 64
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_epsilon: float = 1e-6


class T5RMSNorm(nn.Module):
    """T5 RMSNorm: scale-only, no mean subtraction, fp32 variance."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        var = mx.mean(xf * xf, axis=-1, keepdims=True)
        x = (xf * mx.rsqrt(var + self.eps)).astype(x.dtype)
        return self.weight * x


def _relative_position_bucket(
    rel_pos: np.ndarray, num_buckets: int, max_distance: int
) -> np.ndarray:
    """Bidirectional T5 relative-position bucketing (encoder)."""
    num_buckets //= 2
    ret = (rel_pos > 0).astype(np.int64) * num_buckets
    n = np.abs(rel_pos)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    # Clamp before the log: small positions take the exact branch via np.where, so
    # their (discarded) log value must not be -inf/NaN.
    safe = np.maximum(n, max_exact)
    large = max_exact + (
        np.log(safe.astype(np.float64) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(np.int64)
    large = np.minimum(large, num_buckets - 1)
    ret += np.where(is_small, n, large)
    return ret


class UMT5Attention(nn.Module):
    def __init__(self, cfg: UMT5Config, has_relative_attention_bias: bool):
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


class UMT5LayerSelfAttention(nn.Module):
    def __init__(self, cfg: UMT5Config):
        super().__init__()
        self.SelfAttention = UMT5Attention(cfg, has_relative_attention_bias=True)
        self.layer_norm = T5RMSNorm(cfg.d_model, cfg.layer_norm_epsilon)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        bias = self.SelfAttention.position_bias(x.shape[1], x.shape[1])
        if mask is not None:
            bias = bias + mask
        return x + self.SelfAttention(self.layer_norm(x), bias)


class UMT5DenseGatedActDense(nn.Module):
    def __init__(self, cfg: UMT5Config):
        super().__init__()
        self.wi_0 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wi_1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.wo = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.wo(nn.gelu_approx(self.wi_0(x)) * self.wi_1(x))


class UMT5LayerFF(nn.Module):
    def __init__(self, cfg: UMT5Config):
        super().__init__()
        self.DenseReluDense = UMT5DenseGatedActDense(cfg)
        self.layer_norm = T5RMSNorm(cfg.d_model, cfg.layer_norm_epsilon)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.DenseReluDense(self.layer_norm(x))


class UMT5Block(nn.Module):
    def __init__(self, cfg: UMT5Config):
        super().__init__()
        self.layer = [UMT5LayerSelfAttention(cfg), UMT5LayerFF(cfg)]

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        x = self.layer[0](x, mask)
        return self.layer[1](x)


class _Stack(nn.Module):
    def __init__(self, cfg: UMT5Config):
        super().__init__()
        self.block = [UMT5Block(cfg) for _ in range(cfg.num_layers)]
        self.final_layer_norm = T5RMSNorm(cfg.d_model, cfg.layer_norm_epsilon)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        for block in self.block:
            x = block(x, mask)
        return self.final_layer_norm(x)


class UMT5EncoderModel(ModelMixin[UMT5Config]):
    """umT5 text encoder. Produces per-token embeddings for WAN cross-attention."""

    config_class = UMT5Config

    def __init__(self, config: UMT5Config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = _Stack(config)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        """``input_ids`` ``(B, L)`` -> embeddings ``(B, L, d_model)``.

        ``attention_mask`` ``(B, L)`` with 1 for real tokens, 0 for padding.
        """
        # Run activations in fp32: T5/umT5 activations routinely exceed the fp16
        # range, so a low-precision compute path overflows to inf/NaN on some
        # prompts. The (quantized) weights stay small; only the activations are fp32.
        x = self.shared(input_ids).astype(mx.float32)
        mask = None
        if attention_mask is not None:
            neg = (1.0 - attention_mask.astype(mx.float32)) * -1e9
            mask = neg[:, None, None, :]  # (B, 1, 1, L) additive
        return self.encoder(x, mask)
