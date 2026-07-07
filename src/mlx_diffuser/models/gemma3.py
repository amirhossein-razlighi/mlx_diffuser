"""Gemma3TextEncoder: the Gemma-3 language model as a text encoder, in MLX.

LTX-2 conditions its audio-video transformer on **all** hidden states of a
Gemma-3-12B decoder (the embedding output plus every layer, 49 states for the
12B model), which downstream connectors normalize and project per modality.
This port mirrors transformers' ``Gemma3TextModel`` module-for-module so the
official weights load via the converter, and returns the stacked per-layer
hidden states rather than logits.

Gemma-3 specifics reproduced here: sqrt(hidden)-scaled embeddings (the scale is
rounded in the weight dtype, matching the reference), zero-centered RMSNorms
computed in float32, per-head q/k RMSNorm, GQA, rotate-half RoPE with a
dual-frequency scheme (theta 10k for sliding-window layers, theta 1M with
linear position scaling for full-attention layers, every 6th layer), and
sandwich (pre+post) norms around both attention and the gated-GELU MLP.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn

from ..configuration import Config
from ..modeling import ModelMixin


@dataclasses.dataclass
class Gemma3Config(Config):
    vocab_size: int = 262208
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    rope_linear_factor: float = 8.0
    query_pre_attn_scalar: float = 256.0
    sliding_window: int = 1024
    sliding_window_pattern: int = 6

    @classmethod
    def gemma3_12b(cls) -> Gemma3Config:
        return cls()  # defaults match the Gemma-3-12B text tower

    def is_full_attention(self, layer_idx: int) -> bool:
        """Every ``sliding_window_pattern``-th layer uses full (global) attention."""
        return (layer_idx + 1) % self.sliding_window_pattern == 0


class Gemma3RMSNorm(nn.Module):
    """Zero-centered RMSNorm: ``norm(x) * (1 + weight)``, computed in float32."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        normed = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (normed * (1.0 + self.weight.astype(mx.float32))).astype(x.dtype)


def _rope_cos_sin(positions: mx.array, head_dim: int, theta: float, factor: float = 1.0):
    """Rotate-half RoPE tables ``(L, head_dim)``; ``factor`` linearly scales positions."""
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    freqs = (positions.astype(mx.float32) / factor)[:, None] * inv_freq[None]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb), mx.sin(emb)


def _apply_rotate_half(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate-half RoPE on ``(B, heads, L, head_dim)`` with ``(L, head_dim)`` tables."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


class Gemma3Attention(nn.Module):
    def __init__(self, cfg: Gemma3Config):
        super().__init__()
        d, hd = cfg.hidden_size, cfg.head_dim
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = hd
        self.scale = cfg.query_pre_attn_scalar**-0.5
        self.q_proj = nn.Linear(d, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * hd, d, bias=False)
        self.q_norm = Gemma3RMSNorm(hd, cfg.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(hd, cfg.rms_norm_eps)

    def __call__(self, x: mx.array, rope, mask: mx.array) -> mx.array:
        b, n, _ = x.shape
        q = self.q_proj(x).reshape(b, n, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, n, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, n, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = _apply_rotate_half(q, *rope).astype(v.dtype)
        k = _apply_rotate_half(k, *rope).astype(v.dtype)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask.astype(v.dtype)
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, n, -1)
        return self.o_proj(out)


class Gemma3MLP(nn.Module):
    def __init__(self, cfg: Gemma3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, cfg: Gemma3Config):
        super().__init__()
        d, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.self_attn = Gemma3Attention(cfg)
        self.mlp = Gemma3MLP(cfg)
        self.input_layernorm = Gemma3RMSNorm(d, eps)
        self.post_attention_layernorm = Gemma3RMSNorm(d, eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(d, eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(d, eps)

    def __call__(self, x: mx.array, rope, mask: mx.array) -> mx.array:
        h = self.self_attn(self.input_layernorm(x), rope, mask)
        x = x + self.post_attention_layernorm(h)
        h = self.mlp(self.pre_feedforward_layernorm(x))
        return x + self.post_feedforward_layernorm(h)


class Gemma3TextEncoder(ModelMixin[Gemma3Config]):
    """Gemma-3 decoder returning every hidden state, stacked ``(B, L, D, layers+1)``."""

    config_class = Gemma3Config

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = Gemma3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def _masks(self, attention_mask: mx.array | None, seq_len: int) -> tuple[mx.array, mx.array]:
        """Additive ``(B or 1, 1, L, L)`` float32 masks: (full-causal, sliding-window)."""
        pos = mx.arange(seq_len)
        causal = pos[None, :] <= pos[:, None]
        window = pos[None, :] > (pos[:, None] - self.config.sliding_window)
        full = causal
        sliding = causal & window
        if attention_mask is not None:
            valid = attention_mask.astype(mx.bool_)[:, None, None, :]  # key-side padding
            full = full[None, None] & valid
            sliding = sliding[None, None] & valid
        else:
            full = full[None, None]
            sliding = sliding[None, None]
        # Additive finite bias (exact in the softmax, avoids NaN on fully-padded rows).
        to_bias = lambda m: (1.0 - m.astype(mx.float32)) * -1e4  # noqa: E731
        return to_bias(full), to_bias(sliding)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        """Encode ``(B, L)`` token ids into all hidden states ``(B, L, D, num_layers+1)``.

        Matches transformers' ``output_hidden_states=True`` ordering: the scaled
        embedding output, then each decoder layer's output, with the final
        RMSNorm applied to the last entry (it doubles as ``last_hidden_state``).
        """
        cfg = self.config
        seq_len = input_ids.shape[1]
        x = self.embed_tokens(input_ids)
        # The reference casts sqrt(hidden) to the weight dtype (55.4 -> 55.5 in bf16).
        x = x * mx.array(cfg.hidden_size**0.5).astype(x.dtype)

        positions = mx.arange(seq_len)
        rope_local = _rope_cos_sin(positions, cfg.head_dim, cfg.rope_local_base_freq)
        rope_global = _rope_cos_sin(
            positions, cfg.head_dim, cfg.rope_theta, factor=cfg.rope_linear_factor
        )
        full_mask, sliding_mask = self._masks(attention_mask, seq_len)

        states = [x]
        for i, layer in enumerate(self.layers):
            full = cfg.is_full_attention(i)
            x = layer(x, rope_global if full else rope_local, full_mask if full else sliding_mask)
            states.append(x)
        states[-1] = self.norm(states[-1])
        return mx.stack(states, axis=-1)
