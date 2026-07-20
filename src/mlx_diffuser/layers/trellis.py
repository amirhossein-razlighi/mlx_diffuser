"""Transformer layers used by the native TRELLIS port.

The module names intentionally mirror Microsoft's reference implementation.  Keeping
the same parameter tree lets the official safetensors checkpoints load without
weight fusion, splitting, or a PyTorch dependency.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class LayerNorm32(nn.Module):
    """LayerNorm computed in FP32 and cast back to the input dtype."""

    def __init__(self, dim: int, *, affine: bool, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if affine:
            self.weight = mx.ones((dim,), dtype=mx.float32)
            self.bias = mx.zeros((dim,), dtype=mx.float32)
        else:
            # Private arrays are not included in an MLX module's parameter tree.
            self._weight = None
            self._bias = None

    @property
    def affine(self) -> bool:
        return hasattr(self, "weight")

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        h = x.astype(mx.float32)
        mean = mx.mean(h, axis=-1, keepdims=True)
        var = mx.mean(mx.square(h - mean), axis=-1, keepdims=True)
        h = (h - mean) * mx.rsqrt(var + self.eps)
        if self.affine:
            h = h * self.weight + self.bias
        return h.astype(dtype)


class MultiHeadRMSNorm(nn.Module):
    """TRELLIS q/k normalization with one learned scale per attention head."""

    def __init__(self, head_dim: int, heads: int):
        super().__init__()
        self.scale = math.sqrt(head_dim)
        self.gamma = mx.ones((heads, head_dim), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        h = x.astype(mx.float32)
        norm = mx.sqrt(mx.sum(mx.square(h), axis=-1, keepdims=True))
        h = h / mx.maximum(norm, mx.array(1e-12, dtype=mx.float32))
        return (h * self.gamma * self.scale).astype(dtype)


def absolute_position_embedding(coords: mx.array, channels: int) -> mx.array:
    """Reference-compatible multi-axis sinusoidal absolute position embedding."""

    if coords.ndim != 2:
        raise ValueError(f"coords must have shape (N, axes), got {coords.shape}")
    axes = coords.shape[1]
    freq_dim = channels // axes // 2
    if freq_dim == 0:
        raise ValueError(f"channels ({channels}) is too small for {axes} coordinate axes")
    freqs = 1.0 / (10000 ** (mx.arange(freq_dim, dtype=mx.float32) / freq_dim))
    phases = coords.astype(mx.float32).reshape(-1, 1) * freqs.reshape(1, -1)
    embed = mx.concatenate([mx.sin(phases), mx.cos(phases)], axis=-1)
    embed = embed.reshape(coords.shape[0], -1)
    if embed.shape[1] < channels:
        embed = mx.pad(embed, [(0, 0), (0, channels - embed.shape[1])])
    return embed


class TrellisTimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by TRELLIS's two-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        # A list, rather than nn.Sequential, preserves keys ``mlp.0`` / ``mlp.2``.
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        ]

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / max(half, 1))
        phases = t.astype(mx.float32).reshape(-1, 1) * freqs.reshape(1, -1)
        embedding = mx.concatenate([mx.cos(phases), mx.sin(phases)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate(
                [embedding, mx.zeros((embedding.shape[0], 1), dtype=embedding.dtype)], axis=-1
            )
        return embedding

    def __call__(self, t: mx.array) -> mx.array:
        h = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp[2](self.mlp[1](self.mlp[0](h)))


class TrellisFeedForward(nn.Module):
    """Checkpoint-compatible ``Linear -> GELU(tanh) -> Linear`` MLP."""

    def __init__(self, channels: int, mlp_ratio: float):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.mlp = [
            nn.Linear(channels, hidden),
            nn.GELU(approx="precise"),
            nn.Linear(hidden, channels),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # PyTorch's GELU(approximate="tanh") is MLX's gelu_approx.
        return self.mlp[2](nn.gelu_approx(self.mlp[0](x)))


class TrellisMultiHeadAttention(nn.Module):
    """Fused MLX self/cross attention with TRELLIS-compatible projections."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        *,
        context_channels: int | None = None,
        cross: bool = False,
        qkv_bias: bool = True,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.cross = cross
        context_channels = context_channels if context_channels is not None else channels
        if cross:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(context_channels, channels * 2, bias=qkv_bias)
        else:
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        else:
            self._q_rms_norm = None
            self._k_rms_norm = None
        self.to_out = nn.Linear(channels, channels)

    @property
    def qk_rms_norm(self) -> bool:
        return hasattr(self, "q_rms_norm")

    def _attend(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
        # MLX SDPA uses (batch, heads, sequence, head_dim).
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        h = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
        return h.transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, context: mx.array | None = None) -> mx.array:
        batch, length, _ = x.shape
        if self.cross:
            if context is None:
                raise ValueError("cross-attention requires context")
            q = self.to_q(x).reshape(batch, length, self.num_heads, self.head_dim)
            kv = self.to_kv(context).reshape(
                batch, context.shape[1], 2, self.num_heads, self.head_dim
            )
            k, v = kv[:, :, 0], kv[:, :, 1]
        else:
            qkv = self.to_qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        h = self._attend(q, k, v).reshape(batch, length, self.channels)
        return self.to_out(h)

    def cast_linears(self, dtype: mx.Dtype) -> None:
        names = ("to_q", "to_kv") if self.cross else ("to_qkv",)
        for name in (*names, "to_out"):
            layer = getattr(self, name)
            layer.weight = layer.weight.astype(dtype)
            if "bias" in layer:
                layer.bias = layer.bias.astype(dtype)


class TrellisModulatedCrossBlock(nn.Module):
    """TRELLIS adaLN transformer block with self-, cross-attention, and MLP."""

    def __init__(
        self,
        channels: int,
        context_channels: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, affine=False)
        self.norm2 = LayerNorm32(channels, affine=True)
        self.norm3 = LayerNorm32(channels, affine=False)
        self.self_attn = TrellisMultiHeadAttention(channels, num_heads, qk_rms_norm=qk_rms_norm)
        self.cross_attn = TrellisMultiHeadAttention(
            channels,
            num_heads,
            context_channels=context_channels,
            cross=True,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = TrellisFeedForward(channels, mlp_ratio)
        if not share_mod:
            self.adaLN_modulation = [nn.SiLU(), nn.Linear(channels, 6 * channels)]
            self.adaLN_modulation[1].weight = mx.zeros_like(self.adaLN_modulation[1].weight)
            self.adaLN_modulation[1].bias = mx.zeros_like(self.adaLN_modulation[1].bias)

    def __call__(self, x: mx.array, mod: mx.array, context: mx.array) -> mx.array:
        if not self.share_mod:
            mod = self.adaLN_modulation[1](nn.silu(mod))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(mod, 6, axis=-1)

        h = self.norm1(x)
        h = h * (1 + scale_msa[:, None]) + shift_msa[:, None]
        x = x + self.self_attn(h) * gate_msa[:, None]
        x = x + self.cross_attn(self.norm2(x), context)
        h = self.norm3(x)
        h = h * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + self.mlp(h) * gate_mlp[:, None]

    def cast_linears(self, dtype: mx.Dtype) -> None:
        self.self_attn.cast_linears(dtype)
        self.cross_attn.cast_linears(dtype)
        for index in (0, 2):
            layer = self.mlp.mlp[index]
            layer.weight = layer.weight.astype(dtype)
            layer.bias = layer.bias.astype(dtype)
        if not self.share_mod:
            layer = self.adaLN_modulation[1]
            layer.weight = layer.weight.astype(dtype)
            layer.bias = layer.bias.astype(dtype)


__all__ = [
    "LayerNorm32",
    "MultiHeadRMSNorm",
    "TrellisFeedForward",
    "TrellisModulatedCrossBlock",
    "TrellisMultiHeadAttention",
    "TrellisTimestepEmbedder",
    "absolute_position_embedding",
]
