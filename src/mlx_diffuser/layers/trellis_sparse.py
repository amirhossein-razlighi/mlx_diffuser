"""Sparse transformer layers shared by TRELLIS SLAT models and decoders."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .sparse import SparseTensor, sparse_cross_attention, sparse_self_attention
from .trellis import LayerNorm32, MultiHeadRMSNorm


class SparseLinear(nn.Module):
    """Checkpoint-compatible linear projection over sparse point features."""

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True):
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims),
        )
        if bias:
            self.bias = mx.zeros((output_dims,))

    def __call__(self, x: SparseTensor) -> SparseTensor:
        features = x.features @ self.weight.T
        if hasattr(self, "bias"):
            features = features + self.bias
        return x.replace(features)


class SparseFeedForward(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.mlp = [SparseLinear(channels, hidden), nn.GELU(), SparseLinear(hidden, channels)]

    def __call__(self, x: SparseTensor) -> SparseTensor:
        h = self.mlp[0](x)
        h = h.replace(nn.gelu_approx(h.features))
        return self.mlp[2](h)

    def cast_linears(self, dtype: mx.Dtype) -> None:
        for index in (0, 2):
            layer = self.mlp[index]
            layer.weight = layer.weight.astype(dtype)
            layer.bias = layer.bias.astype(dtype)


class SparseMultiHeadAttention(nn.Module):
    """TRELLIS sparse attention backed by MLX fused SDPA."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        *,
        context_channels: int | None = None,
        cross: bool = False,
        window_size: int | None = None,
        shift_window: tuple[int, int, int] = (0, 0, 0),
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
        self.window_size = window_size
        self.shift_window = shift_window
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

    def __call__(self, x: SparseTensor, context: mx.array | None = None) -> SparseTensor:
        points = x.num_points
        if self.cross:
            if context is None:
                raise ValueError("cross-attention requires context")
            q = self.to_q(x.features).reshape(points, self.num_heads, self.head_dim)
            kv = self.to_kv(context).reshape(
                context.shape[0], context.shape[1], 2, self.num_heads, self.head_dim
            )
            k, v = kv[:, :, 0], kv[:, :, 1]
            if self.qk_rms_norm:
                q, k = self.q_rms_norm(q), self.k_rms_norm(k)
            h = sparse_cross_attention(x, q, k, v)
        else:
            qkv = self.to_qkv(x.features).reshape(points, 3, self.num_heads, self.head_dim)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            if self.qk_rms_norm:
                q, k = self.q_rms_norm(q), self.k_rms_norm(k)
            h = sparse_self_attention(
                x,
                q,
                k,
                v,
                window_size=self.window_size,
                shift_window=self.shift_window,
            )
        h = h.reshape(points, self.channels)
        return x.replace(self.to_out(h))

    def cast_linears(self, dtype: mx.Dtype) -> None:
        names = ("to_q", "to_kv") if self.cross else ("to_qkv",)
        for name in (*names, "to_out"):
            layer = getattr(self, name)
            layer.weight = layer.weight.astype(dtype)
            layer.bias = layer.bias.astype(dtype)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """Sparse counterpart of TRELLIS's modulated cross-attention block."""

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
        self.self_attn = SparseMultiHeadAttention(channels, num_heads, qk_rms_norm=qk_rms_norm)
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            num_heads,
            context_channels=context_channels,
            cross=True,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForward(channels, mlp_ratio)
        if not share_mod:
            self.adaLN_modulation = [nn.SiLU(), nn.Linear(channels, 6 * channels)]
            self.adaLN_modulation[1].weight = mx.zeros_like(self.adaLN_modulation[1].weight)
            self.adaLN_modulation[1].bias = mx.zeros_like(self.adaLN_modulation[1].bias)

    def __call__(self, x: SparseTensor, mod: mx.array, context: mx.array) -> SparseTensor:
        if not self.share_mod:
            mod = self.adaLN_modulation[1](nn.silu(mod))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(mod, 6, axis=-1)
        h = x.replace(self.norm1(x.features))
        h = h * (1 + scale_msa) + shift_msa
        x = x + self.self_attn(h) * gate_msa
        x = x + self.cross_attn(x.replace(self.norm2(x.features)), context)
        h = x.replace(self.norm3(x.features))
        h = h * (1 + scale_mlp) + shift_mlp
        return x + self.mlp(h) * gate_mlp

    def cast_linears(self, dtype: mx.Dtype) -> None:
        self.self_attn.cast_linears(dtype)
        self.cross_attn.cast_linears(dtype)
        self.mlp.cast_linears(dtype)
        if not self.share_mod:
            layer = self.adaLN_modulation[1]
            layer.weight = layer.weight.astype(dtype)
            layer.bias = layer.bias.astype(dtype)


class SparseTransformerBlock(nn.Module):
    """Unconditioned sparse transformer block used by TRELLIS representation decoders."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        window_size: int | None = None,
        shift_window: tuple[int, int, int] = (0, 0, 0),
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm32(channels, affine=False)
        self.norm2 = LayerNorm32(channels, affine=False)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads,
            window_size=window_size,
            shift_window=shift_window,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForward(channels, mlp_ratio)

    def __call__(self, x: SparseTensor) -> SparseTensor:
        x = x + self.attn(x.replace(self.norm1(x.features)))
        return x + self.mlp(x.replace(self.norm2(x.features)))

    def cast_linears(self, dtype: mx.Dtype) -> None:
        self.attn.cast_linears(dtype)
        self.mlp.cast_linears(dtype)


__all__ = [
    "ModulatedSparseTransformerCrossBlock",
    "SparseFeedForward",
    "SparseLinear",
    "SparseMultiHeadAttention",
    "SparseTransformerBlock",
]
