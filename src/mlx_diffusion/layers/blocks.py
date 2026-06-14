"""Transformer building blocks (DiT-style adaLN-Zero block)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention
from .normalization import AdaLNModulation, modulate


class FeedForward(nn.Module):
    """Standard MLP with GELU activation."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """A diffusion-transformer block with adaLN-Zero conditioning.

    Self-attention and MLP are each wrapped in (norm -> modulate -> sublayer ->
    gate), where shift/scale/gate are predicted from the conditioning vector ``c``.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, cond_dim: int | None = None):
        super().__init__()
        cond_dim = cond_dim or dim
        self.norm1 = nn.LayerNorm(dim, affine=False)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, affine=False)
        self.mlp = FeedForward(dim, mlp_ratio)
        self.modulation = AdaLNModulation(cond_dim, dim, n=6)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(c)
        x = x + gate_msa[:, None, :] * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp[:, None, :] * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """adaLN-Zero final projection back to patch pixels."""

    def __init__(self, dim: int, patch_dim: int, cond_dim: int | None = None):
        super().__init__()
        cond_dim = cond_dim or dim
        self.norm = nn.LayerNorm(dim, affine=False)
        self.modulation = AdaLNModulation(cond_dim, dim, n=2)
        self.linear = nn.Linear(dim, patch_dim)
        # Zero-init the output projection (adaLN-Zero).
        self.linear.weight = mx.zeros_like(self.linear.weight)
        self.linear.bias = mx.zeros_like(self.linear.bias)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        shift, scale = self.modulation(c)
        return self.linear(modulate(self.norm(x), shift, scale))
