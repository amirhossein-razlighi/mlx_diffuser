"""Convolutional 2D blocks for UNet and VAE (channels-last)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention


def valid_groups(channels: int, groups: int) -> int:
    """Largest group count <= ``groups`` that divides ``channels`` (>=1)."""
    g = min(groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class Identity(nn.Module):
    """A no-op module (placeholder where an optional sublayer is absent)."""

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        return x


class ResnetBlock2D(nn.Module):
    """GroupNorm-SiLU-Conv residual block with optional timestep conditioning."""

    def __init__(
        self, in_channels: int, out_channels: int, temb_dim: int | None = None, groups: int = 32
    ):
        super().__init__()
        groups_in = valid_groups(in_channels, groups)
        groups_out = valid_groups(out_channels, groups)
        self.norm1 = nn.GroupNorm(groups_in, in_channels, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(temb_dim, out_channels) if temb_dim else None
        self.norm2 = nn.GroupNorm(groups_out, out_channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def __call__(self, x: mx.array, temb: mx.array | None = None) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        if self.time_proj is not None and temb is not None:
            h = h + self.time_proj(nn.silu(temb))[:, None, None, :]
        h = self.conv2(nn.silu(self.norm2(h)))
        skip = x if self.skip is None else self.skip(x)
        return skip + h


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(self.up(x))


class SpatialAttention(nn.Module):
    """Self/cross attention over the spatial grid of a channels-last feature map."""

    def __init__(
        self, channels: int, num_heads: int, context_dim: int | None = None, groups: int = 32
    ):
        super().__init__()
        self.norm = nn.GroupNorm(valid_groups(channels, groups), channels, pytorch_compatible=True)
        self.attn = Attention(channels, num_heads, context_dim=context_dim)

    def __call__(self, x: mx.array, context: mx.array | None = None) -> mx.array:
        b, h, w, c = x.shape
        residual = x
        y = self.norm(x).reshape(b, h * w, c)
        y = self.attn(y, context=context)
        return residual + y.reshape(b, h, w, c)
