"""Causal 3D-convolution blocks for the video VAE (channels-last ``(B, T, H, W, C)``).

Video VAEs use *causal* time convolutions so a frame only depends on itself and
past frames — this lets the encoder/decoder stream and keeps the first frame
well-defined. Time is padded on the past side only (replicate), while height and
width are padded symmetrically.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .unet_blocks import valid_groups


class CausalConv3d(nn.Module):
    """3D conv with causal padding on time and symmetric padding on H/W."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: tuple[int, int, int] = (1, 1, 1),
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def __call__(self, x: mx.array) -> mx.array:
        k = self.kernel_size
        pad_t = k - 1  # causal: all padding on the past
        pad_hw = k // 2
        x = mx.pad(
            x,
            [(0, 0), (pad_t, 0), (pad_hw, pad_hw), (pad_hw, pad_hw), (0, 0)],
            mode="edge",  # replicate first frame / border pixels
        )
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    """GroupNorm-SiLU-CausalConv3d residual block with optional time conditioning."""

    def __init__(
        self, in_channels: int, out_channels: int, temb_dim: int | None = None, groups: int = 32
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(valid_groups(in_channels, groups), in_channels, pytorch_compatible=True)
        self.conv1 = CausalConv3d(in_channels, out_channels)
        self.time_proj = nn.Linear(temb_dim, out_channels) if temb_dim else None
        self.norm2 = nn.GroupNorm(valid_groups(out_channels, groups), out_channels, pytorch_compatible=True)
        self.conv2 = CausalConv3d(out_channels, out_channels)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def __call__(self, x: mx.array, temb: mx.array | None = None) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        if self.time_proj is not None and temb is not None:
            h = h + self.time_proj(nn.silu(temb))[:, None, None, None, :]
        h = self.conv2(nn.silu(self.norm2(h)))
        skip = x if self.skip is None else self.skip(x)
        return skip + h


class Downsample3D(nn.Module):
    """Halve H/W (and optionally T) with a strided causal conv."""

    def __init__(self, channels: int, temporal: bool):
        super().__init__()
        stride = (2 if temporal else 1, 2, 2)
        self.conv = CausalConv3d(channels, channels, stride=stride)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Upsample3D(nn.Module):
    """Double H/W (and optionally T) with nearest upsampling + a causal conv."""

    def __init__(self, channels: int, temporal: bool):
        super().__init__()
        scale = (2.0 if temporal else 1.0, 2.0, 2.0)
        self.up = nn.Upsample(scale_factor=scale, mode="nearest")
        self.conv = CausalConv3d(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(self.up(x))
