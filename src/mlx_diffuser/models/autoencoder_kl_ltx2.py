"""LTX2VideoDecoder: the decoder half of LTX-2's causal video VAE, in MLX.

Mirrors ltx-core's ``VideoDecoder`` (the original checkpoint layout, which for
LTX-2.3 differs from the diffusers 2.0 port: nine sequential ``up_blocks``
rather than mid+3). The VAE compresses video 32x spatially and 8x temporally
into 128 latent channels; decoding walks the block list in reverse — causal 3D
convs (temporal padding by edge-frame repetition), pixel-norm (per-location
RMS over channels), depth-to-space upsamplers that drop the duplicated first
frame after temporal expansion — and ends with a 4x4 spatial unpatchify.

Only the decoder is ported: text-to-video never encodes pixels, and skipping
the encoder keeps 0.35B parameters out of memory. Latents are denoised in the
VAE's normalized space; call :meth:`denormalize_latents` before :meth:`decode`.
Tensors are channels-last ``(B, F, H, W, C)``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..configuration import Config
from ..modeling import ModelMixin

# (block_name, params) in the original config order; the decoder runs them reversed.
_LTX_2_3_DECODER_BLOCKS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("res_x", {"num_layers": 4}),
    ("compress_space", {"multiplier": 2}),
    ("res_x", {"num_layers": 6}),
    ("compress_time", {"multiplier": 2}),
    ("res_x", {"num_layers": 4}),
    ("compress_all", {"multiplier": 1}),
    ("res_x", {"num_layers": 2}),
    ("compress_all", {"multiplier": 2}),
    ("res_x", {"num_layers": 2}),
)


@dataclasses.dataclass
class LTX2VAEDecoderConfig(Config):
    latent_channels: int = 128
    out_channels: int = 3
    base_channels: int = 128
    patch_size: int = 4
    decoder_blocks: tuple[tuple[str, dict[str, Any]], ...] = _LTX_2_3_DECODER_BLOCKS
    causal: bool = False  # LTX-2.3 decodes non-causally (symmetric temporal padding)
    temporal_compression: int = 8
    spatial_compression: int = 32

    @classmethod
    def ltx_2_3(cls) -> LTX2VAEDecoderConfig:
        return cls()  # defaults match the LTX-2.3 checkpoint


def _pixel_norm(x: mx.array, eps: float = 1e-8) -> mx.array:
    xf = x.astype(mx.float32)
    return (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)).astype(x.dtype)


class CausalConv3d(nn.Module):
    """3x3x3 conv with spatial zero padding; temporal padding repeats edge frames.

    ``causal=True`` pads only with the first frame (kernel-1 copies); otherwise
    the padding is symmetric (first and last frame).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.time_pad = kernel_size - 1
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(0, kernel_size // 2, kernel_size // 2),
        )

    def __call__(self, x: mx.array, causal: bool) -> mx.array:
        if causal:
            front = mx.repeat(x[:, :1], self.time_pad, axis=1)
            x = mx.concatenate([front, x], axis=1)
        else:
            front = mx.repeat(x[:, :1], self.time_pad // 2, axis=1)
            back = mx.repeat(x[:, -1:], self.time_pad // 2, axis=1)
            x = mx.concatenate([front, x, back], axis=1)
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    """pixel-norm -> silu -> conv, twice, with an identity shortcut."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = CausalConv3d(channels, channels)
        self.conv2 = CausalConv3d(channels, channels)

    def __call__(self, x: mx.array, causal: bool) -> mx.array:
        h = self.conv1(nn.silu(_pixel_norm(x)), causal)
        h = self.conv2(nn.silu(_pixel_norm(h)), causal)
        return x + h


class UNetMidBlock3D(nn.Module):
    def __init__(self, channels: int, num_layers: int):
        super().__init__()
        self.res_blocks = [ResnetBlock3D(channels) for _ in range(num_layers)]

    def __call__(self, x: mx.array, causal: bool) -> mx.array:
        for block in self.res_blocks:
            x = block(x, causal)
        return x


class DepthToSpaceUpsample(nn.Module):
    """Conv to ``prod(stride) * C / multiplier`` channels, then depth-to-space.

    After temporal expansion the first output frame is dropped (it corresponds
    to the causal duplicate the encoder introduced).
    """

    def __init__(self, in_channels: int, stride: tuple[int, int, int], multiplier: int):
        super().__init__()
        self.stride = stride
        self.out_channels = math.prod(stride) * in_channels // multiplier
        self.conv = CausalConv3d(in_channels, self.out_channels)

    def __call__(self, x: mx.array, causal: bool) -> mx.array:
        x = self.conv(x, causal)
        b, f, h, w, _ = x.shape
        st, sh, sw = self.stride
        c = x.shape[-1] // (st * sh * sw)
        # channels-last depth-to-space: channel index is (c, p1, p2, p3), c outermost
        x = x.reshape(b, f, h, w, c, st, sh, sw)
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)  # (B, f, st, h, sh, w, sw, c)
        x = x.reshape(b, f * st, h * sh, w * sw, c)
        if st == 2:
            x = x[:, 1:]
        return x


class LTX2VideoDecoder(ModelMixin[LTX2VAEDecoderConfig]):
    """LTX-2 video VAE decoder (channels-last ``(B, F', H', W', 128)`` latents)."""

    config_class = LTX2VAEDecoderConfig

    def __init__(self, config: LTX2VAEDecoderConfig):
        super().__init__()
        self.config = config
        # 3 channel-multiplying upsamplers in the standard config: base * 8.
        channels = config.base_channels * 8
        self.conv_in = CausalConv3d(config.latent_channels, channels)

        blocks: list[nn.Module] = []
        for name, params in reversed(config.decoder_blocks):
            if name == "res_x":
                blocks.append(UNetMidBlock3D(channels, params["num_layers"]))
            elif name in ("compress_space", "compress_time", "compress_all"):
                stride = {
                    "compress_space": (1, 2, 2),
                    "compress_time": (2, 1, 1),
                    "compress_all": (2, 2, 2),
                }[name]
                up = DepthToSpaceUpsample(channels, stride, params.get("multiplier", 1))
                channels = channels // params.get("multiplier", 1)
                blocks.append(up)
            else:
                raise ValueError(f"unknown decoder block: {name}")
        self.up_blocks = blocks

        self.conv_out = CausalConv3d(channels, config.out_channels * config.patch_size**2)
        self.latents_mean = mx.zeros((config.latent_channels,))
        self.latents_std = mx.ones((config.latent_channels,))

    def denormalize_latents(self, z: mx.array) -> mx.array:
        """Map latents from the (normalized) diffusion space to the VAE space."""
        return z * self.latents_std + self.latents_mean

    def decode(self, z: mx.array) -> mx.array:
        """Decode denormalized latents ``(B, F', H', W', 128)`` to ``[-1, 1]`` video.

        Output is ``(B, 8*(F'-1)+1, 32*H', 32*W', 3)``.
        """
        causal = self.config.causal
        x = self.conv_in(z, causal)
        for block in self.up_blocks:
            x = block(x, causal)
        x = self.conv_out(nn.silu(_pixel_norm(x)), causal)

        # unpatchify: channel index is (c, p_h, p_w) -> expand H by p_h... (see below)
        p = self.config.patch_size
        b, f, h, w, _ = x.shape
        c = x.shape[-1] // (p * p)
        # original: rearrange "b (c r q) f h w -> b c f (h q) (w r)" — q rows, r cols,
        # with channel index (c, r, q): r is the *column* offset, q the *row* offset.
        x = x.reshape(b, f, h, w, c, p, p)  # (..., c, r, q)
        x = x.transpose(0, 1, 2, 6, 3, 5, 4)  # (B, f, h, q, w, r, c)
        return x.reshape(b, f, h * p, w * p, c)
