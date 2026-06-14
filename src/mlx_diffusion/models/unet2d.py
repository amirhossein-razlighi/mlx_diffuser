"""UNet2D: a Stable-Diffusion-style convolutional denoiser.

Channels-last ``(B, H, W, C)``. Supports timestep conditioning, optional spatial
self/cross-attention per level (cross-attention enables text conditioning via a
``context`` sequence), and the standard down/mid/up skip-connection structure.

Like every model here it is config-driven and scales down to tiny dims for tests.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn

from ..configuration import Config
from ..modeling import ModelMixin
from ..layers.embeddings import TimestepEmbedder
from ..layers.unet_blocks import (
    Downsample2D,
    Identity,
    ResnetBlock2D,
    SpatialAttention,
    Upsample2D,
    valid_groups,
)


@dataclasses.dataclass
class UNet2DConfig(Config):
    in_channels: int = 4
    out_channels: int = 4
    block_out_channels: tuple[int, ...] = (224, 448, 672, 896)
    layers_per_block: int = 2
    attention_levels: tuple[bool, ...] | None = None  # per level; default: all but first
    num_heads: int = 8
    cross_attention_dim: int | None = None
    norm_groups: int = 32

    @property
    def time_embed_dim(self) -> int:
        return self.block_out_channels[0] * 4


class UNet2D(ModelMixin):
    config_class = UNet2DConfig

    def __init__(self, config: UNet2DConfig):
        super().__init__()
        self.config = config
        boc = list(config.block_out_channels)
        n = len(boc)
        lpb = config.layers_per_block
        groups = config.norm_groups
        heads = config.num_heads
        ctx = config.cross_attention_dim
        attn_levels = (
            list(config.attention_levels)
            if config.attention_levels is not None
            else [i > 0 for i in range(n)]
        )

        self.time_embed = TimestepEmbedder(config.time_embed_dim)
        self.conv_in = nn.Conv2d(config.in_channels, boc[0], 3, padding=1)

        # --- down path ---
        self.down_resnets: list[ResnetBlock2D] = []
        self.down_attns: list[nn.Module] = []
        self.downsamplers: list[nn.Module] = []
        skip_channels = [boc[0]]
        prev = boc[0]
        for i, ch in enumerate(boc):
            for _ in range(lpb):
                self.down_resnets.append(ResnetBlock2D(prev, ch, config.time_embed_dim, groups))
                self.down_attns.append(
                    SpatialAttention(ch, heads, ctx, groups) if attn_levels[i] else Identity()
                )
                prev = ch
                skip_channels.append(ch)
            if i < n - 1:
                self.downsamplers.append(Downsample2D(ch))
                skip_channels.append(ch)
            else:
                self.downsamplers.append(Identity())

        # --- mid ---
        mid_ch = boc[-1]
        self.mid_resnet1 = ResnetBlock2D(mid_ch, mid_ch, config.time_embed_dim, groups)
        self.mid_attn = SpatialAttention(mid_ch, heads, ctx, groups)
        self.mid_resnet2 = ResnetBlock2D(mid_ch, mid_ch, config.time_embed_dim, groups)

        # --- up path ---
        self.up_resnets: list[ResnetBlock2D] = []
        self.up_attns: list[nn.Module] = []
        self.upsamplers: list[nn.Module] = []
        for i, ch in enumerate(reversed(boc)):
            level = n - 1 - i
            for _ in range(lpb + 1):
                skip = skip_channels.pop()
                self.up_resnets.append(ResnetBlock2D(prev + skip, ch, config.time_embed_dim, groups))
                self.up_attns.append(
                    SpatialAttention(ch, heads, ctx, groups) if attn_levels[level] else Identity()
                )
                prev = ch
            self.upsamplers.append(Upsample2D(ch) if i < n - 1 else Identity())

        self.norm_out = nn.GroupNorm(valid_groups(boc[0], groups), boc[0], pytorch_compatible=True)
        self.conv_out = nn.Conv2d(boc[0], config.out_channels, 3, padding=1)

        self._n_levels = n
        self._lpb = lpb

    def __call__(self, x: mx.array, t: mx.array, context: mx.array | None = None) -> mx.array:
        temb = self.time_embed(t)
        x = self.conv_in(x)

        skips = [x]
        ri = 0
        for i in range(self._n_levels):
            for _ in range(self._lpb):
                x = self.down_resnets[ri](x, temb)
                x = self.down_attns[ri](x, context)
                skips.append(x)
                ri += 1
            if i < self._n_levels - 1:
                x = self.downsamplers[i](x)
                skips.append(x)

        x = self.mid_resnet1(x, temb)
        x = self.mid_attn(x, context)
        x = self.mid_resnet2(x, temb)

        ri = 0
        for i in range(self._n_levels):
            for _ in range(self._lpb + 1):
                x = mx.concatenate([x, skips.pop()], axis=-1)
                x = self.up_resnets[ri](x, temb)
                x = self.up_attns[ri](x, context)
                ri += 1
            x = self.upsamplers[i](x)

        x = self.conv_out(nn.silu(self.norm_out(x)))
        return x
