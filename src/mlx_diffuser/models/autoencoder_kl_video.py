"""AutoencoderKLVideo: a causal 3D VAE mapping videos <-> spatiotemporal latents.

This is the video counterpart of :class:`AutoencoderKL`. It compresses a video
``(B, T, H, W, C)`` both spatially and temporally with causal 3D convolutions, the
design shared by the LTX-Video and WAN VAEs. Latent-space video diffusion runs on
``encode(x).sample() * scaling_factor``.

Spatial compression is ``2 ** (len(block_out_channels) - 1)``; temporal
compression is ``temporal_compression`` (a power of two), applied at the deepest
levels. Input ``T`` must be divisible by ``temporal_compression`` and ``H``/``W``
by the spatial compression.
"""

from __future__ import annotations

import dataclasses
import math

import mlx.core as mx
import mlx.nn as nn

from ..configuration import Config
from ..layers.unet_blocks import valid_groups
from ..layers.vae3d_blocks import CausalConv3d, Downsample3D, ResnetBlock3D, Upsample3D
from ..modeling import ModelMixin
from .autoencoder_kl import DiagonalGaussian


@dataclasses.dataclass
class AutoencoderKLVideoConfig(Config):
    in_channels: int = 3
    latent_channels: int = 16
    block_out_channels: tuple[int, ...] = (64, 128, 128)
    layers_per_block: int = 1
    temporal_compression: int = 4  # power of two, <= number of spatial downsamples
    norm_groups: int = 32
    scaling_factor: float = 1.0

    @property
    def num_downsamples(self) -> int:
        return len(self.block_out_channels) - 1

    @property
    def spatial_compression(self) -> int:
        return 2**self.num_downsamples

    @property
    def temporal_levels(self) -> int:
        return int(round(math.log2(self.temporal_compression)))


def _is_temporal_down(level: int, c: AutoencoderKLVideoConfig) -> bool:
    """Whether encoder downsample ``level`` also compresses time (deepest levels do)."""
    return level >= c.num_downsamples - c.temporal_levels


class _Encoder(nn.Module):
    def __init__(self, c: AutoencoderKLVideoConfig):
        super().__init__()
        boc = list(c.block_out_channels)
        n = len(boc)
        self.conv_in = CausalConv3d(c.in_channels, boc[0])
        self.resnets: list[ResnetBlock3D] = []
        self.downsamplers: list[nn.Module | None] = []
        prev = boc[0]
        for i, ch in enumerate(boc):
            for _ in range(c.layers_per_block):
                self.resnets.append(ResnetBlock3D(prev, ch, None, c.norm_groups))
                prev = ch
            self.downsamplers.append(Downsample3D(ch, _is_temporal_down(i, c)) if i < n - 1 else None)
        self.mid_resnet1 = ResnetBlock3D(prev, prev, None, c.norm_groups)
        self.mid_resnet2 = ResnetBlock3D(prev, prev, None, c.norm_groups)
        self.norm_out = nn.GroupNorm(valid_groups(prev, c.norm_groups), prev, pytorch_compatible=True)
        self.conv_out = CausalConv3d(prev, 2 * c.latent_channels)
        self._n = n
        self._lpb = c.layers_per_block

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        ri = 0
        for i in range(self._n):
            for _ in range(self._lpb):
                x = self.resnets[ri](x)
                ri += 1
            down = self.downsamplers[i]
            if down is not None:
                x = down(x)
        x = self.mid_resnet1(x)
        x = self.mid_resnet2(x)
        return self.conv_out(nn.silu(self.norm_out(x)))


class _Decoder(nn.Module):
    def __init__(self, c: AutoencoderKLVideoConfig):
        super().__init__()
        boc = list(c.block_out_channels)
        n = len(boc)
        rev = list(reversed(boc))
        self.conv_in = CausalConv3d(c.latent_channels, rev[0])
        self.mid_resnet1 = ResnetBlock3D(rev[0], rev[0], None, c.norm_groups)
        self.mid_resnet2 = ResnetBlock3D(rev[0], rev[0], None, c.norm_groups)
        self.resnets: list[ResnetBlock3D] = []
        self.upsamplers: list[nn.Module | None] = []
        prev = rev[0]
        for j, ch in enumerate(rev):
            for _ in range(c.layers_per_block + 1):
                self.resnets.append(ResnetBlock3D(prev, ch, None, c.norm_groups))
                prev = ch
            # Upsample j inverts encoder downsample (n-2-j); temporal on the deepest first.
            temporal = j <= c.temporal_levels - 1
            self.upsamplers.append(Upsample3D(ch, temporal) if j < n - 1 else None)
        self.norm_out = nn.GroupNorm(valid_groups(prev, c.norm_groups), prev, pytorch_compatible=True)
        self.conv_out = CausalConv3d(prev, c.in_channels)
        self._n = n
        self._lpb = c.layers_per_block

    def __call__(self, z: mx.array) -> mx.array:
        x = self.conv_in(z)
        x = self.mid_resnet1(x)
        x = self.mid_resnet2(x)
        ri = 0
        for j in range(self._n):
            for _ in range(self._lpb + 1):
                x = self.resnets[ri](x)
                ri += 1
            up = self.upsamplers[j]
            if up is not None:
                x = up(x)
        return self.conv_out(nn.silu(self.norm_out(x)))


class AutoencoderKLVideo(ModelMixin[AutoencoderKLVideoConfig]):
    config_class = AutoencoderKLVideoConfig

    def __init__(self, config: AutoencoderKLVideoConfig):
        super().__init__()
        self.config = config
        self.encoder = _Encoder(config)
        self.decoder = _Decoder(config)

    @property
    def scaling_factor(self) -> float:
        return self.config.scaling_factor

    def encode(self, x: mx.array) -> DiagonalGaussian:
        return DiagonalGaussian(self.encoder(x))

    def decode(self, z: mx.array) -> mx.array:
        return self.decoder(z)

    def __call__(
        self, x: mx.array, key: mx.array | None = None
    ) -> tuple[mx.array, DiagonalGaussian]:
        posterior = self.encode(x)
        return self.decode(posterior.sample(key)), posterior
