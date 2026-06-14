"""AutoencoderKL: a VAE mapping images <-> latents for latent diffusion.

Channels-last. The encoder outputs Gaussian moments; the decoder reconstructs the
image. Latent-space diffusion models operate on ``encode(x).sample() * scaling``.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn

from ..configuration import Config
from ..modeling import ModelMixin
from ..layers.unet_blocks import (
    Downsample2D,
    ResnetBlock2D,
    SpatialAttention,
    Upsample2D,
    valid_groups,
)


class DiagonalGaussian:
    """A diagonal Gaussian parameterized by concatenated ``[mean, logvar]``."""

    def __init__(self, parameters: mx.array):
        self.mean, logvar = mx.split(parameters, 2, axis=-1)
        self.logvar = mx.clip(logvar, -30.0, 20.0)
        self.std = mx.exp(0.5 * self.logvar)

    def sample(self, key: mx.array | None = None) -> mx.array:
        key = key if key is not None else mx.random.key(0)
        return self.mean + self.std * mx.random.normal(self.mean.shape, key=key)

    def mode(self) -> mx.array:
        return self.mean

    def kl(self) -> mx.array:
        return 0.5 * mx.sum(self.mean**2 + mx.exp(self.logvar) - 1.0 - self.logvar)


@dataclasses.dataclass
class AutoencoderKLConfig(Config):
    in_channels: int = 3
    latent_channels: int = 4
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_groups: int = 32
    scaling_factor: float = 0.18215


class _Encoder(nn.Module):
    def __init__(self, c: AutoencoderKLConfig):
        super().__init__()
        boc = list(c.block_out_channels)
        n = len(boc)
        self.conv_in = nn.Conv2d(c.in_channels, boc[0], 3, padding=1)
        self.resnets: list[ResnetBlock2D] = []
        self.downsamplers: list[nn.Module] = []
        prev = boc[0]
        for i, ch in enumerate(boc):
            for _ in range(c.layers_per_block):
                self.resnets.append(ResnetBlock2D(prev, ch, None, c.norm_groups))
                prev = ch
            self.downsamplers.append(Downsample2D(ch) if i < n - 1 else None)
        self.mid_resnet1 = ResnetBlock2D(prev, prev, None, c.norm_groups)
        self.mid_attn = SpatialAttention(prev, max(1, prev // 64), None, c.norm_groups)
        self.mid_resnet2 = ResnetBlock2D(prev, prev, None, c.norm_groups)
        self.norm_out = nn.GroupNorm(valid_groups(prev, c.norm_groups), prev, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(prev, 2 * c.latent_channels, 3, padding=1)
        self._n = n
        self._lpb = c.layers_per_block

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        ri = 0
        for i in range(self._n):
            for _ in range(self._lpb):
                x = self.resnets[ri](x)
                ri += 1
            if self.downsamplers[i] is not None:
                x = self.downsamplers[i](x)
        x = self.mid_resnet1(x)
        x = self.mid_attn(x)
        x = self.mid_resnet2(x)
        return self.conv_out(nn.silu(self.norm_out(x)))


class _Decoder(nn.Module):
    def __init__(self, c: AutoencoderKLConfig):
        super().__init__()
        boc = list(c.block_out_channels)
        n = len(boc)
        rev = list(reversed(boc))
        self.conv_in = nn.Conv2d(c.latent_channels, rev[0], 3, padding=1)
        self.mid_resnet1 = ResnetBlock2D(rev[0], rev[0], None, c.norm_groups)
        self.mid_attn = SpatialAttention(rev[0], max(1, rev[0] // 64), None, c.norm_groups)
        self.mid_resnet2 = ResnetBlock2D(rev[0], rev[0], None, c.norm_groups)
        self.resnets: list[ResnetBlock2D] = []
        self.upsamplers: list[nn.Module] = []
        prev = rev[0]
        for i, ch in enumerate(rev):
            for _ in range(c.layers_per_block + 1):
                self.resnets.append(ResnetBlock2D(prev, ch, None, c.norm_groups))
                prev = ch
            self.upsamplers.append(Upsample2D(ch) if i < n - 1 else None)
        self.norm_out = nn.GroupNorm(valid_groups(prev, c.norm_groups), prev, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(prev, c.in_channels, 3, padding=1)
        self._n = n
        self._lpb = c.layers_per_block

    def __call__(self, z: mx.array) -> mx.array:
        x = self.conv_in(z)
        x = self.mid_resnet1(x)
        x = self.mid_attn(x)
        x = self.mid_resnet2(x)
        ri = 0
        for i in range(self._n):
            for _ in range(self._lpb + 1):
                x = self.resnets[ri](x)
                ri += 1
            if self.upsamplers[i] is not None:
                x = self.upsamplers[i](x)
        return self.conv_out(nn.silu(self.norm_out(x)))


class AutoencoderKL(ModelMixin):
    config_class = AutoencoderKLConfig

    def __init__(self, config: AutoencoderKLConfig):
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

    def __call__(self, x: mx.array, key: mx.array | None = None) -> tuple[mx.array, DiagonalGaussian]:
        posterior = self.encode(x)
        return self.decode(posterior.sample(key)), posterior
