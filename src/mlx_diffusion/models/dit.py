"""Diffusion Transformer (DiT).

A patchify -> transformer -> unpatchify backbone with adaLN-Zero conditioning,
following Peebles & Xie (2023). It is the general-purpose network in this library:
image generation today, and (by treating frames/tokens as the sequence) a path to
video and discrete modalities later. Pairs naturally with flow-matching.

Tensors are channels-last: input/output are ``(B, H, W, C)``.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx

from ..configuration import Config
from ..layers.blocks import DiTBlock, FinalLayer
from ..layers.embeddings import (
    LabelEmbedder,
    PatchEmbed,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
)
from ..modeling import ModelMixin


@dataclasses.dataclass
class DiTConfig(Config):
    in_channels: int = 4
    out_channels: int | None = None
    patch_size: int = 2
    hidden_size: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    num_classes: int = 0  # 0 => unconditional (no label embedding)
    class_dropout_prob: float = 0.1

    @property
    def resolved_out_channels(self) -> int:
        return self.out_channels if self.out_channels is not None else self.in_channels


class DiT(ModelMixin[DiTConfig]):
    config_class = DiTConfig

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        if config.hidden_size % 4 != 0:
            raise ValueError("hidden_size must be divisible by 4 (2D positional embedding).")

        self.patch_embed = PatchEmbed(config.in_channels, config.hidden_size, config.patch_size)
        self.t_embed = TimestepEmbedder(config.hidden_size)
        self.y_embed = (
            LabelEmbedder(config.num_classes, config.hidden_size, config.class_dropout_prob)
            if config.num_classes > 0
            else None
        )
        self.blocks = [
            DiTBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
            for _ in range(config.depth)
        ]
        patch_dim = config.patch_size**2 * config.resolved_out_channels
        self.final = FinalLayer(config.hidden_size, patch_dim)

    def _unpatchify(self, x: mx.array, h: int, w: int) -> mx.array:
        p = self.config.patch_size
        c = self.config.resolved_out_channels
        b = x.shape[0]
        x = x.reshape(b, h, w, p, p, c)
        x = x.transpose(0, 1, 3, 2, 4, 5)  # (b, h, p, w, p, c)
        return x.reshape(b, h * p, w * p, c)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        y: mx.array | None = None,
        *,
        training: bool = False,
        key: mx.array | None = None,
    ) -> mx.array:
        tokens, (h, w) = self.patch_embed(x)
        pos = get_2d_sincos_pos_embed(self.config.hidden_size, h, w)
        tokens = tokens + pos[None].astype(tokens.dtype)

        c = self.t_embed(t)
        if self.y_embed is not None:
            if y is None:
                raise ValueError("This DiT is class-conditional; pass labels `y`.")
            c = c + self.y_embed(y, training=training, key=key)

        for block in self.blocks:
            tokens = block(tokens, c)

        tokens = self.final(tokens, c)
        return self._unpatchify(tokens, h, w)
