"""Native MLX components for Microsoft's TRELLIS image-to-3D model.

Milestone one covers the complete dense sparse-structure stage: a 16^3 flow
transformer predicts an occupancy latent and a compact Conv3D decoder expands it to
the 64^3 voxel coordinates consumed by TRELLIS's sparse structured-latent stage.
All public volumes are channels-last ``(B, D, H, W, C)``.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..configuration import Config
from ..layers.trellis import (
    LayerNorm32,
    TrellisModulatedCrossBlock,
    TrellisTimestepEmbedder,
    absolute_position_embedding,
)
from ..modeling import ModelMixin


@dataclasses.dataclass
class TrellisSparseStructureFlowConfig(Config):
    resolution: int = 16
    in_channels: int = 8
    model_channels: int = 1024
    cond_channels: int = 1024
    out_channels: int = 8
    num_blocks: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    patch_size: int = 1
    pe_mode: str = "ape"
    use_fp16: bool = True
    share_mod: bool = False
    qk_rms_norm: bool = True
    qk_rms_norm_cross: bool = False


class TrellisSparseStructureFlowModel(ModelMixin[TrellisSparseStructureFlowConfig]):
    """Checkpoint-compatible dense TRELLIS sparse-structure flow transformer."""

    config_class = TrellisSparseStructureFlowConfig

    def __init__(self, config: TrellisSparseStructureFlowConfig):
        super().__init__()
        self.config = config
        if config.resolution % config.patch_size:
            raise ValueError("resolution must be divisible by patch_size")
        if config.pe_mode != "ape":
            raise NotImplementedError("the first native TRELLIS release supports pe_mode='ape'")
        if config.model_channels % config.num_heads:
            raise ValueError("model_channels must be divisible by num_heads")

        self.t_embedder = TrellisTimestepEmbedder(config.model_channels)
        if config.share_mod:
            self.adaLN_modulation = [
                nn.SiLU(),
                nn.Linear(config.model_channels, 6 * config.model_channels),
            ]
            self.adaLN_modulation[1].weight = mx.zeros_like(self.adaLN_modulation[1].weight)
            self.adaLN_modulation[1].bias = mx.zeros_like(self.adaLN_modulation[1].bias)

        token_resolution = config.resolution // config.patch_size
        grid = list(
            mx.meshgrid(
                *[mx.arange(token_resolution, dtype=mx.float32) for _ in range(3)],
                indexing="ij",
            )
        )
        coords = mx.stack(grid, axis=-1).reshape(-1, 3)
        # Kept public so the key matches the official persistent PyTorch buffer.
        self.pos_emb = absolute_position_embedding(coords, config.model_channels)

        patch_channels = config.in_channels * config.patch_size**3
        self.input_layer = nn.Linear(patch_channels, config.model_channels)
        self.blocks = [
            TrellisModulatedCrossBlock(
                config.model_channels,
                config.cond_channels,
                config.num_heads,
                config.mlp_ratio,
                share_mod=config.share_mod,
                qk_rms_norm=config.qk_rms_norm,
                qk_rms_norm_cross=config.qk_rms_norm_cross,
            )
            for _ in range(config.num_blocks)
        ]
        self.out_layer = nn.Linear(
            config.model_channels, config.out_channels * config.patch_size**3
        )
        self.out_layer.weight = mx.zeros_like(self.out_layer.weight)
        self.out_layer.bias = mx.zeros_like(self.out_layer.bias)

        if config.use_fp16:
            for block in self.blocks:
                block.cast_linears(mx.float16)
            if config.share_mod:
                layer = self.adaLN_modulation[1]
                layer.weight = layer.weight.astype(mx.float16)
                layer.bias = layer.bias.astype(mx.float16)

    def _patchify(self, x: mx.array) -> mx.array:
        p = self.config.patch_size
        b, depth, height, width, channels = x.shape
        x = x.reshape(b, depth // p, p, height // p, p, width // p, p, channels)
        # Match PyTorch's channel-major patch order: C, pD, pH, pW.
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
        return x.reshape(b, -1, channels * p**3)

    def _unpatchify(self, x: mx.array) -> mx.array:
        p = self.config.patch_size
        r = self.config.resolution // p
        b = x.shape[0]
        x = x.reshape(b, r, r, r, self.config.out_channels, p, p, p)
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        return x.reshape(
            b,
            self.config.resolution,
            self.config.resolution,
            self.config.resolution,
            self.config.out_channels,
        )

    def __call__(self, x: mx.array, t: mx.array, cond: mx.array) -> mx.array:
        expected = (
            x.shape[0],
            self.config.resolution,
            self.config.resolution,
            self.config.resolution,
            self.config.in_channels,
        )
        if x.shape != expected:
            raise ValueError(f"input shape must be {expected}, got {x.shape}")
        if t.ndim == 0:
            t = mx.broadcast_to(t, (x.shape[0],))
        if t.shape != (x.shape[0],):
            raise ValueError(f"timesteps must have shape ({x.shape[0]},), got {t.shape}")
        if (
            cond.ndim != 3
            or cond.shape[0] != x.shape[0]
            or cond.shape[-1] != self.config.cond_channels
        ):
            raise ValueError(
                "conditioning must have shape "
                f"(B, tokens, {self.config.cond_channels}), got {cond.shape}"
            )

        h = self.input_layer(self._patchify(x)) + self.pos_emb[None]
        mod = self.t_embedder(t)
        if self.config.share_mod:
            mod = self.adaLN_modulation[1](nn.silu(mod))
        if self.config.use_fp16:
            h = h.astype(mx.float16)
            mod = mod.astype(mx.float16)
            cond = cond.astype(mx.float16)
        for block in self.blocks:
            h = block(h, mod, cond)
        h = LayerNorm32(self.config.model_channels, affine=False, eps=1e-5)(h.astype(x.dtype))
        return self._unpatchify(self.out_layer(h))


@dataclasses.dataclass
class TrellisSparseStructureDecoderConfig(Config):
    out_channels: int = 1
    latent_channels: int = 8
    num_res_blocks: int = 2
    channels: tuple[int, ...] = (512, 128, 32)
    num_res_blocks_middle: int = 2
    norm_type: str = "layer"
    use_fp16: bool = True


class _TrellisResBlock3D(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        out_channels = out_channels or channels
        self.norm1 = LayerNorm32(channels, affine=True, eps=1e-5)
        self.norm2 = LayerNorm32(out_channels, affine=True, eps=1e-5)
        self.conv1 = nn.Conv3d(channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.conv2.weight = mx.zeros_like(self.conv2.weight)
        self.conv2.bias = mx.zeros_like(self.conv2.bias)
        self.skip_connection = (
            nn.Conv3d(channels, out_channels, 1) if channels != out_channels else nn.Identity()
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        return h + self.skip_connection(x)

    def cast_convs(self, dtype: mx.Dtype) -> None:
        for layer in (self.conv1, self.conv2):
            layer.weight = layer.weight.astype(dtype)
            layer.bias = layer.bias.astype(dtype)
        if not isinstance(self.skip_connection, nn.Identity):
            self.skip_connection.weight = self.skip_connection.weight.astype(dtype)
            self.skip_connection.bias = self.skip_connection.bias.astype(dtype)


class _TrellisUpsampleBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        b, depth, height, width, _ = x.shape
        # Channel order mirrors the reference pixel_shuffle_3d implementation.
        x = x.reshape(b, depth, height, width, self.out_channels, 2, 2, 2)
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        return x.reshape(b, depth * 2, height * 2, width * 2, self.out_channels)

    def cast_convs(self, dtype: mx.Dtype) -> None:
        self.conv.weight = self.conv.weight.astype(dtype)
        self.conv.bias = self.conv.bias.astype(dtype)


class TrellisSparseStructureDecoder(ModelMixin[TrellisSparseStructureDecoderConfig]):
    """Decode a TRELLIS 16^3 latent to 64^3 occupancy logits."""

    config_class = TrellisSparseStructureDecoderConfig

    def __init__(self, config: TrellisSparseStructureDecoderConfig):
        super().__init__()
        self.config = config
        if config.norm_type != "layer":
            raise NotImplementedError("the official TRELLIS decoder uses norm_type='layer'")
        if not config.channels:
            raise ValueError("channels cannot be empty")

        self.input_layer = nn.Conv3d(config.latent_channels, config.channels[0], 3, padding=1)
        self.middle_block = [
            _TrellisResBlock3D(config.channels[0]) for _ in range(config.num_res_blocks_middle)
        ]
        blocks: list[nn.Module] = []
        for i, channels in enumerate(config.channels):
            blocks.extend(_TrellisResBlock3D(channels) for _ in range(config.num_res_blocks))
            if i < len(config.channels) - 1:
                blocks.append(_TrellisUpsampleBlock3D(channels, config.channels[i + 1]))
        self.blocks = blocks
        self.out_layer = [
            LayerNorm32(config.channels[-1], affine=True, eps=1e-5),
            nn.SiLU(),
            nn.Conv3d(config.channels[-1], config.out_channels, 3, padding=1),
        ]

        if config.use_fp16:
            for block in [*self.middle_block, *self.blocks]:
                block.cast_convs(mx.float16)

    @property
    def scale_factor(self) -> int:
        return 2 ** (len(self.config.channels) - 1)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 5 or x.shape[-1] != self.config.latent_channels:
            raise ValueError(
                f"input must have shape (B, D, H, W, {self.config.latent_channels}), got {x.shape}"
            )
        h = self.input_layer(x)
        if self.config.use_fp16:
            h = h.astype(mx.float16)
        for middle_block in self.middle_block:
            h = middle_block(h)
        for decoder_block in self.blocks:
            h = decoder_block(h)
        h = h.astype(x.dtype)
        return self.out_layer[2](nn.silu(self.out_layer[0](h)))

    def occupied_coordinates(self, latent: mx.array, threshold: float = 0.0) -> mx.array:
        """Return TRELLIS coordinates ``(batch, x, y, z)`` above ``threshold``."""

        logits = self(latent)[..., 0]
        # MLX intentionally has no nonzero/argwhere primitive. Occupancy topology is
        # non-differentiable, so materializing this small 64^3 mask on the CPU is safe;
        # all following feature math immediately returns to Metal.
        return mx.array(np.argwhere(np.asarray(logits > threshold)).astype(np.int32))


__all__ = [
    "TrellisSparseStructureDecoder",
    "TrellisSparseStructureDecoderConfig",
    "TrellisSparseStructureFlowConfig",
    "TrellisSparseStructureFlowModel",
]
