"""Sparse structured-latent (SLAT) flow model for native MLX TRELLIS."""

from __future__ import annotations

import dataclasses
import math

import mlx.core as mx
import mlx.nn as nn

from ..configuration import Config
from ..layers.sparse import SparseConv3D, SparseTensor, sparse_downsample, sparse_upsample
from ..layers.trellis import LayerNorm32, TrellisTimestepEmbedder, absolute_position_embedding
from ..layers.trellis_sparse import (
    ModulatedSparseTransformerCrossBlock,
    SparseLinear,
)
from ..modeling import ModelMixin


@dataclasses.dataclass
class TrellisSLatFlowConfig(Config):
    resolution: int = 64
    in_channels: int = 8
    model_channels: int = 1024
    cond_channels: int = 1024
    out_channels: int = 8
    num_blocks: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    patch_size: int = 2
    num_io_res_blocks: int = 2
    io_block_channels: tuple[int, ...] = (128,)
    pe_mode: str = "ape"
    use_fp16: bool = True
    use_skip_connection: bool = True
    share_mod: bool = False
    qk_rms_norm: bool = True
    qk_rms_norm_cross: bool = False
    use_metal_kernels: bool = True


class _SparseResBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        *,
        out_channels: int | None = None,
        downsample: bool = False,
        upsample: bool = False,
        use_metal: bool = True,
    ):
        super().__init__()
        if downsample and upsample:
            raise ValueError("a sparse residual block cannot downsample and upsample")
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        self.norm1 = LayerNorm32(channels, affine=True)
        self.norm2 = LayerNorm32(self.out_channels, affine=False)
        self.conv1 = SparseConv3D(channels, self.out_channels, 3, use_metal=use_metal)
        self.conv2 = SparseConv3D(
            self.out_channels, self.out_channels, 3, use_metal=use_metal
        )
        self.conv2.conv.weight = mx.zeros_like(self.conv2.conv.weight)
        self.conv2.conv.bias = mx.zeros_like(self.conv2.conv.bias)
        self.emb_layers = [nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels)]
        self.skip_connection = (
            SparseLinear(channels, self.out_channels)
            if channels != self.out_channels
            else None
        )

    def __call__(self, x: SparseTensor, emb: mx.array) -> SparseTensor:
        scale, shift = mx.split(self.emb_layers[1](nn.silu(emb)).astype(x.dtype), 2, axis=-1)
        if self.downsample:
            x = sparse_downsample(x, 2)
        elif self.upsample:
            x = sparse_upsample(x, 2)
        h = x.replace(nn.silu(self.norm1(x.features)))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.features)) * (1 + scale) + shift
        h = self.conv2(h.replace(nn.silu(h.features)))
        skip = self.skip_connection(x) if self.skip_connection is not None else x
        return h + skip

    def cast_torso(self, dtype: mx.Dtype) -> None:
        self.conv1.cast_dtype(dtype)
        self.conv2.cast_dtype(dtype)
        layer = self.emb_layers[1]
        layer.weight = layer.weight.astype(dtype)
        layer.bias = layer.bias.astype(dtype)
        if self.skip_connection is not None:
            self.skip_connection.weight = self.skip_connection.weight.astype(dtype)
            self.skip_connection.bias = self.skip_connection.bias.astype(dtype)


class TrellisSLatFlowModel(ModelMixin[TrellisSLatFlowConfig]):
    """Generate sparse structured features on occupancy coordinates."""

    config_class = TrellisSLatFlowConfig

    def __init__(self, config: TrellisSLatFlowConfig):
        super().__init__()
        self.config = config
        if config.pe_mode != "ape":
            raise NotImplementedError("the native SLAT flow currently supports pe_mode='ape'")
        if config.patch_size < 1 or config.patch_size & (config.patch_size - 1):
            raise ValueError("patch_size must be a power of two")
        if len(config.io_block_channels) != int(math.log2(config.patch_size)):
            raise ValueError("one io_block_channels stage is required per 2x patch reduction")

        self.t_embedder = TrellisTimestepEmbedder(config.model_channels)
        if config.share_mod:
            self.adaLN_modulation = [nn.SiLU(), nn.Linear(config.model_channels, 6 * config.model_channels)]
            self.adaLN_modulation[1].weight = mx.zeros_like(self.adaLN_modulation[1].weight)
            self.adaLN_modulation[1].bias = mx.zeros_like(self.adaLN_modulation[1].bias)

        first_channels = (
            config.io_block_channels[0] if config.io_block_channels else config.model_channels
        )
        self.input_layer = SparseLinear(config.in_channels, first_channels)
        input_blocks: list[_SparseResBlock3D] = []
        next_channels = [*config.io_block_channels[1:], config.model_channels]
        for channels, following in zip(config.io_block_channels, next_channels, strict=True):
            input_blocks.extend(
                _SparseResBlock3D(
                    channels,
                    config.model_channels,
                    out_channels=channels,
                    use_metal=config.use_metal_kernels,
                )
                for _ in range(config.num_io_res_blocks - 1)
            )
            input_blocks.append(
                _SparseResBlock3D(
                    channels,
                    config.model_channels,
                    out_channels=following,
                    downsample=True,
                    use_metal=config.use_metal_kernels,
                )
            )
        self.input_blocks = input_blocks

        self.blocks = [
            ModulatedSparseTransformerCrossBlock(
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

        out_blocks: list[_SparseResBlock3D] = []
        previous_channels = [config.model_channels, *reversed(config.io_block_channels[1:])]
        for channels, previous in zip(
            reversed(config.io_block_channels), previous_channels, strict=True
        ):
            out_blocks.append(
                _SparseResBlock3D(
                    previous * 2 if config.use_skip_connection else previous,
                    config.model_channels,
                    out_channels=channels,
                    upsample=True,
                    use_metal=config.use_metal_kernels,
                )
            )
            out_blocks.extend(
                _SparseResBlock3D(
                    channels * 2 if config.use_skip_connection else channels,
                    config.model_channels,
                    out_channels=channels,
                    use_metal=config.use_metal_kernels,
                )
                for _ in range(config.num_io_res_blocks - 1)
            )
        self.out_blocks = out_blocks
        self.out_layer = SparseLinear(first_channels, config.out_channels)
        self.out_layer.weight = mx.zeros_like(self.out_layer.weight)
        self.out_layer.bias = mx.zeros_like(self.out_layer.bias)

        if config.use_fp16:
            for block in [*self.input_blocks, *self.out_blocks]:
                block.cast_torso(mx.float16)
            for transformer_block in self.blocks:
                transformer_block.cast_linears(mx.float16)
            if config.share_mod:
                layer = self.adaLN_modulation[1]
                layer.weight = layer.weight.astype(mx.float16)
                layer.bias = layer.bias.astype(mx.float16)

    def __call__(
        self,
        x: SparseTensor,
        t: mx.array,
        cond: mx.array,
        *,
        low_memory: bool = False,
    ) -> SparseTensor:
        if x.features.shape[-1] != self.config.in_channels:
            raise ValueError(
                f"SLAT features must have {self.config.in_channels} channels, got {x.features.shape}"
            )
        if x.batch_size != cond.shape[0]:
            raise ValueError("sparse input and conditioning batch sizes must match")
        h = self.input_layer(x)
        mod = self.t_embedder(t)
        if self.config.share_mod:
            mod = self.adaLN_modulation[1](nn.silu(mod))
        if self.config.use_fp16:
            h = h.astype(mx.float16)
            mod = mod.astype(mx.float16)
            cond = cond.astype(mx.float16)

        skips: list[mx.array] = []
        for block in self.input_blocks:
            h = block(h, mod)
            skips.append(h.features)
        h = h + absolute_position_embedding(
            h.coords[:, 1:].astype(mx.float32), self.config.model_channels
        ).astype(h.dtype)
        for transformer_block in self.blocks:
            h = transformer_block(h, mod, cond)
            if low_memory:
                mx.eval(h.features)

        for block, skip in zip(self.out_blocks, reversed(skips), strict=True):
            if self.config.use_skip_connection:
                h = h.replace(mx.concatenate([h.features, skip], axis=-1))
            h = block(h, mod)
        h = h.replace(
            LayerNorm32(h.features.shape[-1], affine=False, eps=1e-5)(h.features)
        )
        return self.out_layer(h.astype(x.dtype))


__all__ = ["TrellisSLatFlowConfig", "TrellisSLatFlowModel"]
