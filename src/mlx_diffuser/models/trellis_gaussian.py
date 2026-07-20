"""TRELLIS sparse-latent Gaussian decoder and dependency-free PLY export."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..configuration import Config
from ..layers.sparse import SparseTensor
from ..layers.trellis import LayerNorm32, absolute_position_embedding
from ..layers.trellis_sparse import SparseLinear, SparseTransformerBlock
from ..modeling import ModelMixin


def _radical_inverse(base: int, value: int) -> float:
    result = 0.0
    inverse = 1.0 / base
    fraction = inverse
    while value > 0:
        result += (value % base) * fraction
        value //= base
        fraction *= inverse
    return result


def _hammersley_3d(index: int, samples: int) -> tuple[float, float, float]:
    return index / samples, _radical_inverse(2, index), _radical_inverse(3, index)


@dataclasses.dataclass
class TrellisGaussianDecoderConfig(Config):
    resolution: int = 64
    model_channels: int = 768
    latent_channels: int = 8
    num_blocks: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    attn_mode: str = "swin"
    window_size: int = 8
    pe_mode: str = "ape"
    use_fp16: bool = True
    qk_rms_norm: bool = False
    num_gaussians: int = 32
    perturb_offset: bool = True
    voxel_size: float = 1.5
    filter_kernel_size_3d: float = 9e-4
    scaling_bias: float = 4e-3
    opacity_bias: float = 0.1
    scaling_activation: str = "softplus"
    xyz_lr: float = 1.0
    features_lr: float = 1.0
    opacity_lr: float = 1.0
    scaling_lr: float = 1.0
    rotation_lr: float = 0.1

    @property
    def out_channels(self) -> int:
        return self.num_gaussians * 14


@dataclasses.dataclass(frozen=True)
class GaussianSplat3D:
    """One TRELLIS Gaussian-splat asset, represented entirely by MLX arrays."""

    xyz_normalized: mx.array
    features_dc: mx.array
    scaling_raw: mx.array
    rotation_raw: mx.array
    opacity_raw: mx.array
    minimum_kernel_size: float = 9e-4
    scaling_bias: float = 4e-3
    opacity_bias: float = 0.1
    scaling_activation: str = "softplus"

    @property
    def xyz(self) -> mx.array:
        return self.xyz_normalized - 0.5

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        return float(value + np.log(-np.expm1(-value)))

    @property
    def scaling(self) -> mx.array:
        if self.scaling_activation == "softplus":
            scale = nn_softplus(self.scaling_raw + self._inverse_softplus(self.scaling_bias))
        elif self.scaling_activation == "exp":
            scale = mx.exp(self.scaling_raw + np.log(self.scaling_bias))
        else:
            raise ValueError(f"unknown scaling activation {self.scaling_activation!r}")
        return mx.sqrt(mx.square(scale) + self.minimum_kernel_size**2)

    @property
    def rotation(self) -> mx.array:
        rotation = self.rotation_raw + mx.array([1.0, 0.0, 0.0, 0.0])
        return rotation / mx.maximum(
            mx.sqrt(mx.sum(mx.square(rotation), axis=-1, keepdims=True)), 1e-12
        )

    @property
    def opacity(self) -> mx.array:
        bias = np.log(self.opacity_bias / (1.0 - self.opacity_bias))
        return mx.sigmoid(self.opacity_raw + bias)

    @property
    def rgb(self) -> mx.array:
        # Degree-zero spherical harmonics to display-space RGB.
        return mx.clip(0.5 + 0.28209479177387814 * self.features_dc[:, 0], 0.0, 1.0)

    def save_ply(self, path: str | Path, *, transform: bool = True) -> Path:
        """Write a standard 3DGS PLY in the official TRELLIS coordinate convention."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = np.asarray(self.xyz, dtype=np.float32)
        features = np.asarray(self.features_dc[:, 0], dtype=np.float32)
        opacity = np.asarray(
            self.opacity_raw + np.log(self.opacity_bias / (1.0 - self.opacity_bias)),
            dtype=np.float32,
        )
        scaling = np.log(np.asarray(self.scaling, dtype=np.float32))
        rotation = np.asarray(self.rotation, dtype=np.float32)
        if transform:
            xyz = xyz @ np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
                dtype=np.float32,
            )
            # Left-multiply by the +90 degree X-axis transform used by TRELLIS.
            w, x, y, z = np.moveaxis(rotation, -1, 0)
            c = np.float32(np.sqrt(0.5))
            rotation = np.stack(
                [c * (w - x), c * (w + x), c * (y - z), c * (y + z)],
                axis=-1,
            )
        normals = np.zeros_like(xyz)
        values = np.concatenate([xyz, normals, features, opacity, scaling, rotation], axis=-1)
        properties = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ]
        header = ["ply", "format ascii 1.0", f"element vertex {len(values)}"]
        header.extend(f"property float {name}" for name in properties)
        header.append("end_header")
        with path.open("w", encoding="ascii") as handle:
            handle.write("\n".join(header) + "\n")
            np.savetxt(handle, values, fmt="%.8g")
        return path


def nn_softplus(x: mx.array) -> mx.array:
    """Numerically stable softplus without coupling the representation to nn.Module."""

    return mx.maximum(x, 0) + mx.log1p(mx.exp(-mx.abs(x)))


class TrellisGaussianDecoder(ModelMixin[TrellisGaussianDecoderConfig]):
    """Decode denormalized SLAT features into renderable Gaussian splats."""

    config_class = TrellisGaussianDecoderConfig

    def __init__(self, config: TrellisGaussianDecoderConfig):
        super().__init__()
        self.config = config
        if config.attn_mode not in {"swin", "full"}:
            raise NotImplementedError("Gaussian decoder supports attn_mode='swin' or 'full'")
        if config.pe_mode != "ape":
            raise NotImplementedError("Gaussian decoder currently supports pe_mode='ape'")
        self.input_layer = SparseLinear(config.latent_channels, config.model_channels)
        self.blocks = []
        for index in range(config.num_blocks):
            shift = config.window_size // 2 * (index % 2)
            self.blocks.append(
                SparseTransformerBlock(
                    config.model_channels,
                    config.num_heads,
                    config.mlp_ratio,
                    window_size=config.window_size if config.attn_mode == "swin" else None,
                    shift_window=(shift, shift, shift),
                    qk_rms_norm=config.qk_rms_norm,
                )
            )
        self.out_layer = SparseLinear(config.model_channels, config.out_channels)
        self.out_layer.weight = mx.zeros_like(self.out_layer.weight)
        self.out_layer.bias = mx.zeros_like(self.out_layer.bias)

        perturbation = np.asarray(
            [_hammersley_3d(i, config.num_gaussians) for i in range(config.num_gaussians)],
            dtype=np.float32,
        )
        perturbation = np.arctanh((perturbation * 2.0 - 1.0) / config.voxel_size)
        # Public buffer name matches the official checkpoint.
        self.offset_perturbation = mx.array(perturbation)
        if config.use_fp16:
            for block in self.blocks:
                block.cast_linears(mx.float16)

    def _representations(self, x: SparseTensor) -> list[GaussianSplat3D]:
        cfg = self.config
        g = cfg.num_gaussians
        fields = (
            (3, cfg.xyz_lr),
            (3, cfg.features_lr),
            (3, cfg.scaling_lr),
            (4, cfg.rotation_lr),
            (1, cfg.opacity_lr),
        )
        outputs: list[GaussianSplat3D] = []
        for layout in x.batch_layout():
            features = x.features[layout]
            cursor = 0
            values: list[mx.array] = []
            for width, learning_rate in fields:
                size = g * width
                value = features[:, cursor : cursor + size].reshape(-1, g, width)
                values.append(value * learning_rate)
                cursor += size
            xyz_offset, features_dc, scaling, rotation, opacity = values
            if cfg.perturb_offset:
                xyz_offset = xyz_offset + self.offset_perturbation[None]
            xyz_offset = mx.tanh(xyz_offset) / cfg.resolution * 0.5 * cfg.voxel_size
            center = (x.coords[layout, 1:].astype(mx.float32) + 0.5) / cfg.resolution
            xyz = (center[:, None] + xyz_offset).reshape(-1, 3)
            outputs.append(
                GaussianSplat3D(
                    xyz,
                    features_dc.reshape(-1, 1, 3),
                    scaling.reshape(-1, 3),
                    rotation.reshape(-1, 4),
                    opacity.reshape(-1, 1),
                    cfg.filter_kernel_size_3d,
                    cfg.scaling_bias,
                    cfg.opacity_bias,
                    cfg.scaling_activation,
                )
            )
        return outputs

    def __call__(self, x: SparseTensor, *, low_memory: bool = False) -> list[GaussianSplat3D]:
        h = self.input_layer(x)
        h = h + absolute_position_embedding(
            x.coords[:, 1:].astype(mx.float32), self.config.model_channels
        )
        if self.config.use_fp16:
            h = h.astype(mx.float16)
        for block in self.blocks:
            h = block(h)
            if low_memory:
                mx.eval(h.features)
        h = h.astype(x.dtype)
        h = h.replace(LayerNorm32(self.config.model_channels, affine=False, eps=1e-5)(h.features))
        h = self.out_layer(h)
        return self._representations(h)


__all__ = [
    "GaussianSplat3D",
    "TrellisGaussianDecoder",
    "TrellisGaussianDecoderConfig",
]
