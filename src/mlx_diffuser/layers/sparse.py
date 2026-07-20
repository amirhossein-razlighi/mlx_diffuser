"""Backend-free sparse tensors and 3D primitives for TRELLIS on MLX.

Coordinates are integer ``(batch, x, y, z)`` rows and features are stored once per
occupied voxel. Coordinate topology is non-differentiable and cached on the CPU;
feature math remains in MLX and runs on Metal.
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..kernels import sparse_conv3d


@dataclasses.dataclass
class SparseTensor:
    """Minimal sparse tensor shared by TRELLIS flow and representation decoders."""

    features: mx.array
    coords: mx.array
    batch_size: int | None = None
    spatial_shape: tuple[int, int, int] | None = None
    scale: tuple[int, int, int] = (1, 1, 1)
    cache: dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.features.ndim < 2:
            raise ValueError("features must have shape (N, channels...)")
        if self.coords.ndim != 2 or self.coords.shape[1] != 4:
            raise ValueError(f"coords must have shape (N, 4), got {self.coords.shape}")
        if self.features.shape[0] != self.coords.shape[0]:
            raise ValueError("features and coords must contain the same number of points")
        if self.coords.dtype != mx.int32:
            self.coords = self.coords.astype(mx.int32)
        if self.batch_size is None:
            self.batch_size = (
                int(mx.max(self.coords[:, 0]).item()) + 1 if self.coords.shape[0] else 0
            )
        if self.spatial_shape is None:
            if self.coords.shape[0]:
                maximum = np.asarray(self.coords[:, 1:].max(axis=0), dtype=np.int64)
                self.spatial_shape = (
                    int(maximum[0]) + 1,
                    int(maximum[1]) + 1,
                    int(maximum[2]) + 1,
                )
            else:
                self.spatial_shape = (0, 0, 0)

    @property
    def feats(self) -> mx.array:
        """Alias used by the original TRELLIS sparse modules."""

        return self.features

    @property
    def shape(self) -> tuple[int, ...]:
        assert self.batch_size is not None
        return (self.batch_size, *self.features.shape[1:])

    @property
    def dtype(self) -> mx.Dtype:
        return self.features.dtype

    @property
    def num_points(self) -> int:
        return self.features.shape[0]

    def replace(
        self,
        features: mx.array,
        coords: mx.array | None = None,
        *,
        batch_size: int | None = None,
        spatial_shape: tuple[int, int, int] | None = None,
        scale: tuple[int, int, int] | None = None,
    ) -> SparseTensor:
        return SparseTensor(
            features,
            self.coords if coords is None else coords,
            self.batch_size if batch_size is None else batch_size,
            self.spatial_shape if spatial_shape is None else spatial_shape,
            self.scale if scale is None else scale,
            self.cache,
        )

    def batch_layout(self) -> tuple[slice, ...]:
        batch = np.asarray(self.coords[:, 0], dtype=np.int32)
        assert self.batch_size is not None
        counts = np.bincount(batch, minlength=self.batch_size)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        return tuple(
            slice(int(offsets[i]), int(offsets[i + 1])) for i in range(self.batch_size)
        )

    def _broadcast_other(self, other: Any) -> Any:
        if isinstance(other, SparseTensor):
            if other.coords.shape != self.coords.shape:
                raise ValueError("sparse tensors must have matching coordinates")
            return other.features
        if isinstance(other, mx.array) and other.ndim >= 1 and other.shape[0] == self.batch_size:
            return other[self.coords[:, 0]]
        return other

    def __add__(self, other: Any) -> SparseTensor:
        return self.replace(self.features + self._broadcast_other(other))

    def __radd__(self, other: Any) -> SparseTensor:
        return self + other

    def __sub__(self, other: Any) -> SparseTensor:
        return self.replace(self.features - self._broadcast_other(other))

    def __rsub__(self, other: Any) -> SparseTensor:
        return self.replace(self._broadcast_other(other) - self.features)

    def __mul__(self, other: Any) -> SparseTensor:
        return self.replace(self.features * self._broadcast_other(other))

    def __rmul__(self, other: Any) -> SparseTensor:
        return self * other

    def __truediv__(self, other: Any) -> SparseTensor:
        return self.replace(self.features / self._broadcast_other(other))

    def astype(self, dtype: mx.Dtype) -> SparseTensor:
        return self.replace(self.features.astype(dtype))

    def dense(self, fill_value: float = 0.0) -> mx.array:
        assert self.batch_size is not None and self.spatial_shape is not None
        out = mx.full(
            (self.batch_size, *self.spatial_shape, *self.features.shape[1:]),
            fill_value,
            dtype=self.features.dtype,
        )
        index = tuple(self.coords[:, i] for i in range(4))
        return out.at[index].add(self.features)


def build_neighbor_map(
    coords: mx.array,
    kernel_size: int | tuple[int, int, int],
    dilation: int | tuple[int, int, int] = 1,
) -> mx.array:
    """Build the input-point map for submanifold cross-correlation."""

    kernel = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size
    dilations = (dilation,) * 3 if isinstance(dilation, int) else dilation
    if len(kernel) != 3 or len(dilations) != 3 or any(k % 2 == 0 for k in kernel):
        raise ValueError("sparse submanifold kernels must have three odd dimensions")

    host_coords = np.asarray(coords, dtype=np.int32)
    lookup = {tuple(int(value) for value in row): index for index, row in enumerate(host_coords)}
    offsets = [
        (
            (z - kernel[0] // 2) * dilations[0],
            (y - kernel[1] // 2) * dilations[1],
            (x - kernel[2] // 2) * dilations[2],
        )
        for z in range(kernel[0])
        for y in range(kernel[1])
        for x in range(kernel[2])
    ]
    neighbors = np.full((len(host_coords), len(offsets)), -1, dtype=np.int32)
    for point, coord in enumerate(host_coords):
        batch, z, y, x = (int(value) for value in coord)
        for kernel_index, (dz, dy, dx) in enumerate(offsets):
            neighbors[point, kernel_index] = lookup.get((batch, z + dz, y + dy, x + dx), -1)
    return mx.array(neighbors)


class _SparseConv3DCore(nn.Module):
    """Parameter holder matching spconv's ``(O, kD, kH, kW, I)`` layout."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool):
        super().__init__()
        kernel_volume = kernel_size**3
        bound = 1.0 / math.sqrt(in_channels * kernel_volume)
        self.weight = mx.random.uniform(
            low=-bound,
            high=bound,
            shape=(out_channels, kernel_size, kernel_size, kernel_size, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self._bias = mx.zeros((out_channels,))


class SparseConv3D(nn.Module):
    """Submanifold Conv3D using the fused custom Metal kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        dilation: int = 1,
        bias: bool = True,
        use_metal: bool = True,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("SparseConv3D requires an odd kernel_size")
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_metal = use_metal
        self.conv = _SparseConv3DCore(in_channels, out_channels, kernel_size, bias)

    def __call__(self, x: SparseTensor) -> SparseTensor:
        key = f"neighbors_{self.kernel_size}_{self.dilation}_{x.scale}"
        neighbors = x.cache.get(key)
        if neighbors is None:
            neighbors = build_neighbor_map(x.coords, self.kernel_size, self.dilation)
            x.cache[key] = neighbors
        # Convert spconv's checkpoint layout to the fused kernel's (K, I, O) view.
        weight = self.conv.weight.transpose(1, 2, 3, 4, 0).reshape(
            self.kernel_size**3, self.conv.weight.shape[-1], self.conv.weight.shape[0]
        )
        bias = self.conv.bias if hasattr(self.conv, "bias") else self.conv._bias
        features = sparse_conv3d(
            x.features,
            neighbors,
            weight,
            bias,
            use_metal=self.use_metal and not self.training,
        )
        return x.replace(features)

    def cast_dtype(self, dtype: mx.Dtype) -> None:
        self.conv.weight = self.conv.weight.astype(dtype)
        if hasattr(self.conv, "bias"):
            self.conv.bias = self.conv.bias.astype(dtype)
        else:
            self.conv._bias = self.conv._bias.astype(dtype)


def sparse_downsample(x: SparseTensor, factor: int = 2) -> SparseTensor:
    """Average-pool occupied coordinates and cache the inverse for upsampling."""

    if factor < 1:
        raise ValueError("factor must be positive")
    host = np.asarray(x.coords, dtype=np.int32).copy()
    host[:, 1:] //= factor
    new_coords, inverse = np.unique(host, axis=0, return_inverse=True)
    inverse_mx = mx.array(inverse.astype(np.int32))
    count = mx.zeros((len(new_coords), 1), dtype=mx.float32).at[inverse_mx].add(
        mx.ones((x.num_points, 1), dtype=mx.float32)
    )
    summed = mx.zeros((len(new_coords), x.features.shape[1]), dtype=x.features.dtype).at[
        inverse_mx
    ].add(x.features)
    # Match torch.scatter_reduce(..., reduce="mean") with its default
    # include_self=True and a zero-initialized destination.
    features = summed / (count + 1).astype(summed.dtype)
    scale = tuple(value * factor for value in x.scale)
    scale = (scale[0], scale[1], scale[2])
    source_shape = x.spatial_shape or (0, 0, 0)
    spatial_shape = (
        math.ceil(source_shape[0] / factor),
        math.ceil(source_shape[1] / factor),
        math.ceil(source_shape[2] / factor),
    )
    out = SparseTensor(
        features,
        mx.array(new_coords.astype(np.int32)),
        x.batch_size,
        spatial_shape,
        scale,
        x.cache,
    )
    out.cache[f"upsample_{factor}_{scale}"] = (x.coords, inverse_mx, x.spatial_shape, x.scale)
    return out


def sparse_upsample(x: SparseTensor, factor: int = 2) -> SparseTensor:
    """Nearest-neighbor inverse of :func:`sparse_downsample`."""

    key = f"upsample_{factor}_{x.scale}"
    cached = x.cache.get(key)
    if cached is None:
        raise ValueError("sparse_upsample must be paired with a cached sparse_downsample")
    coords, inverse, spatial_shape, scale = cached
    return SparseTensor(
        x.features[inverse], coords, x.batch_size, spatial_shape, scale, x.cache
    )


def sparse_subdivide(x: SparseTensor) -> SparseTensor:
    """Replace every occupied voxel with its eight children."""

    offsets = mx.array(
        [[0, z, y, x_] for z in range(2) for y in range(2) for x_ in range(2)],
        dtype=mx.int32,
    )
    coords = x.coords[:, None, :] * mx.array([1, 2, 2, 2], dtype=mx.int32)
    coords = (coords + offsets[None]).reshape(-1, 4)
    features = mx.broadcast_to(x.features[:, None], (x.num_points, 8, *x.features.shape[1:]))
    features = features.reshape(x.num_points * 8, *x.features.shape[1:])
    source_shape = x.spatial_shape or (0, 0, 0)
    spatial_shape = (source_shape[0] * 2, source_shape[1] * 2, source_shape[2] * 2)
    scale = (
        max(x.scale[0] // 2, 1),
        max(x.scale[1] // 2, 1),
        max(x.scale[2] // 2, 1),
    )
    # Topology-dependent neighbor/window maps from the parent have incompatible row
    # counts even when a clamped scale happens to be unchanged.
    return SparseTensor(features, coords, x.batch_size, spatial_shape, scale, {})


def _attention_groups(
    x: SparseTensor,
    window_size: int | None,
    shift_window: Sequence[int] = (0, 0, 0),
) -> list[np.ndarray]:
    cache_key = f"attention_groups_{window_size}_{tuple(shift_window)}_{x.scale}"
    if cache_key in x.cache:
        return x.cache[cache_key]
    coords = np.asarray(x.coords, dtype=np.int32)
    if window_size is None:
        keys = coords[:, :1]
    else:
        shift = np.asarray(shift_window, dtype=np.int32)
        keys = np.concatenate(
            [coords[:, :1], ((coords[:, 1:] + shift) // window_size)], axis=1
        )
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        groups[tuple(int(value) for value in key)].append(index)
    result = [np.asarray(indices, dtype=np.int32) for indices in groups.values()]
    x.cache[cache_key] = result
    return result


def sparse_self_attention(
    x: SparseTensor,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    window_size: int | None = None,
    shift_window: Sequence[int] = (0, 0, 0),
) -> mx.array:
    """Fused SDPA over each sparse batch or coordinate window.

    ``q``, ``k`` and ``v`` have shape ``(N, heads, head_dim)``. Windows with equal
    occupancy are bucketed into one MLX SDPA launch instead of padded globally.
    """

    if q.shape != k.shape or q.shape != v.shape or q.shape[0] != x.num_points:
        raise ValueError("q, k, and v must share shape (N, heads, head_dim)")
    groups = _attention_groups(x, window_size, shift_window)
    by_length: dict[int, list[np.ndarray]] = defaultdict(list)
    for group in groups:
        by_length[len(group)].append(group)

    out = mx.zeros_like(v)
    for length, bucket in by_length.items():
        indices = mx.array(np.stack(bucket).astype(np.int32))
        q_group = q[indices].transpose(0, 2, 1, 3)
        k_group = k[indices].transpose(0, 2, 1, 3)
        v_group = v[indices].transpose(0, 2, 1, 3)
        attended = mx.fast.scaled_dot_product_attention(
            q_group, k_group, v_group, scale=q.shape[-1] ** -0.5
        ).transpose(0, 2, 1, 3)
        out = out.at[indices.reshape(-1)].add(
            attended.reshape(len(bucket) * length, *attended.shape[2:])
        )
    return out


def sparse_cross_attention(
    x: SparseTensor,
    q: mx.array,
    k: mx.array,
    v: mx.array,
) -> mx.array:
    """Cross-attend sparse point queries to one dense context sequence per batch."""

    if q.ndim != 3 or q.shape[0] != x.num_points:
        raise ValueError("q must have shape (N, heads, head_dim)")
    if k.shape != v.shape or k.ndim != 4 or k.shape[0] != x.batch_size:
        raise ValueError("k and v must have shape (B, tokens, heads, head_dim)")
    if q.shape[1:] != k.shape[2:]:
        raise ValueError("query and context head dimensions must match")

    out = mx.zeros_like(q)
    for batch, layout in enumerate(x.batch_layout()):
        if layout.start == layout.stop:
            continue
        q_batch = q[layout][None].transpose(0, 2, 1, 3)
        k_batch = k[batch : batch + 1].transpose(0, 2, 1, 3)
        v_batch = v[batch : batch + 1].transpose(0, 2, 1, 3)
        attended = mx.fast.scaled_dot_product_attention(
            q_batch, k_batch, v_batch, scale=q.shape[-1] ** -0.5
        )[0].transpose(1, 0, 2)
        indices = mx.arange(layout.start, layout.stop, dtype=mx.int32)
        out = out.at[indices].add(attended)
    return out


__all__ = [
    "SparseConv3D",
    "SparseTensor",
    "build_neighbor_map",
    "sparse_downsample",
    "sparse_cross_attention",
    "sparse_self_attention",
    "sparse_subdivide",
    "sparse_upsample",
]
