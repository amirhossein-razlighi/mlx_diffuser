"""Fused sparse operations implemented with MLX custom Metal kernels."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

_SPARSE_CONV3D: Any = mx.fast.metal_kernel(
    name="mlx_diffuser_sparse_conv3d",
    input_names=["features", "neighbors", "weights", "bias"],
    output_names=["out"],
    source=r"""
        uint elem = thread_position_in_grid.x;
        const int in_channels = features_shape[1];
        const int kernel_volume = neighbors_shape[1];
        const int out_channels = weights_shape[2];
        const int point = elem / out_channels;
        const int out_channel = elem - point * out_channels;

        float value = float(bias[out_channel]);
        for (int kernel_index = 0; kernel_index < kernel_volume; ++kernel_index) {
            int neighbor = neighbors[point * kernel_volume + kernel_index];
            if (neighbor < 0) {
                continue;
            }
            int feature_offset = neighbor * in_channels;
            int weight_offset = kernel_index * in_channels * out_channels + out_channel;
            for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
                value += float(features[feature_offset + in_channel]) *
                         float(weights[weight_offset + in_channel * out_channels]);
            }
        }
        out[elem] = T(value);
    """,
)


def _sparse_conv3d_fallback(
    features: mx.array,
    neighbors: mx.array,
    weights: mx.array,
    bias: mx.array,
) -> mx.array:
    """Pure-MLX implementation used for training and kernel validation."""

    points, kernel_volume = neighbors.shape
    out = mx.broadcast_to(bias, (points, bias.shape[0]))
    for kernel_index in range(kernel_volume):
        index = neighbors[:, kernel_index]
        valid = index >= 0
        safe_index = mx.maximum(index, 0)
        contribution = features[safe_index] @ weights[kernel_index]
        out = out + contribution * valid[:, None]
    return out


def sparse_conv3d(
    features: mx.array,
    neighbors: mx.array,
    weights: mx.array,
    bias: mx.array,
    *,
    use_metal: bool = True,
) -> mx.array:
    """Apply a submanifold sparse Conv3D over a precomputed neighbor map.

    Args:
        features: Point features ``(N, Cin)``.
        neighbors: Input row for every ``(point, kernel_offset)``, shape ``(N, K)``;
            ``-1`` denotes an empty neighbor.
        weights: Kernel weights ``(K, Cin, Cout)``.
        bias: Output bias ``(Cout,)``.
        use_metal: Use the fused inference kernel. Set false while training to retain
            MLX autodiff through the portable fallback.
    """

    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.shape}")
    if neighbors.ndim != 2 or neighbors.shape[0] != features.shape[0]:
        raise ValueError("neighbors must have shape (N, kernel_volume)")
    if weights.ndim != 3 or weights.shape[:2] != (neighbors.shape[1], features.shape[1]):
        raise ValueError(
            "weights must have shape "
            f"({neighbors.shape[1]}, {features.shape[1]}, Cout), got {weights.shape}"
        )
    if bias.shape != (weights.shape[2],):
        raise ValueError(f"bias must have shape ({weights.shape[2]},), got {bias.shape}")
    if neighbors.dtype != mx.int32:
        neighbors = neighbors.astype(mx.int32)

    if not use_metal:
        return _sparse_conv3d_fallback(features, neighbors, weights, bias)

    size = features.shape[0] * weights.shape[2]
    if size == 0:
        return mx.zeros((features.shape[0], weights.shape[2]), dtype=features.dtype)
    return _SPARSE_CONV3D(
        inputs=[features, neighbors, weights, bias],
        template=[("T", features.dtype)],
        grid=(size, 1, 1),
        threadgroup=(min(256, size), 1, 1),
        output_shapes=[(features.shape[0], weights.shape[2])],
        output_dtypes=[features.dtype],
    )[0]


__all__ = ["sparse_conv3d"]
