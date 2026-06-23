"""Factorized 3D rotary position embedding (RoPE) for video transformers.

Video tokens carry a ``(time, height, width)`` coordinate. We give each axis its
own slice of the rotary spectrum and concatenate them, so a single rotation
encodes all three positions — the scheme used by LTX-Video and the WAN series.

The returned ``(cos, sin)`` pair is shaped ``(T*H*W, head_dim)`` and applied to
the per-head queries/keys inside :class:`~mlx_diffuser.layers.attention.Attention`
(which expects them broadcastable to ``(B, heads, T, head_dim)``).
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np


def _axis_angles(positions: np.ndarray, n_freqs: int, theta: float) -> np.ndarray:
    """``(P,) positions -> (P, n_freqs)`` angles for one spatial/temporal axis."""
    if n_freqs == 0:
        return np.zeros((positions.shape[0], 0), dtype=np.float64)
    inv_freq = 1.0 / (theta ** (np.arange(n_freqs, dtype=np.float64) / n_freqs))
    return positions[:, None] * inv_freq[None]


def _split_dims(half: int) -> tuple[int, int, int]:
    """Partition ``half`` rotary slots across (time, height, width)."""
    base = half // 3
    return half - 2 * base, base, base  # time gets the remainder


@lru_cache(maxsize=64)
def rope_3d_freqs(
    head_dim: int, frames: int, height: int, width: int, theta: float = 10000.0
) -> tuple[mx.array, mx.array]:
    """Return ``(cos, sin)`` of shape ``(frames*height*width, head_dim)``.

    Cached: the frequencies are a constant function of the grid, not a learned
    parameter, so they must not live on a module.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}.")
    half = head_dim // 2
    d_t, d_h, d_w = _split_dims(half)

    t = np.arange(frames, dtype=np.float64)
    h = np.arange(height, dtype=np.float64)
    w = np.arange(width, dtype=np.float64)
    # Token order is (t, h, w) flattened — matches the patch-embed token layout.
    grid_t, grid_h, grid_w = np.meshgrid(t, h, w, indexing="ij")
    ang_t = _axis_angles(grid_t.reshape(-1), d_t, theta)
    ang_h = _axis_angles(grid_h.reshape(-1), d_h, theta)
    ang_w = _axis_angles(grid_w.reshape(-1), d_w, theta)

    angles = np.concatenate([ang_t, ang_h, ang_w], axis=1)  # (N, half)
    emb = np.concatenate([angles, angles], axis=1)  # (N, head_dim); pairs (i, i+half)
    cos = mx.array(np.cos(emb).astype(np.float32))
    sin = mx.array(np.sin(emb).astype(np.float32))
    return cos, sin
