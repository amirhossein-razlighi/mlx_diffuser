"""Embeddings: timestep, class label, patch, and 2D positional."""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
    """Sinusoidal timestep embedding, ``(B,) -> (B, dim)``."""
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
    args = t.astype(mx.float32)[:, None] * freqs[None]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1))], axis=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding followed by a small MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        ]

    def __call__(self, t: mx.array) -> mx.array:
        x = timestep_embedding(t, self.frequency_embedding_size)
        for layer in self.mlp:
            x = layer(x)
        return x


class LabelEmbedder(nn.Module):
    """Class-label embedding with a null token for classifier-free guidance.

    The last row (index ``num_classes``) is the null/unconditional embedding.
    During training, labels are dropped to the null token with ``dropout_prob``.
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)

    @property
    def null_label(self) -> int:
        return self.num_classes

    def __call__(
        self, labels: mx.array, *, training: bool = False, key: mx.array | None = None
    ) -> mx.array:
        if training and self.dropout_prob > 0:
            key = key if key is not None else mx.random.key(0)
            drop = mx.random.uniform(shape=labels.shape, key=key) < self.dropout_prob
            labels = mx.where(drop, mx.array(self.null_label), labels)
        return self.embedding(labels)


class PatchEmbed(nn.Module):
    """Patchify a channels-last image into a token sequence via a strided conv."""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> tuple[mx.array, tuple[int, int]]:
        # x: (B, H, W, C) -> (B, H/p, W/p, hidden) -> (B, T, hidden)
        x = self.proj(x)
        b, h, w, c = x.shape
        return x.reshape(b, h * w, c), (h, w)


@lru_cache(maxsize=64)
def get_2d_sincos_pos_embed(dim: int, grid_h: int, grid_w: int) -> mx.array:
    """Fixed 2D sine-cosine positional embedding, returns ``(grid_h*grid_w, dim)``.

    Cached per (dim, grid_h, grid_w): it is a constant, not a learned parameter, so
    it must not live on the module (which would make it a tracked parameter).
    """
    assert dim % 4 == 0, "pos-embed dim must be divisible by 4"
    half = dim // 2

    def axis_embed(pos: np.ndarray) -> np.ndarray:
        omega = np.arange(half // 2, dtype=np.float64) / (half // 2)
        omega = 1.0 / (10000**omega)
        out = pos[:, None] * omega[None]
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    gh = np.arange(grid_h, dtype=np.float32)
    gw = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(gw, gh)  # x, then y
    emb_w = axis_embed(grid[0].reshape(-1))
    emb_h = axis_embed(grid[1].reshape(-1))
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return mx.array(emb.astype(np.float32))
