"""Normalization helpers, including adaLN-Zero modulation used by DiT."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    """Apply FiLM-style modulation: ``x * (1 + scale) + shift`` over tokens.

    ``x`` is ``(B, T, D)``; ``shift``/``scale`` are ``(B, D)`` and broadcast over T.
    """
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class AdaLNModulation(nn.Module):
    """Produces ``n`` modulation tensors of size ``dim`` from a conditioning vector.

    Initialized to zero (adaLN-Zero) so blocks start as identity, which stabilizes
    diffusion-transformer training.
    """

    def __init__(self, cond_dim: int, dim: int, n: int):
        super().__init__()
        self.n = n
        self.dim = dim
        self.act = nn.SiLU()
        self.linear = nn.Linear(cond_dim, n * dim)
        # adaLN-Zero: start as identity.
        self.linear.weight = mx.zeros_like(self.linear.weight)
        self.linear.bias = mx.zeros_like(self.linear.bias)

    def __call__(self, c: mx.array) -> list[mx.array]:
        out = self.linear(self.act(c))
        return mx.split(out, self.n, axis=-1)
