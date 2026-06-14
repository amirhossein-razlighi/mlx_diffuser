"""Exponential moving average of model parameters."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map


class EMA:
    """Tracks an EMA of a model's parameters for more stable sampling weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = tree_map(lambda p: mx.array(p), model.parameters())

    def update(self, model: nn.Module) -> None:
        d = self.decay
        self.shadow = tree_map(
            lambda s, p: d * s + (1.0 - d) * p, self.shadow, model.parameters()
        )
        mx.eval(self.shadow)

    def copy_to(self, model: nn.Module) -> None:
        """Overwrite the model's parameters with the EMA weights (in place)."""
        model.update(self.shadow)
