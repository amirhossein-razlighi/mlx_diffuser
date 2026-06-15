"""Diffusion training losses and loss weightings."""

from __future__ import annotations

import mlx.core as mx

from ..schedulers.base import expand_to
from ..schedulers.ddpm import DDPMScheduler


def mse_loss(pred: mx.array, target: mx.array, weights: mx.array | None = None) -> mx.array:
    """Mean squared error, optionally weighted per-sample."""
    se = (pred - target) ** 2
    if weights is not None:
        se = expand_to(weights, se.ndim) * se
    return mx.mean(se)


def min_snr_weights(scheduler: DDPMScheduler, t: mx.array, gamma: float = 5.0) -> mx.array:
    """Min-SNR-gamma loss weighting (Hang et al., 2023) for VP diffusion.

    Returns per-sample weights ``min(SNR, gamma) / SNR`` for epsilon prediction
    (``/(SNR+1)`` adjustment is applied for v-prediction).
    """
    acp = scheduler.alphas_cumprod[t]
    snr = acp / (1.0 - acp)
    weights = mx.minimum(snr, gamma) / snr
    if scheduler.config.prediction_type == "v_prediction":
        weights = mx.minimum(snr, gamma) / (snr + 1.0)
    return weights
