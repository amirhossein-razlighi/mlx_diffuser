"""Flow-Euler sampler used by the official TRELLIS image-to-3D checkpoints."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import mlx.core as mx


@dataclasses.dataclass(frozen=True)
class TrellisFlowSample:
    """Result of a TRELLIS flow integration."""

    samples: Any
    pred_x0: Any
    trajectory: tuple[Any, ...] | None = None


class TrellisFlowEulerSampler:
    """Reference-compatible velocity sampler with interval CFG.

    TRELLIS conditions its network on ``1000 * t`` and uses the guidance expression
    ``(1 + strength) * positive - strength * negative``.  ``keep_trajectory`` is off
    by default because retaining every 3D latent is wasteful on unified-memory Macs.
    """

    def __init__(self, sigma_min: float = 1e-5):
        if not 0.0 <= sigma_min < 1.0:
            raise ValueError("sigma_min must be in [0, 1)")
        self.sigma_min = sigma_min

    @staticmethod
    def timesteps(steps: int, rescale_t: float = 1.0) -> mx.array:
        if steps < 1:
            raise ValueError("steps must be positive")
        if rescale_t <= 0:
            raise ValueError("rescale_t must be positive")
        t = mx.linspace(1.0, 0.0, steps + 1, dtype=mx.float32)
        return rescale_t * t / (1.0 + (rescale_t - 1.0) * t)

    def velocity_to_x0(self, x_t: Any, t: float | mx.array, velocity: Any) -> Any:
        sigma = self.sigma_min + (1.0 - self.sigma_min) * t
        return (1.0 - self.sigma_min) * x_t - sigma * velocity

    @staticmethod
    def step(x_t: Any, velocity: Any, t: float, t_prev: float) -> Any:
        return x_t - (t - t_prev) * velocity

    @staticmethod
    def _expand_conditioning(cond: mx.array, batch_size: int) -> mx.array:
        if cond.shape[0] == batch_size:
            return cond
        if cond.shape[0] != 1:
            raise ValueError(
                f"conditioning batch must be 1 or {batch_size}, got {cond.shape[0]}"
            )
        return mx.broadcast_to(cond, (batch_size, *cond.shape[1:]))

    def sample(
        self,
        model: Callable[[Any, mx.array, mx.array], Any],
        noise: Any,
        cond: mx.array,
        *,
        negative_cond: mx.array | None = None,
        steps: int = 25,
        rescale_t: float = 3.0,
        cfg_strength: float = 5.0,
        cfg_interval: tuple[float, float] = (0.5, 1.0),
        keep_trajectory: bool = False,
        callback: Callable[[int, float, Any], None] | None = None,
    ) -> TrellisFlowSample:
        if not 0.0 <= cfg_interval[0] <= cfg_interval[1] <= 1.0:
            raise ValueError("cfg_interval must satisfy 0 <= start <= end <= 1")
        if negative_cond is None and cfg_strength != 0.0:
            raise ValueError("negative_cond is required when cfg_strength is non-zero")

        batch_size = noise.shape[0]
        cond = self._expand_conditioning(cond, batch_size)
        if negative_cond is not None:
            negative_cond = self._expand_conditioning(negative_cond, batch_size)

        schedule = self.timesteps(steps, rescale_t)
        sample = noise
        pred_x0 = noise
        trajectory: list[Any] | None = [] if keep_trajectory else None
        # Scalars are deliberately materialized on the CPU; the model graph remains lazy.
        times = [float(schedule[index].item()) for index in range(schedule.shape[0])]
        for index, (t, t_prev) in enumerate(zip(times[:-1], times[1:], strict=True)):
            model_t = mx.full((batch_size,), 1000.0 * t, dtype=mx.float32)
            velocity = model(sample, model_t, cond)
            if (
                negative_cond is not None
                and cfg_strength != 0.0
                and cfg_interval[0] <= t <= cfg_interval[1]
            ):
                negative = model(sample, model_t, negative_cond)
                velocity = (1.0 + cfg_strength) * velocity - cfg_strength * negative
            pred_x0 = self.velocity_to_x0(sample, t, velocity)
            sample = self.step(sample, velocity, t, t_prev)
            mx.eval(sample.features if hasattr(sample, "features") else sample)
            if trajectory is not None:
                trajectory.append(sample)
            if callback is not None:
                callback(index, t, sample)

        return TrellisFlowSample(
            samples=sample,
            pred_x0=pred_x0,
            trajectory=tuple(trajectory) if trajectory is not None else None,
        )


__all__ = ["TrellisFlowEulerSampler", "TrellisFlowSample"]
