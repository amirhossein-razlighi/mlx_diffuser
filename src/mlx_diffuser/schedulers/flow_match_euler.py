"""Flow-matching / rectified-flow Euler scheduler (SD3, FLUX style).

Continuous time ``sigma in [0, 1]`` with the linear interpolation path
``x_sigma = (1 - sigma) * x0 + sigma * noise``. The network predicts the constant
velocity ``noise - x0``; sampling integrates ``dx/dsigma = velocity`` with Euler.

This scheduler's timestep convention is the float ``sigma`` itself, so train-time
and inference-time conditioning match exactly.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx

from .base import PredictionType, Scheduler, SchedulerConfig, expand_to


@dataclasses.dataclass
class FlowMatchConfig(SchedulerConfig):
    prediction_type: PredictionType = "velocity"
    shift: float = 1.0  # >1 spends more steps at high noise (resolution-dependent)


class FlowMatchEulerScheduler(Scheduler):
    config_class = FlowMatchConfig
    config: FlowMatchConfig

    def __init__(self, config: FlowMatchConfig | None = None):
        super().__init__(config or FlowMatchConfig())
        self.sigmas: mx.array | None = None

    def _shift(self, sigma: mx.array) -> mx.array:
        s = self.config.shift
        if s == 1.0:
            return sigma
        return s * sigma / (1.0 + (s - 1.0) * sigma)

    # --- training ---------------------------------------------------------
    def sample_timesteps(self, batch_size: int, key: mx.array) -> mx.array:
        u = mx.random.uniform(shape=(batch_size,), key=key)
        return self._shift(u)

    def add_noise(self, x0: mx.array, noise: mx.array, t: mx.array) -> mx.array:
        sigma = expand_to(t, x0.ndim)
        return (1.0 - sigma) * x0 + sigma * noise

    def get_target(self, x0: mx.array, noise: mx.array, t: mx.array) -> mx.array:
        if self.config.prediction_type != "velocity":
            raise ValueError("FlowMatchEulerScheduler only supports prediction_type='velocity'.")
        return noise - x0

    # --- sampling ---------------------------------------------------------
    def set_timesteps(self, num_inference_steps: int) -> None:
        self.num_inference_steps = num_inference_steps
        sigmas = mx.linspace(1.0, 0.0, num_inference_steps + 1)
        sigmas = self._shift(sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas[:-1]
        self._step_index = 0

    def step(
        self, model_output: mx.array, t: mx.array, sample: mx.array, key: mx.array | None = None
    ) -> mx.array:
        assert self.sigmas is not None, "call set_timesteps() before sampling"
        i = self._step_index
        sigma = self.sigmas[i]
        sigma_next = self.sigmas[i + 1]
        self._step_index += 1
        return sample + (sigma_next - sigma) * model_output
