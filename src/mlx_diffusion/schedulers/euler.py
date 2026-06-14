"""Euler discrete scheduler (k-diffusion / Stable Diffusion style).

Trains with the inherited VP math (discrete timesteps) and samples with a
deterministic first-order ODE solver in sigma space, where
``sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod)``.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import numpy as np

from .ddpm import DDPMConfig, DDPMScheduler
from .base import expand_to


@dataclasses.dataclass
class EulerConfig(DDPMConfig):
    pass


class EulerDiscreteScheduler(DDPMScheduler):
    config_class = EulerConfig

    def __init__(self, config: EulerConfig | None = None):
        super().__init__(config or EulerConfig())
        # Full-resolution sigma table over the training schedule.
        self._train_sigmas = mx.sqrt((1.0 - self.alphas_cumprod) / self.alphas_cumprod)
        self.sigmas: mx.array | None = None

    def set_timesteps(self, num_inference_steps: int) -> None:
        T = self.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        train_sigmas = np.array(self._train_sigmas)  # ascending in index
        # Sample fractional timesteps high->low and interpolate sigmas onto them.
        ts = np.linspace(0, T - 1, num_inference_steps)[::-1].copy()
        sigmas = np.interp(ts, np.arange(T), train_sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.sigmas = mx.array(sigmas)
        self.timesteps = mx.array(ts.astype(np.float32))
        self._step_index = 0

    def scale_model_input(self, sample: mx.array, t: mx.array) -> mx.array:
        sigma = self.sigmas[self._step_index]
        return sample / mx.sqrt(sigma**2 + 1.0)

    def _pred_original(self, model_output: mx.array, sigma: mx.array, sample: mx.array) -> mx.array:
        pt = self.config.prediction_type
        if pt == "epsilon":
            return sample - sigma * model_output
        if pt == "sample":
            return model_output
        if pt == "v_prediction":
            return model_output * (-sigma / mx.sqrt(sigma**2 + 1.0)) + sample / (sigma**2 + 1.0)
        raise ValueError(f"EulerDiscreteScheduler does not support prediction_type={pt!r}.")

    def step(self, model_output: mx.array, t: mx.array, sample: mx.array, key: mx.array | None = None) -> mx.array:
        i = self._step_index
        sigma = self.sigmas[i]
        sigma_next = self.sigmas[i + 1]
        pred_original = self._pred_original(model_output, sigma, sample)
        derivative = (sample - pred_original) / sigma
        dt = sigma_next - sigma
        self._step_index += 1
        return sample + derivative * dt

    def add_noise_sigma(self, x0: mx.array, noise: mx.array, sigma: mx.array) -> mx.array:
        """VE-style corruption used by img2img-style sampling starts."""
        return x0 + expand_to(sigma, x0.ndim) * noise
