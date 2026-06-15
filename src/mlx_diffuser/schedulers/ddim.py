"""DDIM scheduler: DDPM's training math with a (near-)deterministic reverse step."""

from __future__ import annotations

import dataclasses

import mlx.core as mx

from .ddpm import DDPMConfig, DDPMScheduler


@dataclasses.dataclass
class DDIMConfig(DDPMConfig):
    eta: float = 0.0  # 0 -> deterministic; 1 -> DDPM-like stochasticity


class DDIMScheduler(DDPMScheduler):
    config_class = DDIMConfig
    config: DDIMConfig

    def __init__(self, config: DDIMConfig | None = None):
        super().__init__(config or DDIMConfig())

    def step(
        self, model_output: mx.array, t: mx.array, sample: mx.array, key: mx.array | None = None
    ) -> mx.array:
        ti = int(t.item()) if isinstance(t, mx.array) else int(t)
        prev = ti - self._stride
        acp_t = self.alphas_cumprod[ti]
        acp_prev = self.alphas_cumprod[prev] if prev >= 0 else mx.array(1.0)

        x0 = self.predict_x0(model_output, mx.array(ti), sample)
        pred_eps = (sample - mx.sqrt(acp_t) * x0) / mx.sqrt(1.0 - acp_t)

        eta = self.config.eta
        sigma = (
            (eta * mx.sqrt((1.0 - acp_prev) / (1.0 - acp_t)) * mx.sqrt(1.0 - acp_t / acp_prev))
            if prev >= 0
            else mx.array(0.0)
        )

        dir_xt = mx.sqrt(mx.maximum(1.0 - acp_prev - sigma**2, 0.0)) * pred_eps
        prev_sample = mx.sqrt(acp_prev) * x0 + dir_xt

        if eta > 0 and prev >= 0:
            key = key if key is not None else mx.random.key(ti)
            prev_sample = prev_sample + sigma * mx.random.normal(sample.shape, key=key)
        return prev_sample
