"""Class-conditional image generation with a DiT + any scheduler.

Demonstrates the full inference path end-to-end (noise -> denoise -> sample),
including classifier-free guidance. Works in pixel space directly; pair with a VAE
for latent diffusion.
"""

from __future__ import annotations

import mlx.core as mx

from ..models.dit import DiT
from ..schedulers import Scheduler
from .base import DiffusionPipeline, register_pipeline


@register_pipeline
class ClassConditionalPipeline(DiffusionPipeline):
    _component_names = ("model", "scheduler")

    def __init__(self, model: DiT, scheduler: Scheduler):
        super().__init__(model=model, scheduler=scheduler)
        self.model = model
        self.scheduler = scheduler

    def __call__(
        self,
        class_labels: list[int] | mx.array,
        *,
        sample_size: int = 32,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int | None = None,
        key: mx.array | None = None,
        progress: bool = False,
    ) -> mx.array:
        labels = mx.array(class_labels) if not isinstance(class_labels, mx.array) else class_labels
        b = labels.shape[0]
        c = self.model.config.in_channels
        key = self._resolve_key(seed, key)

        key, noise_key = mx.random.split(key)
        latents = mx.random.normal((b, sample_size, sample_size, c), key=noise_key)

        self.scheduler.set_timesteps(num_inference_steps)
        use_cfg = guidance_scale and guidance_scale > 1.0
        null = mx.full((b,), self.model.config.num_classes)

        def predict(scaled: mx.array, t: mx.array) -> mx.array:
            tb = mx.ones((b,)) * t
            if use_cfg:
                model_in = mx.concatenate([scaled, scaled], axis=0)
                y = mx.concatenate([labels, null], axis=0)
                tt = mx.concatenate([tb, tb], axis=0)
                out = self.model(model_in, tt, y)
                cond, uncond = out[:b], out[b:]
                return self.classifier_free_guidance(cond, uncond, guidance_scale)
            return self.model(scaled, tb, labels)

        latents = self.denoising_loop(self.scheduler, latents, predict, key, progress=progress)
        return latents
