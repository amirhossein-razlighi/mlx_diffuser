"""Text-to-video generation: VideoDiT + video VAE + flow-matching scheduler.

This is the inference path used by LTX-Video / WAN-style models: sample noise in
the VAE's spatiotemporal latent space, denoise it with the transformer under
classifier-free guidance on text, then decode to pixels.

Text is supplied as **precomputed embeddings** (``prompt_embeds``). Keeping the
text encoder out of the pipeline means the library carries no heavy
tokenizer/transformer dependency — bring embeddings from any T5/umT5 encoder, or
pass your own. ``negative_embeds`` (default zeros) drives the unconditional branch.
"""

from __future__ import annotations

import mlx.core as mx

from ..models.autoencoder_kl_video import AutoencoderKLVideo
from ..models.video_dit import VideoDiT
from ..perf import compile_model
from ..schedulers import Scheduler
from .base import DiffusionPipeline, register_pipeline


@register_pipeline
class TextToVideoPipeline(DiffusionPipeline):
    _component_names = ("transformer", "vae", "scheduler")

    def __init__(self, transformer: VideoDiT, vae: AutoencoderKLVideo, scheduler: Scheduler):
        super().__init__(transformer=transformer, vae=vae, scheduler=scheduler)
        self.transformer = transformer
        self.vae = vae
        self.scheduler = scheduler

    def _latent_shape(self, batch: int, num_frames: int, height: int, width: int) -> tuple:
        c = self.vae.config
        sp, tc = c.spatial_compression, c.temporal_compression
        if num_frames % tc or height % sp or width % sp:
            raise ValueError(
                f"num_frames must be divisible by {tc} and height/width by {sp} "
                f"(got {num_frames}, {height}, {width})."
            )
        return (batch, num_frames // tc, height // sp, width // sp, self.transformer.config.in_channels)

    def __call__(
        self,
        prompt_embeds: mx.array,
        *,
        negative_embeds: mx.array | None = None,
        num_frames: int = 16,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: int | None = None,
        key: mx.array | None = None,
        compile: bool = True,
        decode: bool = True,
        progress: bool = False,
    ) -> mx.array:
        """Generate video(s) from text embeddings.

        Args:
            prompt_embeds: ``(B, L, D)`` per-token text embeddings.
            negative_embeds: ``(B, L, D)`` for the unconditional branch (CFG);
                defaults to zeros when guidance is enabled.
            num_frames/height/width: output video dimensions (in pixels/frames).
            decode: if ``False``, return raw latents instead of pixels.

        Returns:
            ``(B, T, H, W, C)`` video in roughly ``[-1, 1]``, or latents if
            ``decode=False``.
        """
        b = prompt_embeds.shape[0]
        key = self._resolve_key(seed, key)
        key, noise_key = mx.random.split(key)
        latents = mx.random.normal(self._latent_shape(b, num_frames, height, width), key=noise_key)

        self.scheduler.set_timesteps(num_inference_steps)
        use_cfg = guidance_scale and guidance_scale > 1.0
        if use_cfg and negative_embeds is None:
            negative_embeds = mx.zeros_like(prompt_embeds)

        model = compile_model(self.transformer) if compile else self.transformer

        def predict(scaled: mx.array, t: mx.array) -> mx.array:
            tb = mx.ones((b,)) * t
            if use_cfg:
                x2 = mx.concatenate([scaled, scaled], axis=0)
                t2 = mx.concatenate([tb, tb], axis=0)
                ctx = mx.concatenate([prompt_embeds, negative_embeds], axis=0)
                out = model(x2, t2, ctx)
                cond, uncond = out[:b], out[b:]
                return self.classifier_free_guidance(cond, uncond, guidance_scale)
            return model(scaled, tb, prompt_embeds)

        latents = self.denoising_loop(self.scheduler, latents, predict, key, progress=progress)
        if not decode:
            return latents
        video = self.vae.decode(latents / self.vae.scaling_factor)
        mx.eval(video)
        return video
