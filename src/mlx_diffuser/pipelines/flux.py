"""FluxPipeline: text-to-image with FLUX.1 on Apple silicon, in MLX.

Wires the CLIP + T5 text encoders, the FLUX MMDiT transformer, the VAE, and a
flow-matching scheduler. ``from_diffusers`` converts an official ``FLUX.1-schnell`` /
``FLUX.1-dev`` folder into MLX models, so tokenizing, encoding, denoising, and decoding
all run natively on Metal.

FLUX is a 12B-parameter model, so the transformer and T5 default to **4-bit** weights —
the whole pipeline then fits in roughly 10 GB of unified memory. schnell is
guidance-distilled (4 steps, no classifier-free guidance); dev adds a distilled
``guidance`` embedding and runs ~50 steps. Images come back as ``(B, H, W, 3)`` in
``[-1, 1]``.
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import cast

import mlx.core as mx

from ..caching import FirstBlockCache
from ..models.autoencoder_kl_sd import AutoencoderKLSD
from ..models.clip_text import CLIPTextModel
from ..models.flux_transformer import FluxTransformer2DModel
from ..models.t5 import T5EncoderModel
from ..schedulers.flow_match_euler import FlowMatchConfig, FlowMatchEulerScheduler
from .wan import _DenoiseProgress

CLIP_MAX_TOKENS = 77


class FluxPipeline:
    """FLUX.1 text-to-image pipeline (channels-last ``(B, H, W, C)``)."""

    def __init__(
        self,
        transformer: FluxTransformer2DModel,
        vae: AutoencoderKLSD,
        text_encoder: CLIPTextModel,
        text_encoder_2: T5EncoderModel,
        tokenizer,
        tokenizer_2,
        scheduler: FlowMatchEulerScheduler,
        shift_config: dict,
    ):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler
        self.shift_config = shift_config

    # --- loading -------------------------------------------------------------
    @classmethod
    def from_diffusers(
        cls,
        folder: str | Path,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        quantize_transformer: int | None = 4,
        quantize_t5: int | None = 4,
    ) -> FluxPipeline:
        """Load + convert a ``FLUX.1-schnell`` / ``FLUX.1-dev`` folder into MLX.

        The transformer and CLIP run in ``dtype`` (bf16); the VAE runs in fp32. The
        12B transformer and the T5 encoder default to 4-bit weights so the pipeline
        fits in ~10 GB; pass ``quantize_transformer=None`` / ``quantize_t5=None`` for
        full precision (needs far more memory).
        """
        from ..converters import get_converter

        folder = Path(folder)
        transformer = cast(
            FluxTransformer2DModel,
            get_converter("FluxTransformer2DModel").convert(
                folder / "transformer", dtype=dtype, quantize=quantize_transformer
            ),
        )
        vae = cast(
            AutoencoderKLSD,
            get_converter("AutoencoderKL").convert(folder / "vae", dtype=mx.float32),
        )
        te = cast(
            CLIPTextModel,
            get_converter("CLIPTextModel").convert(folder / "text_encoder", dtype=dtype),
        )
        te2 = cast(
            T5EncoderModel,
            get_converter("T5EncoderModel").convert(
                folder / "text_encoder_2", dtype=dtype, quantize=quantize_t5
            ),
        )
        tok = _load_tokenizer(folder / "tokenizer")
        tok2 = _load_tokenizer(folder / "tokenizer_2")
        shift_config = _load_scheduler_config(folder / "scheduler")
        scheduler = FlowMatchEulerScheduler(FlowMatchConfig(prediction_type="velocity"))
        return cls(transformer, vae, te, te2, tok, tok2, scheduler, shift_config)

    def release_text_encoders(self) -> None:
        """Free the CLIP + T5 encoders after encoding (they are ~half the footprint)."""
        self.text_encoder = None  # type: ignore[assignment]
        self.text_encoder_2 = None  # type: ignore[assignment]
        gc.collect()
        mx.clear_cache()

    # --- text encoding -------------------------------------------------------
    def encode_prompt(self, prompt: str, max_sequence_length: int) -> tuple[mx.array, mx.array]:
        """Encode ``prompt`` with CLIP (pooled) and T5 (sequence).

        Returns ``(t5_embeds (1, L, 4096), pooled (1, 768))``: FLUX conditions joint
        attention on the T5 token sequence and modulation on CLIP's pooled embedding.
        """
        clip_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=CLIP_MAX_TOKENS,
            truncation=True,
            return_tensors="np",
        )["input_ids"].astype("int32")
        _, pooled = self.text_encoder(mx.array(clip_ids))

        t5_ids = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="np",
        )["input_ids"].astype("int32")
        t5_embeds = self.text_encoder_2(mx.array(t5_ids))
        return t5_embeds, pooled

    # --- generation ----------------------------------------------------------
    def __call__(
        self,
        prompt: str,
        *,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        max_sequence_length: int = 256,
        seed: int = 0,
        cache_threshold: float = 0.0,
        release_text_encoders: bool = False,
        tile_vae: bool = False,
        progress: bool = True,
    ) -> mx.array:
        """Generate an image ``(1, height, width, 3)`` in ``[-1, 1]``.

        ``height`` / ``width`` must be multiples of 16 (8× VAE × 2× patch packing).
        For schnell use the defaults (4 steps, ``guidance_scale=0``); for dev use ~50
        steps and ``guidance_scale≈3.5``. ``cache_threshold`` (>0) enables First-Block
        caching for a speed-up. ``release_text_encoders`` frees CLIP + T5 right after
        encoding to lower peak memory before the denoising loop. ``tile_vae`` decodes the
        final latent in tiles: at 1024px the fp32 VAE decode is the memory high-water mark
        (~18 GB), and tiling brings it under 16 GB so it fits a 16 GB Mac without swapping.
        """
        if height % 16 or width % 16:
            raise ValueError("height and width must be multiples of 16.")

        t5_embeds, pooled = self.encode_prompt(prompt, max_sequence_length)
        if release_text_encoders:
            self.release_text_encoders()

        lh, lw = height // 8, width // 8  # VAE latent grid
        ph, pw = lh // 2, lw // 2  # packed token grid
        c = self.vae.config.latent_channels
        latents = mx.random.normal((1, lh, lw, c), key=mx.random.key(seed))
        latents = _pack_latents(latents)  # (1, ph*pw, c*4)

        img_ids = _prepare_image_ids(ph, pw)
        txt_ids = mx.zeros((t5_embeds.shape[1], 3))

        sigmas = self._sigma_schedule(num_inference_steps, ph * pw)
        self.scheduler.set_sigmas(sigmas)
        steps = self.scheduler.timesteps
        assert steps is not None

        guidance = None
        if self.transformer.config.guidance_embeds:
            guidance = mx.full((1,), guidance_scale, dtype=mx.float32)

        cache = FirstBlockCache(cache_threshold) if cache_threshold > 0 else None
        bar = _DenoiseProgress(len(steps), enabled=progress)
        for t in steps:
            v = self.transformer(
                latents,
                mx.broadcast_to(t, (1,)),
                t5_embeds,
                pooled,
                img_ids,
                txt_ids,
                guidance=guidance,
                cache=cache,
            )
            latents = self.scheduler.step(v, t, latents)
            mx.eval(latents)
            bar.update(cache)
        bar.close()

        latents = _unpack_latents(latents, ph, pw, c)  # (1, lh, lw, c)
        z = latents.astype(mx.float32) / self.vae.scaling_factor + self.vae.shift_factor
        image = self.vae.decode(z, tile=tile_vae, tile_latent=48, overlap_latent=8)
        mx.eval(image)
        return image

    def _sigma_schedule(self, num_steps: int, image_seq_len: int) -> mx.array:
        """FLUX sigma schedule: ``linspace(1, 1/N, N)`` with resolution-dependent shift."""
        sc = self.shift_config
        sigmas = mx.linspace(1.0, 1.0 / num_steps, num_steps)
        if sc.get("use_dynamic_shifting", False):
            m = (sc["max_shift"] - sc["base_shift"]) / (
                sc["max_image_seq_len"] - sc["base_image_seq_len"]
            )
            mu = sc["base_shift"] + m * (image_seq_len - sc["base_image_seq_len"])
            shift = math.exp(mu)
            sigmas = shift / (shift + (1.0 / sigmas - 1.0))  # exponential time shift
        else:
            s = sc.get("shift", 1.0)
            if s != 1.0:
                sigmas = s * sigmas / (1.0 + (s - 1.0) * sigmas)
        return sigmas


# --- latent packing (channels-last) -----------------------------------------


def _pack_latents(latents: mx.array) -> mx.array:
    """``(B, H, W, C)`` latents -> ``(B, (H/2)(W/2), C*4)`` 2x2-patch tokens (feature [C, ph, pw])."""
    b, h, w, c = latents.shape
    x = latents.reshape(b, h // 2, 2, w // 2, 2, c)
    x = x.transpose(0, 1, 3, 5, 2, 4)  # (B, H/2, W/2, C, 2, 2)
    return x.reshape(b, (h // 2) * (w // 2), c * 4)


def _unpack_latents(latents: mx.array, ph: int, pw: int, c: int) -> mx.array:
    """Inverse of :func:`_pack_latents` -> ``(B, 2*ph, 2*pw, C)``."""
    b = latents.shape[0]
    x = latents.reshape(b, ph, pw, c, 2, 2)
    x = x.transpose(0, 1, 4, 2, 5, 3)  # (B, ph, 2, pw, 2, C)
    return x.reshape(b, ph * 2, pw * 2, c)


def _prepare_image_ids(ph: int, pw: int) -> mx.array:
    """3-axis position ids for the packed token grid: ``[0, row, col]``, shape ``(ph*pw, 3)``."""
    ids = mx.zeros((ph, pw, 3))
    rows = mx.broadcast_to(mx.arange(ph)[:, None], (ph, pw))
    cols = mx.broadcast_to(mx.arange(pw)[None, :], (ph, pw))
    ids[..., 1] = rows
    ids[..., 2] = cols
    return ids.reshape(ph * pw, 3)


def _load_scheduler_config(folder: Path) -> dict:
    """Read the flow-match scheduler's shift settings (falls back to schnell defaults)."""
    defaults = {
        "use_dynamic_shifting": False,
        "shift": 1.0,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096,
    }
    cfg = folder / "scheduler_config.json"
    if cfg.exists():
        defaults.update(json.loads(cfg.read_text()))
    return defaults


def _load_tokenizer(folder: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The FLUX pipeline needs `transformers` for tokenization. "
            "Install it with `pip install transformers`."
        ) from exc
    return AutoTokenizer.from_pretrained(str(folder))
