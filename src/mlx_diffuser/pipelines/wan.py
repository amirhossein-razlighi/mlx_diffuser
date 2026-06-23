"""WanPipeline: text-to-video generation with WAN 2.1 on Apple silicon.

Wires the umT5 text encoder, the WAN diffusion transformer, the WAN causal 3D VAE,
and a flow-matching scheduler. ``from_diffusers`` converts an official
diffusers-format checkpoint folder into MLX models (quantizing the large text
encoder so the whole stack fits in ~6 GB), so the entire path — load, convert,
encode, denoise, decode — runs natively in MLX.

Latents are denoised in the VAE's *normalized* latent space (the space the
transformer was trained in) and denormalized with the per-channel statistics
before decoding. Tensors are channels-last.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import cast

import mlx.core as mx

from ..models.autoencoder_kl_wan import AutoencoderKLWan
from ..models.umt5 import UMT5EncoderModel
from ..models.wan_transformer import WanTransformer3DModel
from ..schedulers.flow_match_euler import FlowMatchConfig, FlowMatchEulerScheduler

# WAN's text encoder is conditioned on a fixed maximum prompt length.
MAX_PROMPT_TOKENS = 512


class WanPipeline:
    """WAN 2.1 text-to-video pipeline (channels-last ``(B, T, H, W, C)``)."""

    def __init__(
        self,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        text_encoder: UMT5EncoderModel | None,
        tokenizer,
        scheduler: FlowMatchEulerScheduler,
    ):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler

    # --- loading -------------------------------------------------------------
    @classmethod
    def from_diffusers(
        cls,
        folder: str | Path,
        *,
        quantize_text: int | None = 4,
        quantize_transformer: int | None = None,
        transformer_dtype: mx.Dtype = mx.bfloat16,
        shift: float = 3.0,
    ) -> WanPipeline:
        """Load + convert an official ``Wan2.1-T2V-*-Diffusers`` folder into MLX.

        ``quantize_text`` keeps the 5.6B umT5 encoder small (4-bit ≈ 3 GB);
        ``quantize_transformer`` optionally quantizes the DiT (defaults to bf16).
        """
        from ..converters import get_converter

        folder = Path(folder)
        vae = cast(AutoencoderKLWan, get_converter("AutoencoderKLWan").convert(folder / "vae"))
        transformer = cast(
            WanTransformer3DModel,
            get_converter("WanTransformer3DModel").convert(
                folder / "transformer", dtype=transformer_dtype, quantize=quantize_transformer
            ),
        )
        text_encoder = cast(
            UMT5EncoderModel,
            get_converter("UMT5EncoderModel").convert(
                folder / "text_encoder", quantize=quantize_text
            ),
        )
        tokenizer = _load_tokenizer(folder / "tokenizer")
        scheduler = FlowMatchEulerScheduler(FlowMatchConfig(shift=shift))
        return cls(transformer, vae, text_encoder, tokenizer, scheduler)

    # --- text encoding -------------------------------------------------------
    def encode_text(self, prompt: str) -> mx.array:
        """Tokenize and encode ``prompt`` into ``(1, L, text_dim)`` embeddings."""
        if self.text_encoder is None:
            raise RuntimeError("Text encoder was released; re-load the pipeline to encode.")
        enc = self.tokenizer(
            prompt,
            max_length=MAX_PROMPT_TOKENS,
            truncation=True,
            return_tensors="np",
        )
        ids = mx.array(enc["input_ids"].astype("int32"))
        mask = mx.array(enc["attention_mask"].astype("int32"))
        embeds = self.text_encoder(ids, mask)
        mx.eval(embeds)
        return embeds

    def release_text_encoder(self) -> None:
        """Free the text encoder (call after encoding to reclaim memory)."""
        self.text_encoder = None
        gc.collect()
        mx.clear_cache()

    # --- generation ----------------------------------------------------------
    def __call__(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        num_frames: int = 17,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 40,
        guidance_scale: float = 5.0,
        seed: int = 0,
        release_text_encoder: bool = True,
        progress: bool = True,
    ) -> mx.array:
        """Generate a video ``(1, num_frames, height, width, 3)`` in ``[-1, 1]``.

        ``num_frames`` must satisfy ``(num_frames - 1) % 4 == 0`` and ``height`` /
        ``width`` must be multiples of 8 (the VAE's spatial compression).
        """
        if (num_frames - 1) % 4 != 0:
            raise ValueError("num_frames must be 1 + a multiple of 4 (e.g. 17, 33, 49, 81).")
        if height % 8 or width % 8:
            raise ValueError("height and width must be multiples of 8.")

        prompt_embeds = self.encode_text(prompt)
        use_cfg = guidance_scale and guidance_scale > 1.0
        negative_embeds = self.encode_text(negative_prompt) if use_cfg else None
        if release_text_encoder:
            self.release_text_encoder()

        latent_frames = (num_frames - 1) // 4 + 1
        lh, lw = height // 8, width // 8
        c = self.transformer.config.in_channels
        latents = mx.random.normal((1, latent_frames, lh, lw, c), key=mx.random.key(seed))

        self.scheduler.set_timesteps(num_inference_steps)
        steps = self.scheduler.timesteps
        assert steps is not None
        for i, t in enumerate(steps):
            tt = mx.array([t.item() * 1000.0])
            cond = self.transformer(latents, tt, prompt_embeds)
            if use_cfg:
                assert negative_embeds is not None
                uncond = self.transformer(latents, tt, negative_embeds)
                v = uncond + guidance_scale * (cond - uncond)
            else:
                v = cond
            latents = self.scheduler.step(v, t, latents)
            mx.eval(latents)
            if progress:
                print(f"  step {i + 1}/{len(steps)}", end="\r")

        z = self.vae.denormalize_latents(latents).astype(mx.float32)
        video = self.vae.decode(z)
        mx.eval(video)
        return video


def _load_tokenizer(folder: Path):
    """Load the WAN (T5) tokenizer via transformers (lazy optional dependency)."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The WAN pipeline needs `transformers` for tokenization. "
            "Install it with `pip install transformers`."
        ) from exc
    return AutoTokenizer.from_pretrained(str(folder))
