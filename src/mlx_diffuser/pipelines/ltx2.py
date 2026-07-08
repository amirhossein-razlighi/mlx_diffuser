"""LTX2Pipeline: text-to-video-with-audio with LTX-2.3, natively in MLX.

Wires the Gemma-3-12B text encoder, the LTX-2 text connectors, the 22B
audio-video diffusion transformer, and the decoders of both modalities. LTX-2
denoises video and audio latents *jointly* (the streams exchange information
through per-block cross-attention); the video latents decode through the video
VAE and the audio latents through the audio VAE decoder + BigVGAN vocoder into
a 48 kHz stereo waveform.

The distilled checkpoint needs just 8 steps at CFG=1 with a fixed sigma
schedule. To fit 16 GB of unified memory the pipeline is *staged*: the text
stack (4-bit Gemma + connectors) is loaded, used, and freed before the 4-bit
transformer loads; the transformer is freed before the decoders run. Peak
memory therefore tracks the largest single stage, not the sum.

Load from a checkpoint folder produced by
:func:`mlx_diffuser.converters.ltx2.convert_ltx2_checkpoint`.
"""

from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..caching import FirstBlockCache
from ..models.autoencoder_kl_ltx2 import LTX2VideoDecoder
from ..models.gemma3 import Gemma3TextEncoder
from ..models.ltx2_audio import LTX2AudioDecoder, LTX2Vocoder
from ..models.ltx2_connectors import LTX2TextConnectors
from ..models.ltx2_transformer import LTX2Transformer3DModel
from .wan import _DenoiseProgress

# Tuned to match the distillation process (Lightricks' ltx-pipelines constants).
DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]

MAX_PROMPT_TOKENS = 1024  # Gemma prompts are left-padded to this fixed budget

# Audio latent rate (16 kHz mel spectrogram, 4x compressed by the audio VAE).
AUDIO_LATENTS_PER_SECOND = 16000 / 160 / 4  # sampling_rate / hop_length / compression

AUDIO_SAMPLE_RATE = 48000  # the vocoder's bandwidth-extended output rate


class LTX2Pipeline:
    """LTX-2.3 text-to-video-with-audio (channels-last MLX tensors).

    Calling the pipeline returns ``(video, audio)``: frames ``(1, F, H, W, 3)``
    in [-1, 1] and a 48 kHz stereo waveform ``(2, samples)`` in [-1, 1].
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        required = (
            "transformer",
            "connectors",
            "vae_decoder",
            "audio_decoder",
            "vocoder",
            "text_encoder",
            "tokenizer",
        )
        for sub in required:
            if not (self.folder / sub).exists():
                raise FileNotFoundError(
                    f"{self.folder / sub} not found — run the LTX-2.3 conversion first "
                    "(mlx-diffuser generate --model ltx-2.3 --download; it only fetches "
                    "components that are missing)."
                )
        self.text_encoder: Gemma3TextEncoder | None = None
        self.connectors: LTX2TextConnectors | None = None
        self.transformer: LTX2Transformer3DModel | None = None
        self.vae: LTX2VideoDecoder | None = None
        self.audio_decoder: LTX2AudioDecoder | None = None
        self.vocoder: LTX2Vocoder | None = None
        self._tokenizer = None

    @classmethod
    def from_converted(cls, folder: str | Path) -> LTX2Pipeline:
        """Open a converted LTX-2.3 checkpoint folder (components load lazily)."""
        return cls(folder)

    # --- staged component management ------------------------------------------
    def _release(self, *names: str) -> None:
        for name in names:
            setattr(self, name, None)
        gc.collect()
        mx.clear_cache()

    def _load_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "The LTX-2 pipeline needs `transformers` for tokenization."
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.folder / "tokenizer"))
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    # --- text encoding ----------------------------------------------------------
    def encode_prompt(self, prompt: str) -> tuple[mx.array, mx.array]:
        """Encode ``prompt`` into per-modality text streams (video, audio).

        Loads the Gemma encoder + connectors on first use; call
        :meth:`release_text_stack` afterwards to reclaim ~8 GB.
        """
        tokenizer = self._load_tokenizer()
        if self.text_encoder is None:
            self.text_encoder = Gemma3TextEncoder.from_pretrained(self.folder / "text_encoder")
        if self.connectors is None:
            self.connectors = LTX2TextConnectors.from_pretrained(self.folder / "connectors")

        enc = tokenizer(
            prompt.strip(),
            padding="max_length",
            max_length=MAX_PROMPT_TOKENS,
            truncation=True,
            return_tensors="np",
        )
        ids = mx.array(enc["input_ids"].astype("int32"))
        mask = mx.array(enc["attention_mask"].astype("int32"))
        hidden = self.text_encoder(ids, mask)  # (1, L, D, 49)
        video_text, audio_text = self.connectors(hidden, mask)
        mx.eval(video_text, audio_text)
        return video_text, audio_text

    def release_text_stack(self) -> None:
        self._release("text_encoder", "connectors")

    # --- generation ---------------------------------------------------------------
    def __call__(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        guidance_scale: float = 1.0,
        seed: int = 0,
        cache_threshold: float = 0.0,
        release_stages: bool = True,
        progress: bool = True,
    ) -> tuple[mx.array, mx.array]:
        """Generate ``(video, audio)`` for ``prompt``.

        ``video`` is ``(1, num_frames, height, width, 3)`` in [-1, 1]; ``audio``
        is a 48 kHz stereo waveform ``(2, samples)`` in [-1, 1] covering the
        same duration. ``height``/``width`` must be multiples of 32 and
        ``num_frames`` must be ``1 + 8*k`` (the VAE's compression grid). The
        distilled model runs its fixed 8-step schedule at ``guidance_scale=1``
        (no CFG); values > 1 add a classifier-free pass against
        ``negative_prompt``.

        ``release_stages`` frees each component when the next stage begins —
        essential on 16 GB machines, only worth disabling for repeated calls on
        big-memory hosts.
        """
        if height % 32 or width % 32:
            raise ValueError("height and width must be multiples of 32.")
        if (num_frames - 1) % 8 != 0:
            raise ValueError("num_frames must be 1 + a multiple of 8 (e.g. 57, 97, 121).")

        use_cfg = guidance_scale > 1.0

        # Stage 1: text encoding (Gemma-3-12B + connectors), then free ~8 GB.
        video_text, audio_text = self.encode_prompt(prompt)
        if use_cfg:
            neg_video_text, neg_audio_text = self.encode_prompt(negative_prompt)
            video_text = mx.concatenate([video_text, neg_video_text], axis=0)
            audio_text = mx.concatenate([audio_text, neg_audio_text], axis=0)
        if release_stages:
            self.release_text_stack()

        # Stage 2: joint audio-video denoising with the 22B transformer.
        if self.transformer is None:
            self.transformer = LTX2Transformer3DModel.from_pretrained(self.folder / "transformer")
        cfg = self.transformer.config

        f = (num_frames - 1) // cfg.vae_scale_factors[0] + 1
        h = height // cfg.vae_scale_factors[1]
        w = width // cfg.vae_scale_factors[2]
        audio_frames = round(num_frames / frame_rate * AUDIO_LATENTS_PER_SECOND)

        key = mx.random.key(seed)
        k_video, k_audio = mx.random.split(key)
        latents = mx.random.normal((1, f * h * w, cfg.in_channels), key=k_video)
        audio_latents = mx.random.normal(
            (1, audio_frames, cfg.audio_in_channels),
            key=k_audio,  # 8 ch x 16 mel bins packed
        )

        video_coords, audio_coords = self.transformer.prepare_coords(
            video_text.shape[0], f, h, w, audio_frames, frame_rate
        )

        sigmas = np.array(DISTILLED_SIGMAS + [0.0], dtype=np.float32)
        cache = FirstBlockCache(cache_threshold) if cache_threshold > 0 else None
        bar = _DenoiseProgress(len(sigmas) - 1, enabled=progress)
        for i in range(len(sigmas) - 1):
            sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
            t = mx.full((video_text.shape[0],), sigma * 1000.0)
            if use_cfg:
                x_in = mx.concatenate([latents, latents], axis=0)
                a_in = mx.concatenate([audio_latents, audio_latents], axis=0)
            else:
                x_in, a_in = latents, audio_latents
            v_video, v_audio = self.transformer(
                x_in.astype(video_text.dtype),
                a_in.astype(audio_text.dtype),
                video_text,
                audio_text,
                t,
                video_coords,
                audio_coords,
                cache=cache,
            )
            v_video = v_video.astype(mx.float32)
            v_audio = v_audio.astype(mx.float32)
            if use_cfg:
                v_video = _cfg_x0(v_video, latents, sigma, guidance_scale)
                v_audio = _cfg_x0(v_audio, audio_latents, sigma, guidance_scale)
            latents = latents + (sigma_next - sigma) * v_video
            audio_latents = audio_latents + (sigma_next - sigma) * v_audio
            mx.eval(latents, audio_latents)
            bar.update(cache)
        bar.close()
        if release_stages:
            self._release("transformer")

        # Stage 3: audio decode (mel spectrogram -> 48 kHz stereo waveform).
        # Small models, but run in float32: bf16 accumulation across the
        # vocoder's 100+ sequential convolutions audibly degrades the spectrum.
        if self.audio_decoder is None:
            self.audio_decoder = LTX2AudioDecoder.from_pretrained(
                self.folder / "audio_decoder", dtype=mx.float32
            )
        if self.vocoder is None:
            self.vocoder = LTX2Vocoder.from_pretrained(self.folder / "vocoder", dtype=mx.float32)
        mel = self.audio_decoder.decode(audio_latents.astype(mx.float32))
        audio = self.vocoder(mel)[0].transpose(1, 0)  # (2, samples) at 48 kHz
        mx.eval(audio)
        if release_stages:
            self._release("audio_decoder", "vocoder")

        # Stage 4: unpack + denormalize + decode the video latents.
        if self.vae is None:
            self.vae = LTX2VideoDecoder.from_pretrained(
                self.folder / "vae_decoder", dtype=mx.bfloat16
            )
        z = latents.reshape(1, f, h, w, cfg.in_channels)
        z = self.vae.denormalize_latents(z.astype(mx.float32))
        video = self.vae.decode(z.astype(mx.bfloat16))
        video = mx.clip(video.astype(mx.float32), -1.0, 1.0)
        mx.eval(video)
        if release_stages:
            self._release("vae")
        return video, audio


def _cfg_x0(v: mx.array, sample: mx.array, sigma: float, scale: float) -> mx.array:
    """Classifier-free guidance in x0 space (LTX-2's delta formulation)."""
    cond, uncond = v[:1], v[1:]
    x0_cond = sample - sigma * cond
    x0_uncond = sample - sigma * uncond
    x0 = x0_cond + (scale - 1.0) * (x0_cond - x0_uncond)
    return (sample - x0) / sigma
