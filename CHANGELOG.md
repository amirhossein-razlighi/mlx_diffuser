# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.4] — 2026-06-25

### Added

- **FLUX.1 (real checkpoints)** — faithful, weight-compatible MLX ports that load the
  official `FLUX.1-schnell` / `FLUX.1-dev` weights:
  - `FluxTransformer2DModel` — the 12B-parameter MMDiT (19 double-stream joint-attention
    blocks + 38 single-stream blocks, 3-axis RoPE, adaLN-Zero, qk-RMSNorm) and
    `T5EncoderModel` (the T5-XXL text encoder). Each is verified **bit-exact** vs
    diffusers/transformers (cosine 1.0); the 16-channel FLUX VAE reuses `AutoencoderKLSD`
    (now with optional `shift_factor` and quant-conv-free loading).
  - `FluxPipeline.from_diffusers` — tokenize (CLIP + T5), encode, flow-match denoise, and
    decode, all natively in MLX, with the resolution-dependent (`mu`-shifted) FLUX sigma
    schedule. schnell runs in ~4 steps; dev adds the distilled `guidance` embedding.
  - `examples/flux_text_to_image.py` (download + convert + generate a 1024px image).
- **4-bit by default for FLUX** — the transformer and T5 load weight-quantized so the
  whole 12B pipeline fits in ~10 GB of unified memory (it is ~34 GB in bf16). Conversion
  is memory-safe (lazy mmap + quantize). First-Block caching (`cache_threshold`) and
  `release_text_encoders` further cut compute and peak memory.
- **Unified `generate` CLI** — `mlx-diffuser generate --model sdxl|flux|flux-dev|wan
  --prompt "…" --out out.png` drives the real text-to-image / text-to-video pipelines by
  name (with `--download` to fetch the checkpoint, `--modality` to cross-check, and the
  usual `--steps`/`--guidance`/`--size`/`--quantize`/`--tile-vae`/`--frames` knobs). The
  legacy class-conditional path (`generate MODEL --labels …`) is unchanged.

## [0.1.3] — 2026-06-24

### Added

- **Stable Diffusion XL (real checkpoints)** — faithful, weight-compatible MLX ports
  that load the official `stable-diffusion-xl-base` weights:
  - `SDXLUNet` (cross-attention UNet with size micro-conditioning), `AutoencoderKLSD`
    (the SD/SDXL VAE, with tiled decode), and `CLIPTextModel` (both CLIP-L and bigG
    encoders). Each is verified **bit-exact** vs diffusers/transformers (cosine 1.0).
  - `StableDiffusionXLPipeline.from_diffusers` — tokenize, encode, denoise with
    classifier-free guidance + add_time_ids conditioning, and decode, all natively in
    MLX. The Euler scheduler gained `leading` timestep spacing + `init_noise_sigma`
    to match SDXL exactly.
  - `examples/sdxl_text_to_image.py` (download + convert + generate a 1024px image).
- **DeepCache** (`cache_interval`) — skips the deep UNet blocks on most steps,
  reusing the cached bottleneck feature: ~**1.70×** on SDXL at 1024px with no visible
  quality change. Plus 8-bit UNet (`quantize_unet`) and VAE tiling (`tile_vae`).

## [0.1.2] — 2026-06-23

### Added

- **Video models** — text-to-video support in the LTX-Video / WAN style:
  - `VideoDiT`, a spatiotemporal diffusion transformer (3D patch embedding,
    factorized 3D-RoPE self-attention, adaLN-Zero timestep conditioning, gated
    text cross-attention). Ships `VideoDiTConfig` presets matching the published
    architectures — `wan_t2v_1_3b()`, `wan_t2v_14b()`, and `ltx_video()`.
  - `AutoencoderKLVideo`, a causal-3D-convolution VAE that compresses video both
    spatially and temporally into latents for latent-space video diffusion.
  - `TextToVideoPipeline`, wiring the transformer, video VAE, and flow-matching
    scheduler with classifier-free guidance over precomputed text embeddings.
- **Layers** — `rope_3d_freqs` (factorized 3D rotary embeddings), `PatchEmbed3D`,
  `VideoDiTBlock`, and causal 3D VAE blocks (`CausalConv3d`, `ResnetBlock3D`,
  `Downsample3D`, `Upsample3D`); `Attention` now accepts an optional RoPE pair.
- **WAN 2.1 (real checkpoints)** — faithful MLX ports that load the official
  weights, plus a checkpoint-converter subsystem:
  - `AutoencoderKLWan` (causal-3D streaming VAE), `WanTransformer3DModel` (the DiT,
    with interleaved 3D-RoPE, qk-RMSNorm, and shared-time/per-block modulation),
    and `UMT5EncoderModel` (the umT5-xxl text encoder, loadable 4-bit).
  - `mlx_diffuser.converters`: a registry + `Converter` that turns a diffusers
    component folder into the matching MLX model, reading safetensors natively
    (no PyTorch) and validating that every parameter is covered. `convert()` can
    cast dtype and weight-quantize on the fly (memory-safe via lazy mmap).
  - `WanPipeline.from_diffusers` runs the whole text-to-video path — tokenize,
    umT5 encode, flow-matching denoise with CFG, decode — natively in MLX; the
    1.3B model fits in ~6 GB (umT5 4-bit + DiT bf16 + VAE).
- **Example** — `examples/text_to_video.py` (from-scratch arch, `--quantize`
  low-memory path) and `examples/wan_text_to_video.py` (download + convert + run
  the real WAN 2.1 weights), both saving an animated GIF.
- **WAN speed/memory** — batched classifier-free guidance (one transformer call
  per step) and `FirstBlockCache` (`cache_threshold`): reuses the cached residual of
  later transformer blocks on steps where the first block barely changes (~2.2× at
  256px with no visible quality change). 8-bit transformer weights
  (`quantize_transformer=8`) halve the DiT's memory at cosine 0.99996 vs bf16.
  `scripts/bench_wan.py` measures the hot path.

### Notes

- The generic `VideoDiT` / `AutoencoderKLVideo` are from-scratch architectures for
  training; the `WanTransformer3DModel` / `AutoencoderKLWan` / `UMT5EncoderModel`
  ports are weight-compatible with the official WAN 2.1 release and verified
  numerically against the reference (`scripts/check_wan_*.py`).

## [0.1.1] — 2026-06-15

### Changed

- Renamed distribution and Python package from `mlx-diffusion`/`mlx_diffusion` to
  `mlx-diffuser`/`mlx_diffuser` to match the PyPI project name.
- Added README badges (PyPI version, downloads, Python versions, license, docs).

## [0.1.0] — 2026-06-15

First public release.

### Added

- **Core** — dataclass `Config` (JSON round-trip), `ModelMixin`
  (`from_pretrained`/`save_pretrained` over safetensors, load-time dtype casting
  and weight quantization), lazy Hugging Face Hub integration.
- **Models** — `DiT` (config-driven diffusion transformer, class-conditional with
  CFG), `UNet2D` (Stable-Diffusion-style, optional cross-attention), and
  `AutoencoderKL` (VAE for latent diffusion).
- **Schedulers** — unified train+sample interface across `DDPM`, `DDIM`,
  `EulerDiscrete`, and `FlowMatchEuler` (rectified flow).
- **Training** — `DiffusionTrainer` with a compiled step, EMA, gradient clipping,
  min-SNR loss weighting, and classifier-free-guidance dropout.
- **LoRA** — `inject_lora` / `merge_lora` / `save_lora` / `load_lora` for
  parameter-efficient fine-tuning.
- **Performance** — `mx.compile` helpers, weight quantization, fused attention,
  and unified-memory controls.
- **CLI** — `mlx-diffuser generate | train | convert`.
- **Tooling** — ruff, mypy, pytest (61 tests), GitHub Actions CI on Apple-silicon
  runners, and a mkdocs-material documentation site.

[0.1.0]: https://github.com/AmirHossein-razlighi/mlx_diffuser/releases/tag/v0.1.0
