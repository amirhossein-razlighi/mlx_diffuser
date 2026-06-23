# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

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
- **Example** — `examples/text_to_video.py` generates a clip and saves an animated
  GIF, with a `--quantize {2,3,4,6,8}` low-memory path that fits large video
  models on a 16 GB Mac.

### Notes

- Video architectures are implemented from scratch; loading official pretrained
  LTX-Video / WAN weights requires a separate checkpoint converter (not included).

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
