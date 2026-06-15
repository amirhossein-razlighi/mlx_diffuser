# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] — Unreleased

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

[0.1.0]: https://github.com/AmirHossein-razlighi/mlx_diffusion/releases/tag/v0.1.0
