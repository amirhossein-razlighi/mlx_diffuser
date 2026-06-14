# mlx-diffusion

**Diffusion & flow models on Apple silicon, powered by [MLX](https://github.com/ml-explore/mlx).**

Train from scratch, fine-tune, or run inference — for image, video, and discrete
modalities — from one small, readable codebase. If you know PyTorch and
🤗 `diffusers`, you already know this library.

```python
from mlx_diffusion import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("path/or/hub-id", dtype="bf16", quantize=4)
images = pipe([1, 2, 3], num_inference_steps=50, guidance_scale=4.0, seed=0)
```

## Why MLX?

- **Unified memory** — no host↔device copies; run models larger than a discrete
  GPU's VRAM on a Mac.
- **`mx.compile` + fused kernels** — `mx.fast.scaled_dot_product_attention`, lazy
  evaluation, compiled training and sampling steps.
- **Weight quantization** — 4/8-bit so large models fit on 16–32 GB machines.
- **Low power** — fanless inference and fine-tuning, no cloud GPU rental.

## Where to next

- [Installation](installation.md)
- [Quickstart](quickstart.md) — generate, train, fine-tune
- [Concepts](concepts.md) — the process / network / pipeline model
- [Guides](guides/models.md) and the [API reference](reference/models.md)
