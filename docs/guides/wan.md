# WAN 2.1 — real text-to-video on Apple silicon

mlx-diffuser ships faithful, weight-compatible ports of the **WAN 2.1**
text-to-video model, so you can run the official checkpoint natively in MLX.
Everything — loading, conversion, text encoding, denoising, and decoding — happens
on Metal, and the 1.3B model fits in **~6 GB** of unified memory.

## Components

| Model | Role |
| --- | --- |
| `UMT5EncoderModel` | umT5-xxl text encoder (5.6B params; loadable 4-bit ≈ 3 GB) |
| `WanTransformer3DModel` | the diffusion transformer (DiT) operating on video latents |
| `AutoencoderKLWan` | causal 3D VAE, ÷8 spatial and ÷4 temporal compression |

Each is a faithful port of the diffusers/transformers reference and is verified
numerically against it (`scripts/check_wan_vae.py`, `scripts/check_wan_transformer.py`).

## One-line generation

```python
from mlx_diffuser import WanPipeline

pipe = WanPipeline.from_diffusers("checkpoints/wan2.1-t2v-1.3b")  # converts on load
video = pipe(
    "a red fox trotting through snow, cinematic",
    num_frames=17, height=256, width=256, num_inference_steps=30,
)                                              # (1, T, H, W, 3) in [-1, 1]
```

`from_diffusers` quantizes the text encoder to 4-bit and keeps the transformer in
bf16 by default; pass `quantize_transformer=4` to shrink it further. `num_frames`
must be `1 + 4k` (the VAE's temporal stride) and `height`/`width` multiples of 8.

The full runnable script is [`examples/wan_text_to_video.py`](https://github.com/AmirHossein-razlighi/mlx_diffuser/blob/main/examples/wan_text_to_video.py),
which also handles the one-time checkpoint download.

## The checkpoint converter

Loading official weights is the job of `mlx_diffuser.converters`. A `Converter`
maps one diffusers component folder onto the matching MLX model:

```python
from mlx_diffuser.converters import get_converter

vae = get_converter("AutoencoderKLWan").convert("checkpoints/wan2.1-t2v-1.3b/vae")
dit = get_converter("WanTransformer3DModel").convert(
    "…/transformer", dtype=mx.bfloat16
)
```

It reads safetensors natively (no PyTorch), reorders conv kernels to channels-last,
and uses a **build-and-fill** check: the target model is instantiated and every one
of its parameters must be covered with a matching shape, so an architecture
mismatch fails loudly instead of silently decoding to noise. `convert(...,
quantize=4)` weight-quantizes during load, and because safetensors are memory-mapped
lazily, quantizing the multi-GB encoder never holds it all in RAM at once.

## Memory notes

- umT5 4-bit ≈ 3 GB, DiT bf16 ≈ 2.8 GB, VAE ≈ 0.25 GB → fits comfortably on a
  16 GB Mac. The pipeline can also release the text encoder after encoding
  (`release_text_encoder=True`, the default) to free memory before denoising.
- Smaller `--size` / `--frames` / `--steps` cut runtime and memory substantially.
