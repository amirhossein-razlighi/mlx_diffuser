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

## Going faster

The denoising loop runs the two classifier-free-guidance passes as a **single
batched forward**, so there's only one transformer call per step. At 1.3B the model
is compute-bound (attention + matmuls), so `mx.compile` and batching shave only a
few percent — the real levers are caching and quantization.

**First-Block Cache** (`cache_threshold`) exploits the fact that adjacent denoising
steps produce nearly the same transformer output: it computes only the first block,
and when that block's contribution has barely changed it reuses the cached residual
of the other ~29 blocks. Measured on the 1.3B model at 256px / 25 steps:

| `cache_threshold` | speedup | quality |
| --- | --- | --- |
| `0.0` (default) | 1.0× | exact |
| `0.1` | ~1.5× | no visible change |
| `0.2` | ~2.2× | no visible change (different sample) |
| `≥ 0.3` | 3×+ | degrades — avoid |

```python
video = pipe(prompt, num_frames=17, height=256, width=256,
             num_inference_steps=30, cache_threshold=0.2)   # ~2.2× faster
```

It perturbs the trajectory, so the sample differs from the exact run (like changing
the sampler) but stays sharp and coherent up to ~0.2; it's off by default.

**8-bit transformer** halves the DiT's weight memory (2.6 GB → 1.4 GB) at
essentially no quality cost (cosine 0.99996 vs bf16):

```python
pipe = WanPipeline.from_diffusers(folder, quantize_transformer=8)
```

Benchmark the transformer hot path yourself with
[`scripts/bench_wan.py`](https://github.com/AmirHossein-razlighi/mlx_diffuser/blob/main/scripts/bench_wan.py).
