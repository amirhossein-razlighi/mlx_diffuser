# Stable Diffusion XL — text-to-image on Apple silicon

mlx-diffuser ships faithful, weight-compatible ports of **Stable Diffusion XL**, so
you can run the official checkpoint natively in MLX. Loading, conversion, text
encoding, denoising, and decoding all happen on Metal.

<p align="center">
  <img src="../../assets/sdxl_lion.jpg" width="420" alt="SDXL lion">
  <br><sub><em>"a majestic lion standing on a cliff at sunset, photorealistic, cinematic" — SDXL base, 1024×1024, generated on a Mac</em></sub>
</p>

## Components

| Model | Role |
| --- | --- |
| `CLIPTextModel` ×2 | CLIP ViT-L/14 + OpenCLIP ViT-bigG/14 text encoders (2048-dim joint context) |
| `SDXLUNet` | the cross-attention UNet operating on 4-channel latents |
| `AutoencoderKLSD` | the VAE (÷8 spatial), with optional tiled decode |

Every port is verified numerically against diffusers/transformers — the UNet, VAE,
and both text encoders are **bit-exact** (cosine 1.0), and the Euler scheduler matches
diffusers' sigmas/timesteps exactly.

## One-line generation

```python
from mlx_diffuser import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_diffusers("checkpoints/sdxl-base-1.0")  # converts on load
image = pipe(
    "a majestic lion standing on a cliff at sunset, photorealistic, cinematic",
    negative_prompt="blurry, low quality",
    height=1024, width=1024, num_inference_steps=30, guidance_scale=5.0,
)                                              # (1, 1024, 1024, 3) in [-1, 1]
```

`from_diffusers` runs the UNet and text encoders in fp16 (SDXL's native precision) and
the VAE in fp32 (SDXL's VAE overflows fp16). `height`/`width` must be multiples of 8.
The runnable script is
[`examples/sdxl_text_to_image.py`](https://github.com/AmirHossein-razlighi/mlx_diffuser/blob/main/examples/sdxl_text_to_image.py).

## The checkpoint converter

`mlx_diffuser.converters` maps each diffusers/transformers component folder onto the
matching MLX model, reading safetensors natively (no PyTorch) with a build-and-fill
check so an architecture mismatch fails loudly. The CLIP encoders convert nearly
identity; the VAE and UNet only need conv kernels reordered to channels-last.

## Going faster & smaller

The UNet is compute-bound, so the wins come from skipping work and shrinking weights:

**DeepCache** (`cache_interval`) skips the deep UNet blocks — the expensive
1280-channel / 10-transformer-layer levels — on most steps, reusing the cached
bottleneck feature and recomputing only the shallow blocks. `2` runs the full network
every other step:

```python
image = pipe(prompt, cache_interval=2)   # ~1.7× faster, no visible quality change
```

Measured on SDXL base at 1024px / 25 steps: **153 s → 90 s (1.70×)**, visually identical.

**8-bit UNet** halves the UNet's weight memory at essentially no quality cost:

```python
pipe = StableDiffusionXLPipeline.from_diffusers(folder, quantize_unet=8)
```

**VAE tiling** (`tile_vae=True`) decodes the final latent in overlapping tiles, bounding
peak memory so very large images fit on a Mac. Classifier-free guidance already runs as
a single batched forward per step.

## Memory notes

- fp16 UNet ≈ 5 GB, CLIP ≈ 1.6 GB, fp32 VAE ≈ 0.3 GB. `quantize_unet=8` brings the UNet
  to ≈ 2.5 GB.
- Lower `num_inference_steps`, enable `cache_interval=2`, and `tile_vae=True` for the
  lightest footprint.
