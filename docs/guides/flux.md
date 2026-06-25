# FLUX.1 — text-to-image on Apple silicon

mlx-diffuser ships faithful, weight-compatible ports of **FLUX.1**, Black Forest Labs'
12B-parameter rectified-flow image model, so you can run the official checkpoint
natively in MLX. Loading, conversion, text encoding, denoising, and decoding all happen
on Metal.

<table>
<tr>
<td align="center"><img src="../../assets/flux_lion.jpg" width="360" alt="FLUX.1 lion"></td>
<td align="center"><img src="../../assets/flux_fox.jpg" width="360" alt="FLUX.1 fox"></td>
</tr>
<tr>
<td align="center"><sub><em>"a majestic lion standing on a cliff at sunset, photorealistic, cinematic"</em></sub></td>
<td align="center"><sub><em>"a red fox trotting through snow, cinematic"</em></sub></td>
</tr>
</table>

<p align="center"><sub><em>FLUX.1-schnell, 1024×1024, 4 steps, 4-bit — generated on an M1 Pro (16 GB).
The fox is straight from the CLI: <code>mlx-diffuser generate --model flux --prompt "a red fox trotting through snow, cinematic" --tile-vae</code></em></sub></p>

## Components

| Model | Role |
| --- | --- |
| `CLIPTextModel` | CLIP ViT-L/14 — provides the **pooled** text embedding for modulation |
| `T5EncoderModel` | T5-XXL — the per-token text sequence for joint attention (4096-dim) |
| `FluxTransformer2DModel` | the MMDiT: 19 double-stream + 38 single-stream blocks |
| `AutoencoderKLSD` | the 16-channel FLUX VAE (÷8 spatial, shift+scale latents) |

Every port is verified numerically against diffusers/transformers — the transformer, T5,
and VAE are all **bit-exact** (cosine 1.0).

## Memory: FLUX is big, so quantize

FLUX.1 is a 12B-parameter transformer — about **24 GB** in bf16, which does not fit on a
16 GB Mac. The pipeline therefore loads the transformer and T5 encoder **4-bit by
default**, bringing the whole thing to roughly **10 GB**:

| Component | bf16 | 4-bit (default) |
| --- | --- | --- |
| Transformer (12B) | ~23 GB | ~6.5 GB |
| T5-XXL encoder | ~9.5 GB | ~2.5 GB |
| CLIP-L | ~0.25 GB | — (bf16) |
| VAE | ~0.16 GB (fp32) | — |

Conversion is memory-safe: weights are memory-mapped and quantized in small chunks (each
bf16 tensor is freed right after its 4-bit version is computed), so the full bf16 model is
never resident — converting the 12B transformer peaks at **~6.5 GB**, not ~24 GB. Loading
the whole pipeline peaks at **~9.7 GB**, and a 1024px generation at **~14 GB** with
`tile_vae=True` — comfortably inside 16 GB, no swapping.

## One-line generation

```python
from mlx_diffuser import FluxPipeline

pipe = FluxPipeline.from_diffusers("checkpoints/flux1-schnell")  # 4-bit, converts on load
image = pipe(
    "a majestic lion standing on a cliff at sunset, photorealistic, cinematic",
    height=1024, width=1024, num_inference_steps=4,  # schnell needs only ~4 steps
)                                                    # (1, 1024, 1024, 3) in [-1, 1]
```

`from_diffusers` runs the transformer and CLIP in bf16 (4-bit weights) and the VAE in
fp32. `height`/`width` must be multiples of 16 (the 8× VAE downsample times the 2×2 patch
packing). The runnable script is
[`examples/flux_text_to_image.py`](https://github.com/AmirHossein-razlighi/mlx_diffuser/blob/main/examples/flux_text_to_image.py).

### schnell vs dev

**schnell** (Apache-2.0) is guidance-distilled: ~4 steps, no classifier-free guidance.
**dev** adds a distilled `guidance` embedding — pass `guidance_scale≈3.5`,
`num_inference_steps≈50`, and `max_sequence_length=512`:

```python
pipe = FluxPipeline.from_diffusers("checkpoints/flux1-dev")
image = pipe(prompt, num_inference_steps=50, guidance_scale=3.5, max_sequence_length=512)
```

## The checkpoint converter

`mlx_diffuser.converters` maps each diffusers/transformers component folder onto the
matching MLX model, reading safetensors natively (no PyTorch) with a build-and-fill check
so an architecture mismatch fails loudly. The transformer and T5 are pure-Linear models,
so conversion is essentially an identity remap onto the channels-last tree; only the VAE
needs conv kernels reordered.

## Going faster & lighter

**4/8-bit weights** (`quantize_transformer`, `quantize_t5`) are the main lever — 4-bit is
the default. **First-Block caching** (`cache_threshold > 0`) skips the bulk of the
transformer on steps where the first block barely changes; with schnell's 4 steps the win
is modest, but it helps the longer dev schedule:

```python
image = pipe(prompt, cache_threshold=0.1)
```

**`tile_vae=True`** decodes the final latent in feather-blended tiles. At 1024px the fp32
VAE decode is the memory high-water mark (~18 GB); tiling brings the whole generation under
**14 GB** so it fits a 16 GB Mac without swapping:

```python
image = pipe(prompt, tile_vae=True)   # recommended on 16 GB machines
```

**`release_text_encoders=True`** frees CLIP + T5 right after encoding, lowering memory
before the denoising loop (the prompt is already encoded by then):

```python
image = pipe(prompt, release_text_encoders=True)
```
