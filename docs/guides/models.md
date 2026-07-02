# Models

All models are config-driven `mlx.nn.Module`s with `from_pretrained` /
`save_pretrained` from `ModelMixin`. Inputs/outputs are channels-last
`(B, H, W, C)`.

## DiT — Diffusion Transformer

The general-purpose backbone: patchify → adaLN-Zero transformer → unpatchify.
Pairs naturally with flow-matching; supports optional class conditioning with
classifier-free-guidance dropout.

```python
from mlx_diffuser import DiT, DiTConfig

model = DiT(DiTConfig(
    in_channels=3, patch_size=2, hidden_size=384, depth=12, num_heads=6,
    num_classes=1000,        # 0 => unconditional
))
out = model(x, t, y)          # (B, H, W, C)
```

adaLN-Zero means an untrained DiT outputs zeros (identity residual path), which
stabilizes training.

## UNet2D

A Stable-Diffusion-style convolutional denoiser with down/mid/up skip
connections, per-level attention, and optional cross-attention for text
conditioning:

```python
from mlx_diffuser import UNet2D, UNet2DConfig

unet = UNet2D(UNet2DConfig(
    in_channels=4, out_channels=4,
    block_out_channels=(320, 640, 1280),
    cross_attention_dim=768,   # enables text conditioning via `context`
))
out = unet(latents, t, context=text_embeddings)
```

## AutoencoderKL (VAE)

Maps images ↔ latents for latent diffusion:

```python
from mlx_diffuser import AutoencoderKL, AutoencoderKLConfig

vae = AutoencoderKL(AutoencoderKLConfig(in_channels=3, latent_channels=4))
posterior = vae.encode(image)             # DiagonalGaussian
latents = posterior.sample() * vae.scaling_factor
recon = vae.decode(latents / vae.scaling_factor)
```

## Video models — VideoDiT & AutoencoderKLVideo

The video backbone (the architecture behind **LTX-Video** and the **WAN** series):
a video latent `(B, T, H, W, C)` is patchified over all three axes, processed by
transformer blocks combining adaLN-Zero timestep conditioning, **3D-RoPE**
self-attention, and text cross-attention, then unpatchified. `VideoDiTConfig`
ships presets that mirror the published model shapes:

```python
from mlx_diffuser import VideoDiT, VideoDiTConfig

model = VideoDiT(VideoDiTConfig.wan_t2v_1_3b())   # or .wan_t2v_14b(), .ltx_video()
out = model(latents, t, context=text_embeddings)  # (B, T, H, W, C)
```

`AutoencoderKLVideo` is the matching causal-3D-convolution VAE: it compresses a
video both spatially (`2 ** (len(block_out_channels) - 1)`) and temporally
(`temporal_compression`) into latents. *Causal* time convolutions mean a frame
only depends on itself and past frames.

```python
from mlx_diffuser import AutoencoderKLVideo, AutoencoderKLVideoConfig

vae = AutoencoderKLVideo(AutoencoderKLVideoConfig(
    in_channels=3, latent_channels=16,
    block_out_channels=(128, 256, 512),   # spatial compression 4
    temporal_compression=4,
))
latents = vae.encode(video).sample() * vae.scaling_factor
```

`TextToVideoPipeline` and `examples/text_to_video.py` show the full sampling
loop, including the `--quantize` low-memory path. The architectures are
implemented from scratch;
loading official LTX-Video / WAN weights needs a separate checkpoint converter.

## Saving & loading

```python
model.save_pretrained("my-model")                         # config.json + safetensors
model = DiT.from_pretrained("my-model", dtype="bf16", quantize=4)
model.save_pretrained("my-model", push_to_hub="me/my-model")   # needs [hub]
```

Quantization is a load-time choice and works for every model, including
`VideoDiT` — `VideoDiT.from_pretrained(path, quantize=4)` weight-quantizes the
transformer so large video models fit in 16 GB of unified memory.
