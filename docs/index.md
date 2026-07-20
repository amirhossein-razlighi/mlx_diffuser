# mlx-diffuser

**Diffusion & flow models on Apple silicon, powered by [MLX](https://github.com/ml-explore/mlx).**

Train from scratch, fine-tune, or run inference — for image, video, 3D, and discrete
modalities — from one small, readable codebase. If you know PyTorch and
🤗 `diffusers`, you already know this library.

<table>
<tr>
<td align="center"><img src="assets/sdxl_lion.jpg" width="300" alt="SDXL lion"></td>
<td align="center"><img src="assets/flux_lion.jpg" width="300" alt="FLUX lion"></td>
</tr>
<tr>
<td align="center"><sub>SDXL, 1024×1024</sub></td>
<td align="center"><sub>FLUX.1 schnell, 4-bit</sub></td>
</tr>
</table>

<table>
<tr>
<td align="center"><img src="assets/wan_fox.gif" width="260" alt="WAN fox video"></td>
<td align="center"><img src="assets/ltx2_fox.gif" width="260" alt="LTX-2.3 fox video"></td>
</tr>
<tr>
<td align="center"><sub>WAN 2.1, text-to-video</sub></td>
<td align="center"><sub>LTX-2.3, joint video + audio</sub></td>
</tr>
</table>

## Generate or edit

```bash
# Text to image on a 16 GB Mac
mlx-diffuser generate --model flux --prompt "a red fox in snow" \
  --low-memory --out fox.png

# Image to image with SDXL
mlx-diffuser generate --model sdxl --image photo.jpg --strength 0.65 \
  --prompt "an expressive oil painting" --low-memory --out painted.png

# Image to 3D Gaussian splats with TRELLIS
mlx-diffuser generate --model trellis --image object.png --download \
  --out object.ply
```

| Model | Conditioning | Output |
| --- | --- | --- |
| SDXL | text, image + text | image |
| FLUX.1 | text | image |
| WAN 2.1 | text | video |
| LTX-2.3 | text | video + 48 kHz stereo audio |
| TRELLIS image-large | image | 3D Gaussian PLY |

<table>
<tr>
<td align="center"><img src="assets/trellis_boot_input.png" width="300" alt="TRELLIS boot input"></td>
<td align="center"><img src="assets/trellis_boot_views.png" width="500" alt="TRELLIS boot reconstruction"></td>
</tr>
<tr>
<td align="center"><sub>photorealistic input</sub></td>
<td align="center"><sub>native MLX TRELLIS output, four views</sub></td>
</tr>
</table>

The TRELLIS result above was generated from the official image-large checkpoint on a
16 GB M1 Pro in 202.36 seconds with 2.07 GB MLX peak memory. See the
[reproducible settings and limitations](guides/trellis.md#verified-sample).

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
- [TRELLIS](guides/trellis.md) — native image-to-3D and sparse Metal kernels
- [Roadmap](roadmap.md) — image control, conditioned video, and 3D
- [Guides](guides/models.md) and the [API reference](reference/models.md)
