# Image, video, and 3D roadmap

The target is not “every checkpoint.” It is a small set of useful, weight-compatible
pipelines that run natively in MLX and have a credible path on a 16 GB Apple-silicon
machine. Every new family must preserve the existing process / network / pipeline split.

## Current state

| Area | Shipped |
| --- | --- |
| Image | SDXL text-to-image + image-to-image; FLUX.1 schnell/dev text-to-image |
| Video | WAN 2.1 1.3B text-to-video; LTX-2.3 joint text-to-video-and-audio |
| Training | from-scratch DiT, EMA, LoRA, DDPM and flow-matching objectives |
| Efficiency | 4/8-bit weights, fused attention, compiled steps, DeepCache/FBCache, tiled VAE decode, staged model release |
| 3D | experimental native TRELLIS single-image to 3D Gaussian PLY |

## Next milestones

1. **Controlled images.** Add SDXL inpainting first, then ControlNet and IP-Adapter.
   These reuse the existing SDXL UNet/VAE and the new shared image preparation path.
2. **Conditioned video.** Target WAN VACE 1.3B for reference/video editing on lower-memory
   Macs, then WAN I2V 14B as a staged 4-bit quality tier. The official WAN family ships
   dedicated I2V/VACE weights, so this needs real converter and model support rather
   than injecting an image into the existing T2V checkpoint.
3. **Complete TRELLIS representations.** The native occupancy flow, sparse SLAT flow,
   custom Metal sparse Conv3D, and Gaussian decoder are in place. Next port the mesh
   decoder/FlexiCubes and radiance-field decoder, then add a native Gaussian renderer
   and GLB/OBJ export where the representation permits it.
4. **Text-to-3D composition.** Generate a reference image with SDXL/FLUX, release that
   pipeline, then run image-to-3D. Staging keeps only one large model resident and avoids
   requiring a second text-conditioned 3D stack on 16 GB machines.

Relevant upstream references: [WAN 2.1](https://github.com/Wan-Video/Wan2.1),
[Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2), and
[TRELLIS](https://github.com/microsoft/TRELLIS). The TRELLIS port replaces its sparse
CUDA dependency with native MLX sparse tensors and a Metal submanifold Conv3D kernel.

## Acceptance bar

Each real-model pipeline lands with a converter, tiny offline tests, one reproducible
visual result, peak-memory and latency numbers, CLI and Python examples, and a 16 GB
recipe. Memory claims must come from `memory_report()` on an Apple-silicon run; visual
claims must link the prompt, seed, resolution, step count, quantization, and cache mode.

The native TRELLIS Gaussian path now meets this first acceptance bar on a 16 GB M1 Pro;
component-level numerical parity and the remaining mesh/radiance-field outputs stay open.
