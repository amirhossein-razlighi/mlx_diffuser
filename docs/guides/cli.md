# Command-line interface

Installing the package adds an `mlx-diffuser` command.

## generate

### Text-to-image / text-to-video (real models)

Pick a model by name and give it a prompt:

```bash
# image (Stable Diffusion XL)
mlx-diffuser generate --model sdxl --prompt "a lion at sunset, cinematic" --out lion.png

# image (FLUX.1-schnell — 12B, runs 4-bit, fits 16 GB)
mlx-diffuser generate --model flux --prompt "a red fox in snow" --tile-vae --out fox.png

# video (WAN 2.1)
mlx-diffuser generate --model wan --modality video \
    --prompt "a panda surfing a wave" --frames 17 --out panda.gif

# video (LTX-2.3 — 22B, 8 distilled steps, saves an .mp4)
mlx-diffuser generate --model ltx-2.3 \
    --prompt "a red fox trotting through fresh snow, golden hour" --out fox.mp4
```

| `--model` | modality | notes |
| --- | --- | --- |
| `sdxl` | image | Stable Diffusion XL base |
| `flux` / `flux-schnell` | image | 4 steps, 4-bit by default |
| `flux-dev` | image | ~50 steps, `--guidance 3.5` |
| `wan` / `wan-1.3b` | video | saves an animated GIF |
| `ltx-2.3` / `ltx` | video | 768×512, 121 frames, 24 fps; saves `.mp4` (needs ffmpeg) |

The first run needs the checkpoint locally — add `--download` to fetch it into
`checkpoints/` (or point at one with `--checkpoint PATH`). For LTX-2.3,
`--download` *stream-converts* the ~90 GB originals into ~20 GB of 4-bit MLX
components (see the [LTX-2.3 guide](ltx2.md)). Common options:
`--steps`, `--guidance`, `--size` (or `--height`/`--width`), `--seed`, `--negative`,
`--quantize`, `--cache`, `--tile-vae`, and (video) `--frames`/`--fps`. Per-model defaults
are applied when you leave a knob unset. The output extension picks the format
(`.png` image, `.gif` or `.mp4` video).

```bash
mlx-diffuser generate --model flux --prompt "..." --download   # fetch then generate
```

### Class-conditional (a saved pipeline)

A locally trained `DiffusionPipeline` is driven by class labels instead of a prompt:

```bash
mlx-diffuser generate MODEL --labels 1,2,3 --steps 50 --guidance 4.0 \
    --size 32 --seed 0 --out samples/
```

`MODEL` is a local pipeline directory or a Hub repo id. Writes `sample_000.png`,
`sample_001.png`, ….

## train

Train from scratch or fine-tune on a folder of images:

```bash
# from scratch
mlx-diffuser train --data ./images --out my-model --steps 5000 \
    --batch 16 --size 32 --hidden 384 --depth 12 --scheduler flow --ema 0.999

# LoRA fine-tune of an existing model
mlx-diffuser train --data ./photos --base my-model --lora --lora-rank 8 \
    --out my-lora --steps 1000
```

## convert

Re-save a model with a new dtype or weight quantization:

```bash
mlx-diffuser convert my-model my-model-4bit --quantize 4
mlx-diffuser convert my-model my-model-bf16 --dtype bf16
```

Run `mlx-diffuser <command> --help` for the full option list.
