# LTX-2.3 — text-to-video on Apple silicon

mlx-diffuser ships faithful, weight-compatible ports of **LTX-2.3**, Lightricks'
22B-parameter joint audio-video foundation model, so you can run the official
distilled checkpoint natively in MLX — video **with its generated soundtrack**.
Conversion, text encoding, denoising, and decoding all happen on Metal.

<p align="center">
<video src="../../assets/ltx2_sample.mp4" width="576" controls loop playsinline></video>
</p>
<p align="center"><sub><em>"a red fox trotting through fresh snow in a pine forest, low tracking shot,
golden hour, soft rim light, cinematic" — 768×512, 121 frames (5 s @ 24 fps), 8 steps, with its
48 kHz soundtrack, generated on an M1 Pro (16 GB) straight from the CLI:
<code>mlx-diffuser generate --model ltx-2.3 --prompt "..."</code> (~110 s/step)</em></sub></p>

## Components

| Model | Role |
| --- | --- |
| `Gemma3TextEncoder` | Gemma-3-12B — LTX-2 conditions on **all 49** of its hidden states |
| `LTX2TextConnectors` | per-modality projections + 8-block 1D transformers with 128 learnable registers |
| `LTX2Transformer3DModel` | the 22B DiT: 48 blocks, joint video (4096-d) + audio (2048-d) streams |
| `LTX2VideoDecoder` | the video VAE decoder (32× spatial / 8× temporal, 128 latent channels) |
| `LTX2AudioDecoder` | the audio VAE decoder: latent tokens → stereo log-mel spectrogram |
| `LTX2Vocoder` | BigVGAN-v2 generator + bandwidth extension → 48 kHz stereo waveform |

Every port is verified numerically against the reference implementations
(diffusers / transformers / ltx-core) — all **bit-exact** (cosine 1.0) on tiny
configs. LTX-2 denoises video and audio *jointly* (the streams talk through
per-block cross-attention); the audio latents decode through the audio VAE +
vocoder into a 48 kHz stereo track that the CLI muxes into the mp4. The
vocoder runs in float32 — bfloat16 accumulation across its 100+ sequential
convolutions audibly degrades the spectrum.

## The streaming converter: 94 GB in, 20 GB out, no disk spike

The official release is a single **46 GB** safetensors bundle plus a **48 GB**
fp32 Gemma-3-12B — more than this machine's disk, let alone its RAM. So the
converter never materializes the originals:

- the single file is read **remotely over HTTP range requests**, adjacent
  tensors coalesced into large fetches, and each tensor is quantized the
  moment it arrives, then flushed to sharded MLX safetensors;
- the Gemma fp32 shards are downloaded **one at a time**, converted, and
  deleted before the next one is fetched (peak extra disk ≈ one 5 GB shard);
- only the decode path is kept: the video/audio VAE *encoders* are skipped.

Re-running the conversion against an existing folder only fetches components
that are missing (checkpoints converted before v0.1.6 gain the ~320 MB audio
stack this way).

```bash
mlx-diffuser generate --model ltx-2.3 --prompt "..." --download   # one-time, ~90 GB transfer
```

| Component | Original | Converted |
| --- | --- | --- |
| Transformer (22B) | 42 GB bf16 | ~12 GB (4-bit) |
| Gemma-3-12B | 44 GB fp32 | ~6.5 GB (4-bit) |
| Text connectors (1.2B) | 2.3 GB bf16 | ~1.3 GB (8-bit) |
| VAE decoder | 0.8 GB bf16 | 0.8 GB (bf16) |
| Audio decoder + vocoder | 0.3 GB bf16 | 0.3 GB (bf16) |

## Staged generation on 16 GB

Even 4-bit, text stack + transformer together exceed 16 GB. The pipeline is
therefore *staged*: Gemma + connectors load, encode the prompt, and are freed
**before** the transformer loads; the transformer is freed before the VAE
decodes. Peak memory tracks the largest single stage (the ~12 GB transformer),
not the sum.

```python
from mlx_diffuser import LTX2Pipeline

pipe = LTX2Pipeline.from_converted("checkpoints/ltx-2.3-distilled-mlx")
video, audio = pipe(
    "a golden retriever puppy chasing autumn leaves in a sunny park",
    height=512, width=768, num_frames=121,   # ~5 s at 24 fps
)
# video: (1, 121, 512, 768, 3) in [-1, 1]
# audio: (2, samples) — 48 kHz stereo waveform in [-1, 1], same duration
```

`height`/`width` must be multiples of 32 and `num_frames` must be `1 + 8*k`
(the VAE's compression grid). The runnable script is
[`examples/ltx2_text_to_video.py`](https://github.com/AmirHossein-razlighi/mlx_diffuser/blob/main/examples/ltx2_text_to_video.py).

### The distilled schedule

`ltx-2.3-22b-distilled` is distilled to a **fixed 8-step sigma schedule at
CFG = 1** — one transformer call per step, no negative prompt needed. The
pipeline uses the published schedule verbatim. Passing `guidance_scale > 1`
enables a classifier-free pass (x0-space delta formulation, as in the
reference), which doubles the compute per step; the distilled weights don't
need it.

### What to expect from generation speed

At the default 768×512×121 the transformer sees ~6.1k video tokens, and one
denoising step is ~260 TFLOPs of compute through 21B parameters — about
**110 s/step on an M1 Pro (~12 min for 8 steps)**, which is close to that
GPU's practical ceiling. The run is *compute-bound*, not memory-bound: more
RAM would not make it faster, but a bigger GPU scales it almost linearly
(an M4 Max is roughly 4-5× the FLOPs of an M1 Pro). To trade quality for
time on the same machine, shrink the token count — `--frames 57` (~2.4 s)
or `--size 512` roughly halve the step time — or enable `--cache 0.2`
(First-Block-Cache) to skip near-identical steps.

### Prompting

LTX-2 prompts reward detail: describe the subject, the motion, the camera, and
the scene ("a red fox trotting through fresh snow, low tracking shot, golden
hour, shallow depth of field"). Prompts are encoded with Gemma-3, so natural
sentences work better than tag soups.

## CLI

```bash
mlx-diffuser generate --model ltx-2.3 \
    --prompt "a red fox trotting through fresh snow, low tracking shot, golden hour" \
    --out fox.mp4
```

Defaults: 768×512, 121 frames, 24 fps, seed 0. The generated 48 kHz stereo
soundtrack is muxed into the mp4 (a `.gif` output gets a `.wav` sidecar
instead). `.mp4` output needs `ffmpeg` on PATH (`brew install ffmpeg`). Knobs:
`--frames`, `--size`/`--height`/`--width`, `--seed`, `--fps`, `--guidance`,
and `--cache` (First-Block-Cache threshold; with only 8 distilled steps the
win is modest).
