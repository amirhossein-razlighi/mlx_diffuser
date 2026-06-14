# mlx-diffusion — Architecture & Design

> Diffusion and flow/score generative models on Apple silicon, powered by MLX.
> Train from scratch, fine-tune, or run inference for image, video, and discrete
> (text) modalities — with one small, readable codebase.

## 1. Goals & non-goals

**Goals**

- **Familiar.** If you know PyTorch and 🤗 `diffusers`/`transformers`, you already
  know this. `Model.from_pretrained(...)`, `pipe(prompt)`, `nn.Module` everywhere.
- **Readable & hackable.** A researcher can read any file top-to-bottom and
  understand it. No deep abstraction towers, no metaclass magic, no registries of
  registries. Abstraction is added only when it removes real duplication.
- **Modality-agnostic.** The same core (process + network + sampler) drives image,
  video, and discrete/text diffusion. New modalities are configs + a network, not
  new frameworks.
- **Full lifecycle.** Pretraining, fine-tuning (full + LoRA), and inference share
  the same building blocks.
- **Apple-silicon-fast.** Lean on MLX: `mx.compile`, lazy evaluation, unified
  memory, `mx.fast` kernels, and weight quantization — by default, not as an
  afterthought.

**Non-goals (for v0.x)**

- Being a CUDA/PyTorch library. We target Apple silicon first.
- Re-porting every model the day it drops (that is a treadmill). We provide the
  building blocks and a curated set of reference models.
- A GUI/app. We are a library + CLI. Apps build on top.

## 2. The mental model

Every diffusion / flow model is three orthogonal pieces:

```
        data x0  ──►  [ PROCESS ]  ──►  noisy x_t , target
                          │                      │
                          ▼                      ▼
                     [ NETWORK ]  predicts  ──►  model_output
                          │
                          ▼
        x_t   ──►  [ SAMPLER (=process.step) ]  ──►  x_{t-1}  ──►  ...  ──►  x0
```

1. **Process** (`schedulers/`): the math of corruption + the prediction target.
   Owns `add_noise` (forward / training) and `step` (reverse / sampling). One
   object covers both DDPM-style diffusion and rectified-flow / flow-matching.
   This is the only place the SDE/ODE math lives.
2. **Network** (`models/`): a plain `mlx.nn.Module` that maps
   `(x_t, t, conditioning) -> prediction`. UNet, DiT, etc. Knows nothing about
   noise schedules.
3. **Pipeline** (`pipelines/`): glue for *inference*. Bundles a network, a
   process, a VAE/codec, and a conditioner (tokenizer + text encoder) behind one
   `__call__`. Pure convenience — never required to use the parts.

Training is symmetric: a `Trainer` calls `process.add_noise` + `process.get_target`,
runs the network, and applies a loss. No pipeline needed.

Keeping **process** and **network** orthogonal is the central design decision. It
is what lets the same UNet train under DDPM today and flow-matching tomorrow, and
what lets a researcher swap one axis without touching the other.

## 3. Package layout

```
src/mlx_diffusion/
  __init__.py            # curated top-level exports
  configuration.py       # Config: dataclass-ish JSON config base
  modeling.py            # ModelMixin: from_pretrained/save_pretrained (+safetensors)
  hub.py                 # lazy HF Hub download/upload helpers
  quantization.py        # quantize/dequantize, 4/8-bit weight loading
  utils.py               # dtype maps, logging, image<->array, seeding

  layers/                # small reusable nn.Modules
    attention.py         #   MultiHeadAttention via mx.fast.scaled_dot_product_attention
    embeddings.py        #   timestep, patch, sinusoidal, rotary
    normalization.py     #   RMSNorm, AdaLayerNorm (DiT-style modulation)
    blocks.py            #   ResnetBlock2D, Transformer/DiT block, up/down samplers

  models/                # config-driven networks (scale down to tiny for CI)
    unet2d.py            #   SD-style UNet
    dit.py               #   Diffusion Transformer (image/video/text via config)
    autoencoder_kl.py    #   VAE encoder/decoder

  schedulers/            # processes (training math + sampling step)
    base.py              #   Scheduler ABC + shared helpers
    ddpm.py  ddim.py  euler.py  flow_match_euler.py

  pipelines/
    base.py              #   DiffusionPipeline: from_pretrained/save_pretrained + registry
    <family>.py          #   concrete pipelines

  training/
    trainer.py           #   compiled train step, mixed precision, ckpt, EMA hook
    losses.py            #   mse/eps/v/x0/flow target losses, weightings
    ema.py
    data.py              #   minimal dataset/loader helpers

  lora/
    lora.py              #   LoRALinear/Conv, inject/extract/merge/save/load

  cli/
    __main__.py          #   `mlx-diffusion generate|train|convert`
```

## 4. Key conventions (the "PyTorch/HF feel")

- **Configs are dataclasses** that round-trip to `config.json`. A model's
  `__init__(self, config)` takes its config object; `from_pretrained` reads
  `config.json` + weights and reconstructs. No hidden global state.
- **Weights are `safetensors`** (`*.safetensors`), sharded when large. Param names
  mirror module attribute paths (`mlx.nn` flatten/unflatten), so checkpoints are
  introspectable and partial-loadable.
- **One model = one directory**: `config.json` + `model.safetensors`. One pipeline
  = a directory of subfolders (`unet/`, `vae/`, `scheduler/`, ...), exactly like
  diffusers, so existing repos are convertible.
- **dtype & quantization are load-time args**, not new classes:
  `from_pretrained(..., dtype=mx.bfloat16, quantize=4)`.
- **Determinism**: every stochastic entry point takes a `key` (an
  `mx.random` key) or an int `seed`; no reliance on implicit global RNG in library
  code paths that matter for reproducibility.
- **Lazy by default**: build the graph, evaluate at boundaries (`mx.eval`) — the
  sampling loop and trainer manage evaluation points explicitly.

## 5. MLX performance strategy

- `mx.compile` the per-step network call and the train step; cache compiled fns
  keyed by shape/dtype.
- `mx.fast.scaled_dot_product_attention`, `mx.fast.rms_norm`, `mx.fast.rope`
  wherever applicable.
- Weight-only quantization (`mx.quantize`, groups of 64) for 4/8-bit inference so
  large models fit in unified memory on 16–32 GB Macs.
- bf16/fp16 compute with fp32 master where numerically needed (VAE, norms).
- Unified memory: no host<->device copies; expose `mx.set_memory_limit` knobs and
  document multi-Mac (`mx.distributed`) as a later extension.

## 6. Testing & CI philosophy

- Tests run on **tiny configs** (hidden sizes of 8–32, 1–2 layers) so the whole
  suite runs in seconds on CPU/GPU with **no network downloads**.
- Property tests for process math (e.g. `add_noise` at t=0 is identity; a full
  DDIM trajectory on a trivial model is finite & shaped correctly).
- Save/load round-trips, LoRA merge equivalence, end-to-end "train a few steps,
  loss goes down on a memorizable batch".
- CI on **GitHub `macos-14` arm64 runners** (real Apple silicon) — lint (ruff),
  types (mypy), tests (pytest), build (`python -m build` + `twine check`).

## 7. Roadmap to v0.1 (PyPI)

Core foundation → schedulers → layers/models → pipelines → training/LoRA →
MLX optimizations → CLI → tests+CI → docs site → release polish. Each lands as a
focused, reviewable commit on a `feat/*` branch.
