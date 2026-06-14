# Concepts

Every diffusion / flow model in this library is three orthogonal pieces:

```
        data x0  ──►  [ PROCESS ]  ──►  noisy x_t , target
                          │                      │
                          ▼                      ▼
                     [ NETWORK ]  predicts  ──►  model_output
                          │
                          ▼
        x_t   ──►  [ SAMPLER (= process.step) ]  ──►  x_{t-1}  ──►  ...  ──►  x0
```

## Process (scheduler)

A **scheduler** owns the corruption math and the reverse step. The *same* object
is used for training and sampling:

- `add_noise(x0, noise, t)` — forward/training corruption
- `get_target(x0, noise, t)` — what the network should predict (epsilon, v, x0, or
  velocity)
- `sample_timesteps(batch, key)` — draw training timesteps
- `set_timesteps(n)` + `step(model_output, t, x_t)` — sampling

One abstraction spans both classic diffusion (DDPM/DDIM/Euler) and rectified-flow
/ flow-matching. See [Schedulers](guides/schedulers.md).

## Network (model)

A **network** is a plain `mlx.nn.Module` that maps `(x_t, t, conditioning) ->
prediction`. It knows nothing about noise schedules. Every model is config-driven
and gets `from_pretrained` / `save_pretrained` from `ModelMixin`. See
[Models](guides/models.md).

## Pipeline

A **pipeline** bundles a network + scheduler (+ optional VAE/conditioner) behind a
single `__call__` for inference. It is pure convenience — you never need it to use
the parts. See the [Pipelines reference](reference/pipelines.md).

## Why this split matters

Keeping **process** and **network** independent is the central design decision: it
lets the same network train under DDPM today and flow-matching tomorrow, and lets a
researcher swap one axis without touching the other. Training is symmetric to
sampling — the `DiffusionTrainer` just calls `add_noise` + `get_target`, runs the
network, and applies a loss.

## Conventions

- **Channels-last** tensors `(B, H, W, C)` throughout (MLX-native).
- **Configs are dataclasses** that round-trip to `config.json`.
- **Weights are `safetensors`**; one model = one directory.
- **dtype & quantization are load-time args**: `from_pretrained(..., dtype="bf16",
  quantize=4)`.
- **Determinism** via explicit `seed`/`key` arguments.
