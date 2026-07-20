# Performance on Apple silicon

mlx-diffuser leans on MLX's strengths. Most of it is on by default.

## Compilation

Sampling and training steps are compiled. For custom inference loops, compile the
model once (per-step shapes are constant, so one graph is reused):

```python
from mlx_diffuser.perf import compile_model
fast_model = compile_model(model)        # params passed as implicit inputs
```

Parameters are passed as implicit inputs, so a later `merge_lora` or weight change
is picked up without a stale graph. Use `shapeless=True` only if you understand the
[shapeless-compile caveats](https://ml-explore.github.io/mlx/build/html/usage/compile.html).

## Quantization

Weight-only 4/8-bit quantization at load time fits large models in unified memory:

```python
model = DiT.from_pretrained("my-model", quantize=4)         # 4-bit
# or quantize an in-memory model:
from mlx_diffuser import quantize_module
quantize_module(model, bits=8, group_size=64)
```

## Precision

Use `dtype="bf16"` (or `"fp16"`) for compute; normalization and attention
accumulate in higher precision internally, so no manual up/down-casting is needed.

## Unified memory

```python
from mlx_diffuser.perf import memory_report, set_memory_limit, clear_cache

set_memory_limit(24)          # soft cap, GB
print(memory_report())        # {'active_gb': ..., 'peak_gb': ..., 'cache_gb': ...}
clear_cache()                 # return cached buffers to the OS
```

## The 16 GB preset

The CLI can choose the safe defaults as a group:

```bash
mlx-diffuser generate --model flux --prompt "..." --low-memory --out image.png
```

| Pipeline | `--low-memory` behavior |
| --- | --- |
| SDXL | 8-bit UNet, release both CLIP encoders, tiled VAE decode |
| FLUX.1 | 4-bit transformer/T5, release text encoders, tiled VAE decode |
| WAN 2.1 | 4-bit transformer and released umT5 encoder |
| LTX-2.3 | already staged; only one large component is resident at a time |
| TRELLIS | always staged; 8-bit dense weights, per-block sparse evaluation, Metal sparse Conv3D |

This is a fit-first preset, not always the fastest setting. Once a workload is stable,
benchmark individual switches and keep only those needed for the chosen resolution.

The verified TRELLIS image-large sample (25 + 25 steps, seed 42) completed in 202.36
seconds at 2.07 GB MLX peak memory on an 8-core M1 Pro with 16 GB unified memory.
Checkpoint download/conversion and the documentation preview render are excluded.

## Benchmarking

```bash
uv run python examples/benchmark.py --steps 50 --size 32 --batch 4
```

Reports per-image latency and peak memory, comparing compiled vs eager.

## Checklist

- Keep input shapes stable so compiled graphs aren't retraced.
- Multiply half-precision arrays by **Python** scalars (not `mx.array`) to avoid
  silent up-casting to float32.
- Evaluate once per iteration (the trainer and sampler already do this).
