# Performance on Apple silicon

mlx-diffuser leans on MLX's strengths. Most of it is on by default.

## Compilation

Sampling and training steps are compiled. For custom inference loops, compile the
model once (per-step shapes are constant, so one graph is reused):

```python
from mlx_diffusion.perf import compile_model
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
from mlx_diffusion import quantize_module
quantize_module(model, bits=8, group_size=64)
```

## Precision

Use `dtype="bf16"` (or `"fp16"`) for compute; normalization and attention
accumulate in higher precision internally, so no manual up/down-casting is needed.

## Unified memory

```python
from mlx_diffusion.perf import memory_report, set_memory_limit, clear_cache

set_memory_limit(24)          # soft cap, GB
print(memory_report())        # {'active_gb': ..., 'peak_gb': ..., 'cache_gb': ...}
clear_cache()                 # return cached buffers to the OS
```

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
