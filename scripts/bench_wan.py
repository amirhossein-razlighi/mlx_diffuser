"""Benchmark the WAN transformer hot path (the denoising-loop forward).

Loads only the transformer (fast) and times one classifier-free-guidance step at
the 256 / 512 latent grids under several execution strategies, validating that
every strategy returns the same result as the eager baseline.

    PYTHONPATH=src uv run --no-sync python scripts/bench_wan.py
"""

from __future__ import annotations

import argparse
import time
from typing import cast

import mlx.core as mx

from mlx_diffuser.converters import get_converter
from mlx_diffuser.models.wan_transformer import WanTransformer3DModel

TF = "checkpoints/wan2.1-t2v-1.3b/transformer"
TEXT_TOKENS = 226


def grid(height: int, num_frames: int = 17) -> tuple[int, int, int, int, int]:
    return (1, (num_frames - 1) // 4 + 1, height // 8, height // 8, 16)


def time_fn(fn, *, warmup: int = 2, iters: int = 8) -> tuple[float, float]:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    dt = (time.perf_counter() - t0) / iters
    peak = mx.get_peak_memory() / 1024**3
    return dt * 1000, peak


def cosine(a: mx.array, b: mx.array) -> float:
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    return float((a * b).sum() / (mx.linalg.norm(a) * mx.linalg.norm(b) + 1e-9))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    p.add_argument("--guidance", type=float, default=5.0)
    args = p.parse_args()

    print("loading transformer (bf16)…")
    model = cast(
        WanTransformer3DModel,
        get_converter("WanTransformer3DModel").convert(TF, dtype=mx.bfloat16),
    )
    gs = args.guidance

    for size in args.sizes:
        b, t, h, w, c = grid(size)
        x = mx.random.normal((1, t, h, w, c)).astype(mx.bfloat16)
        ctx = (mx.random.normal((1, TEXT_TOKENS, 4096)) * 0.07).astype(mx.bfloat16)
        nctx = (mx.random.normal((1, TEXT_TOKENS, 4096)) * 0.07).astype(mx.bfloat16)
        tt = mx.array([999.0])
        tokens = t * h * w
        print(f"\n=== {size}px  latent {(1, t, h, w, c)}  ({tokens} tokens) ===")

        x2 = mx.concatenate([x, x], axis=0)
        tt2 = mx.concatenate([tt, tt], axis=0)
        ctx2 = mx.concatenate([ctx, nctx], axis=0)
        cstep = mx.compile(model)

        # Loop-dependent tensors are bound as defaults so each closure captures this
        # iteration's values (and ruff's B023 stays quiet).
        def eager_seq(x=x, tt=tt, ctx=ctx, nctx=nctx):
            cond = model(x, tt, ctx)
            uncond = model(x, tt, nctx)
            return uncond + gs * (cond - uncond)

        def batched(x2=x2, tt2=tt2, ctx2=ctx2):
            out = model(x2, tt2, ctx2)
            cond, uncond = out[:1], out[1:]
            return uncond + gs * (cond - uncond)

        def compiled_batched(x2=x2, tt2=tt2, ctx2=ctx2, cstep=cstep):
            out = cstep(x2, tt2, ctx2)
            cond, uncond = out[:1], out[1:]
            return uncond + gs * (cond - uncond)

        ref = eager_seq()
        mx.eval(ref)

        for name, fn in [
            ("eager  cfg=2 sequential", eager_seq),
            ("batched cfg (1 fwd, b=2)", batched),
            ("compiled batched cfg   ", compiled_batched),
        ]:
            ms, peak = time_fn(fn)
            cos = cosine(fn(), ref)
            print(f"  {name}: {ms:8.1f} ms/step   peak {peak:5.2f} GB   cos_vs_eager {cos:.5f}")


if __name__ == "__main__":
    main()
