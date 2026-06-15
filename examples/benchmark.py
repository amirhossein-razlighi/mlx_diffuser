"""Benchmark sampling throughput and memory for a DiT on Apple silicon.

Usage:
    uv run python examples/benchmark.py --steps 50 --size 32 --batch 4

Reports per-image latency and peak unified memory, comparing compiled vs eager.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from mlx_diffuser.models import DiT, DiTConfig
from mlx_diffuser.perf import memory_report, reset_peak_memory
from mlx_diffuser.pipelines import ClassConditionalPipeline
from mlx_diffuser.schedulers import FlowMatchEulerScheduler


def run(pipe, labels, *, size, steps, compile, warmup=1, iters=3) -> float:
    for _ in range(warmup):
        mx.eval(pipe(labels, sample_size=size, num_inference_steps=steps, seed=0, compile=compile))
    start = time.perf_counter()
    for i in range(iters):
        mx.eval(pipe(labels, sample_size=size, num_inference_steps=steps, seed=i, compile=compile))
    return (time.perf_counter() - start) / iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    args = p.parse_args()

    cfg = DiTConfig(
        in_channels=3,
        patch_size=2,
        hidden_size=args.hidden,
        depth=args.depth,
        num_heads=max(1, args.hidden // 64),
        num_classes=1000,
    )
    pipe = ClassConditionalPipeline(DiT(cfg), FlowMatchEulerScheduler())
    labels = mx.array(list(range(args.batch)))
    print(
        f"model: {pipe.model.num_parameters() / 1e6:.1f}M params  | {args.batch}x{args.size}x{args.size}  {args.steps} steps"
    )

    for compile in (False, True):
        reset_peak_memory()
        secs = run(pipe, labels, size=args.size, steps=args.steps, compile=compile)
        mem = memory_report()
        tag = "compiled" if compile else "eager"
        print(
            f"  {tag:8s}: {secs * 1000:8.1f} ms/batch  {secs / args.batch * 1000:7.1f} ms/image  peak {mem['peak_gb']:.2f} GB"
        )


if __name__ == "__main__":
    main()
