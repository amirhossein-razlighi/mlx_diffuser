"""Apple-silicon performance helpers: compilation and unified-memory control.

These wrap MLX's compile and memory primitives with diffusion-friendly defaults.
The heavy lifting (fused attention via ``mx.fast.scaled_dot_product_attention``,
lazy evaluation, an ``mx.compile``-fused train step, weight quantization) already
lives in the relevant modules; this collects the remaining knobs in one place.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "compile_model",
    "memory_report",
    "set_memory_limit",
    "set_cache_limit",
    "clear_cache",
    "reset_peak_memory",
    "bytes_to_gb",
]


def bytes_to_gb(n: int) -> float:
    return n / (1024**3)


def compile_model(model: nn.Module, *, shapeless: bool = False) -> Callable:
    """Return a compiled callable for *inference* with this model.

    Parameters are passed as implicit inputs so weight updates (e.g. after a LoRA
    merge) are picked up without a stale graph. Use ``shapeless=True`` only when
    input ranks are stable but sizes vary a lot (read MLX's shapeless-compile
    caveats first) to avoid recompiling per resolution.
    """
    return mx.compile(model, inputs=model.state, shapeless=shapeless)


def memory_report() -> dict[str, float]:
    """Snapshot of unified-memory usage in GB (active / peak / cache)."""
    return {
        "active_gb": bytes_to_gb(mx.get_active_memory()),
        "peak_gb": bytes_to_gb(mx.get_peak_memory()),
        "cache_gb": bytes_to_gb(mx.get_cache_memory()),
    }


def set_memory_limit(gb: float) -> int:
    """Cap MLX allocations (soft limit). Returns the previous limit in bytes."""
    return mx.set_memory_limit(int(gb * 1024**3))


def set_cache_limit(gb: float) -> int:
    """Cap the buffer cache; ``0`` disables caching. Returns previous limit."""
    return mx.set_cache_limit(int(gb * 1024**3))


def clear_cache() -> None:
    """Return cached (but unused) buffers to the system."""
    mx.clear_cache()


def reset_peak_memory() -> None:
    """Reset the peak-memory counter (call before a section you want to measure)."""
    mx.reset_peak_memory()
