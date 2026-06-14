"""Tests for performance helpers (compilation + memory)."""

from __future__ import annotations

import mlx.core as mx

from mlx_diffusion.models import DiT, DiTConfig
from mlx_diffusion.perf import (
    bytes_to_gb,
    clear_cache,
    compile_model,
    memory_report,
    reset_peak_memory,
)
from mlx_diffusion.pipelines import ClassConditionalPipeline
from mlx_diffusion.schedulers import FlowMatchEulerScheduler


def _model(num_classes=0):
    return DiT(DiTConfig(in_channels=3, patch_size=2, hidden_size=16, depth=2, num_heads=2, num_classes=num_classes))


def test_compiled_matches_eager():
    model = _model()
    x = mx.random.normal((2, 8, 8, 3))
    t = mx.array([0.2, 0.7])
    eager = model(x, t)
    compiled = compile_model(model)
    assert mx.allclose(compiled(x, t), eager, atol=1e-5)


def test_memory_report_keys():
    report = memory_report()
    assert set(report) == {"active_gb", "peak_gb", "cache_gb"}
    assert all(isinstance(v, float) and v >= 0 for v in report.values())


def test_memory_controls_run():
    reset_peak_memory()
    clear_cache()
    assert bytes_to_gb(1024**3) == 1.0


def test_pipeline_compile_matches_uncompiled():
    model = _model(num_classes=4)
    from mlx.utils import tree_map

    model.update(tree_map(lambda p: p + 0.05 * mx.random.normal(p.shape), model.parameters()))
    mx.eval(model.parameters())
    pipe = ClassConditionalPipeline(model, FlowMatchEulerScheduler())

    out_compiled = pipe([1], sample_size=8, num_inference_steps=5, seed=0, compile=True)
    out_eager = pipe([1], sample_size=8, num_inference_steps=5, seed=0, compile=False)
    assert mx.allclose(out_compiled, out_eager, atol=1e-4)
