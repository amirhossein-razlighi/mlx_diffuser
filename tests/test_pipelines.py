"""End-to-end pipeline tests on tiny models (no downloads)."""

from __future__ import annotations

import mlx.core as mx

from mlx_diffuser.models import DiT, DiTConfig
from mlx_diffuser.pipelines import ClassConditionalPipeline, DiffusionPipeline
from mlx_diffuser.schedulers import DDIMScheduler, FlowMatchEulerScheduler


def build_pipeline(scheduler=None) -> ClassConditionalPipeline:
    cfg = DiTConfig(
        in_channels=3, patch_size=2, hidden_size=16, depth=2, num_heads=2, num_classes=8
    )
    model = DiT(cfg)
    # Break the adaLN-Zero identity so outputs actually move.
    from mlx.utils import tree_map

    model.update(tree_map(lambda p: p + 0.02 * mx.random.normal(p.shape), model.parameters()))
    mx.eval(model.parameters())
    return ClassConditionalPipeline(model, scheduler or FlowMatchEulerScheduler())


def test_generate_shape_and_finite():
    pipe = build_pipeline()
    out = pipe([1, 3], sample_size=8, num_inference_steps=5, guidance_scale=2.0, seed=0)
    assert out.shape == (2, 8, 8, 3)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_generate_without_cfg():
    pipe = build_pipeline()
    out = pipe([0], sample_size=8, num_inference_steps=4, guidance_scale=1.0, seed=1)
    assert out.shape == (1, 8, 8, 3)


def test_generate_deterministic_with_seed():
    pipe = build_pipeline()
    a = pipe([2], sample_size=8, num_inference_steps=4, seed=42)
    b = pipe([2], sample_size=8, num_inference_steps=4, seed=42)
    assert mx.allclose(a, b)


def test_generate_with_ddim_scheduler():
    pipe = build_pipeline(scheduler=DDIMScheduler())
    out = pipe([5], sample_size=8, num_inference_steps=6, guidance_scale=3.0, seed=7)
    assert out.shape == (1, 8, 8, 3)
    assert bool(mx.all(mx.isfinite(out)).item())


def test_pipeline_save_load_roundtrip(tmp_path):
    pipe = build_pipeline()
    out_before = pipe([4], sample_size=8, num_inference_steps=4, seed=3)

    pipe.save_pretrained(tmp_path)
    assert (tmp_path / "model_index.json").exists()
    assert (tmp_path / "model" / "config.json").exists()
    assert (tmp_path / "scheduler" / "config.json").exists()

    # Base-class dispatch should reconstruct the concrete pipeline.
    reloaded = DiffusionPipeline.from_pretrained(tmp_path)
    assert isinstance(reloaded, ClassConditionalPipeline)
    out_after = reloaded([4], sample_size=8, num_inference_steps=4, seed=3)
    assert mx.allclose(out_before, out_after, atol=1e-4)
