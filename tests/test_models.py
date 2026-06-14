"""Tests for layers and the DiT backbone (tiny configs, no downloads)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_diffusion.layers import (
    Attention,
    get_2d_sincos_pos_embed,
    timestep_embedding,
)
from mlx_diffusion.models import DiT, DiTConfig


def tiny_config(**overrides) -> DiTConfig:
    base = dict(
        in_channels=3, patch_size=2, hidden_size=16, depth=2, num_heads=2, mlp_ratio=2.0
    )
    base.update(overrides)
    return DiTConfig(**base)


def test_timestep_embedding_shape():
    emb = timestep_embedding(mx.array([0.0, 1.0, 2.0]), 8)
    assert emb.shape == (3, 8)


def test_pos_embed_cached_and_shaped():
    a = get_2d_sincos_pos_embed(16, 4, 4)
    b = get_2d_sincos_pos_embed(16, 4, 4)
    assert a.shape == (16, 16)
    assert a is b  # lru_cache returns the same object


def test_attention_shapes():
    attn = Attention(dim=16, num_heads=2)
    x = mx.random.normal((2, 9, 16))
    assert attn(x).shape == x.shape
    ctx = mx.random.normal((2, 5, 16))
    assert attn(x, context=ctx).shape == x.shape


def test_dit_output_shape_unconditional():
    model = DiT(tiny_config())
    x = mx.random.normal((2, 8, 8, 3))
    t = mx.array([0.1, 0.5])
    out = model(x, t)
    assert out.shape == x.shape


def test_dit_adaln_zero_init_outputs_zero():
    """adaLN-Zero: an untrained DiT outputs all zeros (identity residual path)."""
    model = DiT(tiny_config())
    x = mx.random.normal((2, 8, 8, 3))
    out = model(x, mx.array([0.3, 0.7]))
    assert mx.max(mx.abs(out)).item() < 1e-6


def test_dit_class_conditional():
    model = DiT(tiny_config(num_classes=10))
    x = mx.random.normal((2, 8, 8, 3))
    t = mx.array([0.2, 0.8])
    y = mx.array([1, 7])
    assert model(x, t, y).shape == x.shape
    with pytest.raises(ValueError):
        model(x, t)  # missing labels


def test_dit_out_channels_override():
    model = DiT(tiny_config(in_channels=3, out_channels=6))
    out = model(mx.random.normal((1, 8, 8, 3)), mx.array([0.5]))
    assert out.shape == (1, 8, 8, 6)


def test_dit_save_load_roundtrip(tmp_path):
    model = DiT(tiny_config(num_classes=4))
    # Perturb params so it is not the trivial zero-output model.
    from mlx.utils import tree_map

    model.update(tree_map(lambda p: p + 0.05 * mx.random.normal(p.shape), model.parameters()))
    mx.eval(model.parameters())

    x = mx.random.normal((1, 8, 8, 3))
    t = mx.array([0.4])
    y = mx.array([2])
    y_before = model(x, t, y)

    model.save_pretrained(tmp_path)
    reloaded = DiT.from_pretrained(tmp_path)
    assert mx.allclose(reloaded(x, t, y), y_before, atol=1e-5)


def test_dit_num_parameters_positive():
    assert DiT(tiny_config()).num_parameters() > 0
