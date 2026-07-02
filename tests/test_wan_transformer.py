"""Tests for the WAN DiT and umT5 ports (tiny configs, no downloads).

Numerical parity against the official weights is checked in scripts/check_wan_*.py.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_diffuser.models.umt5 import UMT5Config, UMT5EncoderModel
from mlx_diffuser.models.wan_transformer import (
    WanRotaryPosEmbed,
    WanTransformer3DModel,
    WanTransformerConfig,
)


def tiny_dit_cfg(**kw) -> WanTransformerConfig:
    base = dict(
        num_attention_heads=2,
        attention_head_dim=12,  # head_dim divisible by 2; 2*(12//6)=4 -> t_dim=4,h=4,w=4
        in_channels=16,
        out_channels=16,
        text_dim=32,
        freq_dim=16,
        ffn_dim=64,
        num_layers=2,
    )
    base.update(kw)
    return WanTransformerConfig(**base)


def test_rope_shapes_and_axis_split():
    rope = WanRotaryPosEmbed(head_dim=128)
    assert rope.dims == (44, 42, 42)  # t,h,w pair dims sum to head_dim
    cos, sin = rope(2, 3, 4)
    assert cos.shape == (1, 2 * 3 * 4, 1, 64)
    assert sin.shape == (1, 2 * 3 * 4, 1, 64)


def test_dit_output_shape():
    model = WanTransformer3DModel(tiny_dit_cfg())
    x = mx.random.normal((1, 3, 8, 8, 16))  # B,T,H,W,C latent
    t = mx.array([500.0])
    ctx = mx.random.normal((1, 6, 32))
    out = model(x, t, ctx)
    mx.eval(out)
    assert out.shape == (1, 3, 8, 8, 16)  # patch (1,2,2) folds back exactly


def test_dit_default_config_matches_1_3b_dims():
    cfg = WanTransformerConfig()
    assert cfg.inner_dim == 1536
    assert cfg.attention_head_dim % 2 == 0


def test_umt5_output_shape():
    cfg = UMT5Config(vocab_size=128, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4)
    model = UMT5EncoderModel(cfg)
    ids = mx.array([[3, 7, 9, 1, 0]])
    mask = mx.array([[1, 1, 1, 1, 0]])
    out = model(ids, mask)
    mx.eval(out)
    assert out.shape == (1, 5, 32)


def test_umt5_relative_bias_present_each_layer():
    cfg = UMT5Config(vocab_size=64, d_model=16, d_kv=8, d_ff=32, num_layers=3, num_heads=2)
    model = UMT5EncoderModel(cfg)
    for block in model.encoder.block:
        assert block.layer[0].SelfAttention.has_relative_attention_bias
