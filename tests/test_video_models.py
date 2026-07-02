"""Tests for the video stack: 3D RoPE, VideoDiT, the causal 3D VAE, and the
text-to-video pipeline (tiny configs, no downloads)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_diffuser.layers import rope_3d_freqs
from mlx_diffuser.models import (
    AutoencoderKLVideo,
    AutoencoderKLVideoConfig,
    VideoDiT,
    VideoDiTConfig,
)
from mlx_diffuser.pipelines import TextToVideoPipeline
from mlx_diffuser.quantization import quantize_module
from mlx_diffuser.schedulers import FlowMatchEulerScheduler


def tiny_dit(**overrides) -> VideoDiTConfig:
    base = dict(
        in_channels=8,
        patch_size=(1, 2, 2),
        hidden_size=32,
        depth=2,
        num_heads=2,
        cross_attn_dim=16,
    )
    base.update(overrides)
    return VideoDiTConfig(**base)


def tiny_vae(**overrides) -> AutoencoderKLVideoConfig:
    base = dict(
        in_channels=3,
        latent_channels=8,
        block_out_channels=(16, 32, 64),  # spatial compression 4
        layers_per_block=1,
        temporal_compression=4,
        norm_groups=8,
    )
    base.update(overrides)
    return AutoencoderKLVideoConfig(**base)


# --- 3D RoPE -----------------------------------------------------------------


def test_rope_3d_shape_and_cache():
    cos, sin = rope_3d_freqs(16, 2, 3, 4)
    assert cos.shape == (2 * 3 * 4, 16)
    assert sin.shape == (2 * 3 * 4, 16)
    again = rope_3d_freqs(16, 2, 3, 4)
    assert again[0] is cos  # lru_cache returns the same object


def test_rope_3d_odd_head_dim_raises():
    with pytest.raises(ValueError):
        rope_3d_freqs(15, 2, 2, 2)


# --- VideoDiT ----------------------------------------------------------------


def test_video_dit_output_shape_conditional():
    model = VideoDiT(tiny_dit())
    x = mx.random.normal((2, 4, 8, 8, 8))
    t = mx.array([0.1, 0.5])
    ctx = mx.random.normal((2, 5, 16))
    assert model(x, t, ctx).shape == x.shape


def test_video_dit_unconditional_runs():
    model = VideoDiT(tiny_dit(cross_attn_dim=None))
    x = mx.random.normal((1, 4, 8, 8, 8))
    assert model(x, mx.array([0.3])).shape == x.shape


def test_video_dit_adaln_zero_init_outputs_zero():
    """adaLN-Zero + zero-init cross-gate: an untrained VideoDiT is the identity
    residual path and outputs all zeros."""
    model = VideoDiT(tiny_dit())
    x = mx.random.normal((1, 4, 8, 8, 8))
    out = model(x, mx.array([0.4]), mx.random.normal((1, 5, 16)))
    assert mx.max(mx.abs(out)).item() < 1e-6


def test_video_dit_out_channels_override():
    model = VideoDiT(tiny_dit(in_channels=8, out_channels=16))
    out = model(mx.random.normal((1, 4, 8, 8, 8)), mx.array([0.5]), mx.random.normal((1, 5, 16)))
    assert out.shape == (1, 4, 8, 8, 16)


def test_video_dit_invalid_head_dim_raises():
    with pytest.raises(ValueError):
        VideoDiT(tiny_dit(hidden_size=33, num_heads=3))  # not divisible


def test_video_dit_save_load_roundtrip(tmp_path):
    from mlx.utils import tree_map

    model = VideoDiT(tiny_dit())
    model.update(tree_map(lambda p: p + 0.05 * mx.random.normal(p.shape), model.parameters()))
    mx.eval(model.parameters())

    x = mx.random.normal((1, 4, 8, 8, 8))
    t = mx.array([0.4])
    ctx = mx.random.normal((1, 5, 16))
    before = model(x, t, ctx)

    model.save_pretrained(tmp_path)
    reloaded = VideoDiT.from_pretrained(tmp_path)
    assert mx.allclose(reloaded(x, t, ctx), before, atol=1e-5)


def test_video_dit_presets_are_well_formed():
    for name in ("wan_t2v_1_3b", "wan_t2v_14b", "ltx_video"):
        cfg = getattr(VideoDiTConfig, name)()
        assert cfg.hidden_size % cfg.num_heads == 0
        assert cfg.head_dim % 2 == 0  # required for RoPE


# --- AutoencoderKLVideo ------------------------------------------------------


def test_video_vae_compression_shapes():
    vae = AutoencoderKLVideo(tiny_vae())
    assert vae.config.spatial_compression == 4
    assert vae.config.temporal_compression == 4
    v = mx.random.normal((1, 8, 32, 32, 3))
    z = vae.encode(v).sample()
    assert z.shape == (1, 8 // 4, 32 // 4, 32 // 4, 8)


def test_video_vae_roundtrip_shape():
    vae = AutoencoderKLVideo(tiny_vae())
    v = mx.random.normal((1, 8, 32, 32, 3))
    recon, posterior = vae(v)
    assert recon.shape == v.shape
    assert posterior.mean.shape == (1, 2, 8, 8, 8)


# --- TextToVideoPipeline -----------------------------------------------------


def build_pipeline() -> TextToVideoPipeline:
    return TextToVideoPipeline(
        VideoDiT(tiny_dit()), AutoencoderKLVideo(tiny_vae()), FlowMatchEulerScheduler()
    )


def test_pipeline_generates_video():
    pipe = build_pipeline()
    video = pipe(
        mx.random.normal((1, 5, 16)),
        num_frames=8,
        height=32,
        width=32,
        num_inference_steps=2,
        guidance_scale=5.0,
        seed=0,
        compile=False,
    )
    assert video.shape == (1, 8, 32, 32, 3)


def test_pipeline_returns_latents_when_not_decoding():
    pipe = build_pipeline()
    latents = pipe(
        mx.random.normal((1, 5, 16)),
        num_frames=8,
        height=32,
        width=32,
        num_inference_steps=2,
        guidance_scale=1.0,  # no CFG
        seed=0,
        compile=False,
        decode=False,
    )
    assert latents.shape == (1, 2, 8, 8, 8)


def test_pipeline_rejects_indivisible_dimensions():
    pipe = build_pipeline()
    with pytest.raises(ValueError):
        pipe(mx.random.normal((1, 5, 16)), num_frames=6, height=32, width=32, compile=False)


def test_pipeline_seed_is_deterministic():
    pipe = build_pipeline()
    kw = dict(num_frames=8, height=32, width=32, num_inference_steps=2, seed=7, compile=False)
    emb = mx.random.normal((1, 5, 16))
    b = pipe(emb, **kw)
    c = pipe(emb, **kw)
    assert mx.allclose(b, c)  # same seed + same embeds -> identical


# --- Quantization ------------------------------------------------------------


def test_quantized_video_dit_runs():
    model = VideoDiT(tiny_dit(hidden_size=64, num_heads=2))
    quantize_module(model, bits=4, group_size=64)
    mx.eval(model.parameters())
    out = model(mx.random.normal((1, 4, 8, 8, 8)), mx.array([0.5]), mx.random.normal((1, 5, 16)))
    assert out.shape == (1, 4, 8, 8, 8)
