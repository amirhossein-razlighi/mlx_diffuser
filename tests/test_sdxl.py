"""Tests for the SDXL component ports on tiny configs (no downloads)."""

from __future__ import annotations

import mlx.core as mx

from mlx_diffuser.caching import DeepCache
from mlx_diffuser.models import (
    AutoencoderKLSD,
    AutoencoderKLSDConfig,
    CLIPTextConfig,
    CLIPTextModel,
    SDXLUNet,
    SDXLUNetConfig,
)
from mlx_diffuser.quantization import quantize_module


def tiny_clip(**kw) -> CLIPTextConfig:
    base = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=8,
        hidden_act="quick_gelu",
        projection_dim=16,
    )
    base.update(kw)
    return CLIPTextConfig(**base)


def tiny_vae(**kw) -> AutoencoderKLSDConfig:
    base = dict(block_out_channels=(8, 16), layers_per_block=1, norm_groups=4)
    base.update(kw)
    return AutoencoderKLSDConfig(**base)


def tiny_unet(**kw) -> SDXLUNetConfig:
    base = dict(
        block_out_channels=(16, 32),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        layers_per_block=1,
        transformer_layers_per_block=(1, 1),
        num_attention_heads=(2, 2),
        cross_attention_dim=16,
        addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=16 + 6 * 8,  # pooled(16) + 6*add_time_embed
        norm_groups=4,
    )
    base.update(kw)
    return SDXLUNetConfig(**base)


# --- CLIP --------------------------------------------------------------------


def test_clip_hidden_states_and_pooled_shapes():
    model = CLIPTextModel(tiny_clip())
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 0]])
    hidden_states, pooled = model(ids)
    assert len(hidden_states) == 3  # embeddings + 2 layers
    assert hidden_states[-2].shape == (1, 8, 16)
    assert pooled.shape == (1, 16)  # no projection -> hidden_size


def test_clip_with_projection():
    model = CLIPTextModel(tiny_clip(with_projection=True, projection_dim=12))
    _, pooled = model(mx.array([[1, 2, 3, 0, 0, 0, 0, 0]]))
    assert pooled.shape == (1, 12)


# --- VAE ---------------------------------------------------------------------


def test_vae_encode_decode_shapes():
    vae = AutoencoderKLSD(tiny_vae())
    x = mx.random.normal((1, 16, 16, 3))
    post = vae.encode(x)
    assert post.mean.shape == (1, 8, 8, 4)  # one downsample, latent_channels=4
    assert vae.decode(post.mode()).shape == (1, 16, 16, 3)


def test_vae_tiled_decode_matches_shape():
    vae = AutoencoderKLSD(tiny_vae())
    z = mx.random.normal((1, 8, 8, 4))
    full = vae.decode(z)
    tiled = vae.decode(z, tile=True, tile_latent=4, overlap_latent=2)
    assert tiled.shape == full.shape == (1, 16, 16, 3)  # tiny VAE upsamples ×2


# --- UNet --------------------------------------------------------------------


def _unet_inputs(unet: SDXLUNet, b: int = 1):
    x = mx.random.normal((b, 16, 16, 4))
    t = mx.array([5.0] * b)
    ehs = mx.random.normal((b, 7, 16))
    pooled = mx.random.normal((b, 16))
    time_ids = mx.broadcast_to(mx.array([[16.0, 16, 0, 0, 16, 16]]), (b, 6))
    return x, t, ehs, pooled, time_ids


def test_unet_output_shape():
    unet = SDXLUNet(tiny_unet())
    out = unet(*_unet_inputs(unet))
    assert out.shape == (1, 16, 16, 4)


def test_unet_save_load_roundtrip(tmp_path):
    from mlx.utils import tree_map

    unet = SDXLUNet(tiny_unet())
    unet.update(tree_map(lambda p: p + 0.02 * mx.random.normal(p.shape), unet.parameters()))
    mx.eval(unet.parameters())
    args = _unet_inputs(unet)
    before = unet(*args)
    unet.save_pretrained(tmp_path)
    reloaded = SDXLUNet.from_pretrained(tmp_path)
    assert mx.allclose(reloaded(*args), before, atol=1e-4)


def test_unet_deepcache_path_runs():
    unet = SDXLUNet(tiny_unet())
    args = _unet_inputs(unet)
    cache = DeepCache(interval=2)
    shapes = []
    for _ in range(4):  # full, cached, full, cached
        shapes.append(unet(*args, cache=cache).shape)
    assert all(s == (1, 16, 16, 4) for s in shapes)
    assert cache.skipped == 2  # steps 1 and 3 reused


def test_unet_quantized_runs():
    unet = SDXLUNet(tiny_unet(block_out_channels=(64, 64), num_attention_heads=(2, 2)))
    quantize_module(unet, bits=8, group_size=64)
    mx.eval(unet.parameters())
    out = unet(*_unet_inputs(unet))
    assert out.shape == (1, 16, 16, 4)


# --- DeepCache ---------------------------------------------------------------


def test_deepcache_schedule():
    cache = DeepCache(interval=3)
    assert cache.enabled
    cache.deep = mx.zeros((1,))  # pretend a full step populated it
    reuse = [cache.should_reuse() for _ in range(6)]
    # step 0,3 full (i % 3 == 0); 1,2,4,5 reuse
    assert reuse == [False, True, True, False, True, True]


def test_deepcache_disabled():
    cache = DeepCache(interval=1)
    assert not cache.enabled
    cache.deep = mx.zeros((1,))
    assert not any(cache.should_reuse() for _ in range(4))
