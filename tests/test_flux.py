"""Tests for the FLUX.1 component ports on tiny configs (no downloads)."""

from __future__ import annotations

import mlx.core as mx

from mlx_diffuser.caching import FirstBlockCache
from mlx_diffuser.models import (
    AutoencoderKLSD,
    AutoencoderKLSDConfig,
    FluxConfig,
    FluxTransformer2DModel,
    T5Config,
    T5EncoderModel,
)
from mlx_diffuser.pipelines.flux import _pack_latents, _prepare_image_ids, _unpack_latents
from mlx_diffuser.quantization import quantize_module


def tiny_flux(**kw) -> FluxConfig:
    base = dict(
        in_channels=64,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=24,
        guidance_embeds=False,
        axes_dims_rope=(4, 6, 6),
    )
    base.update(kw)
    return FluxConfig(**base)


def tiny_t5(**kw) -> T5Config:
    base = dict(
        vocab_size=64,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=2,
        num_heads=4,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=16,
    )
    base.update(kw)
    return T5Config(**base)


def tiny_flux_vae(**kw) -> AutoencoderKLSDConfig:
    base = dict(
        block_out_channels=(8, 16),
        layers_per_block=1,
        norm_groups=4,
        latent_channels=16,
        scaling_factor=0.3611,
        shift_factor=0.1159,
        use_quant_conv=False,
    )
    base.update(kw)
    return AutoencoderKLSDConfig(**base)


def _flux_inputs(b: int = 1, l_img: int = 16, l_txt: int = 5):
    x = mx.random.normal((b, l_img, 64))
    t = mx.array([0.7] * b)
    ehs = mx.random.normal((b, l_txt, 32))
    pooled = mx.random.normal((b, 24))
    img_ids = _prepare_image_ids(4, 4)  # 16 tokens
    txt_ids = mx.zeros((l_txt, 3))
    return x, t, ehs, pooled, img_ids, txt_ids


# --- transformer -------------------------------------------------------------


def test_flux_output_shape():
    model = FluxTransformer2DModel(tiny_flux())
    out = model(*_flux_inputs())
    assert out.shape == (1, 16, 64)


def test_flux_guidance_variant_runs():
    model = FluxTransformer2DModel(tiny_flux(guidance_embeds=True))
    x, t, ehs, pooled, img_ids, txt_ids = _flux_inputs()
    out = model(x, t, ehs, pooled, img_ids, txt_ids, guidance=mx.array([3.5]))
    assert out.shape == (1, 16, 64)


def test_flux_save_load_roundtrip(tmp_path):
    from mlx.utils import tree_map

    model = FluxTransformer2DModel(tiny_flux())
    model.update(tree_map(lambda p: p + 0.02 * mx.random.normal(p.shape), model.parameters()))
    mx.eval(model.parameters())
    args = _flux_inputs()
    before = model(*args)
    model.save_pretrained(tmp_path)
    reloaded = FluxTransformer2DModel.from_pretrained(tmp_path)
    assert mx.allclose(reloaded(*args), before, atol=1e-4)


def test_flux_firstblock_cache_path():
    model = FluxTransformer2DModel(tiny_flux())
    args = _flux_inputs()
    cache = FirstBlockCache(threshold=10.0)  # high threshold -> always reuse after step 1
    shapes = [model(*args, cache=cache).shape for _ in range(4)]
    assert all(s == (1, 16, 64) for s in shapes)
    assert cache.skipped >= 1


def test_flux_quantized_runs():
    model = FluxTransformer2DModel(
        tiny_flux(attention_head_dim=64, joint_attention_dim=128, axes_dims_rope=(16, 24, 24))
    )
    quantize_module(model, bits=8, group_size=64)
    mx.eval(model.parameters())
    x = mx.random.normal((1, 16, 64))
    out = model(
        x,
        mx.array([0.5]),
        mx.random.normal((1, 5, 128)),
        mx.random.normal((1, 24)),
        _prepare_image_ids(4, 4),
        mx.zeros((5, 3)),
    )
    assert out.shape == (1, 16, 64)


# --- T5 ----------------------------------------------------------------------


def test_t5_shapes():
    model = T5EncoderModel(tiny_t5())
    out = model(mx.array([[1, 2, 3, 4, 0, 0]]))
    assert out.shape == (1, 6, 32)


def test_t5_attention_mask_runs():
    model = T5EncoderModel(tiny_t5())
    ids = mx.array([[1, 2, 3, 0, 0, 0]])
    mask = mx.array([[1, 1, 1, 0, 0, 0]])
    assert model(ids, mask).shape == (1, 6, 32)


# --- VAE ---------------------------------------------------------------------


def test_flux_vae_no_quant_conv_and_shift():
    vae = AutoencoderKLSD(tiny_flux_vae())
    assert vae.quant_conv is None and vae.post_quant_conv is None
    assert vae.shift_factor == 0.1159
    z = mx.random.normal((1, 8, 8, 16))
    assert vae.decode(z).shape == (1, 16, 16, 3)  # 2 levels -> one upsample ×2


# --- packing -----------------------------------------------------------------


def test_pack_unpack_roundtrip():
    latents = mx.random.normal((1, 8, 6, 16))
    packed = _pack_latents(latents)
    assert packed.shape == (1, 4 * 3, 16 * 4)
    back = _unpack_latents(packed, 4, 3, 16)
    assert mx.allclose(back, latents, atol=1e-6)


def test_image_ids_grid():
    ids = _prepare_image_ids(2, 3)
    assert ids.shape == (6, 3)
    assert mx.array_equal(ids[:, 1], mx.array([0, 0, 0, 1, 1, 1]))  # row index
    assert mx.array_equal(ids[:, 2], mx.array([0, 1, 2, 0, 1, 2]))  # col index
