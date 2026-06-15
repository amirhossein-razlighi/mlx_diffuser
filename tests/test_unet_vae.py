"""Tests for UNet2D and AutoencoderKL on tiny configs."""

from __future__ import annotations

import mlx.core as mx

from mlx_diffusion.models import AutoencoderKL, AutoencoderKLConfig, UNet2D, UNet2DConfig


def tiny_unet(**kw) -> UNet2DConfig:
    base = dict(
        in_channels=4,
        out_channels=4,
        block_out_channels=(8, 16),
        layers_per_block=1,
        num_heads=2,
        norm_groups=8,
    )
    base.update(kw)
    return UNet2DConfig(**base)


def tiny_vae(**kw) -> AutoencoderKLConfig:
    base = dict(
        in_channels=3,
        latent_channels=4,
        block_out_channels=(8, 16),
        layers_per_block=1,
        norm_groups=8,
    )
    base.update(kw)
    return AutoencoderKLConfig(**base)


def test_unet_output_shape():
    model = UNet2D(tiny_unet())
    x = mx.random.normal((2, 16, 16, 4))
    out = model(x, mx.array([5.0, 10.0]))
    assert out.shape == x.shape


def test_unet_cross_attention():
    model = UNet2D(tiny_unet(cross_attention_dim=12, attention_levels=(True, True)))
    x = mx.random.normal((2, 16, 16, 4))
    context = mx.random.normal((2, 7, 12))
    out = model(x, mx.array([1.0, 2.0]), context=context)
    assert out.shape == x.shape


def test_unet_save_load_roundtrip(tmp_path):
    model = UNet2D(tiny_unet())
    x = mx.random.normal((1, 16, 16, 4))
    t = mx.array([3.0])
    y0 = model(x, t)
    model.save_pretrained(tmp_path)
    reloaded = UNet2D.from_pretrained(tmp_path)
    assert mx.allclose(reloaded(x, t), y0, atol=1e-4)


def test_vae_encode_decode_shapes():
    vae = AutoencoderKL(tiny_vae())
    x = mx.random.normal((2, 16, 16, 3))
    posterior = vae.encode(x)
    # n levels -> n-1 downsamples; here 2 levels -> H/2, latent_channels.
    assert posterior.mean.shape == (2, 8, 8, 4)
    z = posterior.sample(mx.random.key(0))
    recon = vae.decode(z)
    assert recon.shape == x.shape


def test_vae_kl_is_scalar_nonnegative():
    vae = AutoencoderKL(tiny_vae())
    posterior = vae.encode(mx.random.normal((1, 16, 16, 3)))
    kl = posterior.kl()
    assert kl.ndim == 0
    assert kl.item() >= 0.0


def test_vae_call_returns_recon_and_posterior():
    vae = AutoencoderKL(tiny_vae())
    recon, posterior = vae(mx.random.normal((1, 16, 16, 3)), key=mx.random.key(1))
    assert recon.shape == (1, 16, 16, 3)
    assert hasattr(posterior, "kl")
    assert abs(vae.scaling_factor - 0.18215) < 1e-6
