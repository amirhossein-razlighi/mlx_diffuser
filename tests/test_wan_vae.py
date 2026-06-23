"""Tests for the WAN 2.1 VAE port and the checkpoint-converter subsystem.

These use a tiny config (no downloads). Numerical parity against the official
diffusers weights is verified separately in scripts/check_wan_vae.py, which needs
torch + the downloaded checkpoint and so is not part of the unit suite.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_diffuser.converters import convert_conv_weight, get_converter
from mlx_diffuser.converters.base import Converter, _assert_matches
from mlx_diffuser.models import AutoencoderKLWan, AutoencoderKLWanConfig


def tiny_vae() -> AutoencoderKLWan:
    # Real WAN temporal structure (two temporal downsamples -> factor 4), tiny width.
    return AutoencoderKLWan(
        AutoencoderKLWanConfig(
            base_dim=8,
            z_dim=4,
            dim_mult=(1, 2, 4, 4),
            num_res_blocks=1,
            temperal_downsample=(False, True, True),
            latents_mean=tuple(0.0 for _ in range(4)),
            latents_std=tuple(1.0 for _ in range(4)),
        )
    )


def test_convert_conv_weight_layout():
    w5 = mx.zeros((6, 3, 1, 2, 2))  # (O, I, kT, kH, kW)
    assert convert_conv_weight(w5).shape == (6, 1, 2, 2, 3)
    w4 = mx.zeros((6, 3, 3, 3))  # (O, I, kH, kW)
    assert convert_conv_weight(w4).shape == (6, 3, 3, 3)  # transposed -> (O,kH,kW,I)
    w1 = mx.zeros((6,))
    assert convert_conv_weight(w1).shape == (6,)  # untouched


def test_vae_compression_and_frame_count():
    vae = tiny_vae()
    # 3 spatial downsamples -> /8; T=9 -> 1 + (9-1)//4 = 3 latent frames.
    x = mx.random.normal((1, 9, 16, 16, 3))
    z = vae.encode(x).mode()
    mx.eval(z)
    assert z.shape == (1, 3, 2, 2, 4)


def test_vae_roundtrip_shape():
    vae = tiny_vae()
    x = mx.random.normal((1, 5, 16, 16, 3))  # T=5 -> 2 latent frames
    recon, posterior = vae(x)
    mx.eval(recon)
    assert recon.shape == x.shape
    assert posterior.mean.shape == (1, 2, 2, 2, 4)
    assert float(mx.max(mx.abs(recon))) <= 1.0  # decoder clamps to [-1, 1]


def test_vae_latent_normalization_roundtrip():
    vae = AutoencoderKLWan(
        AutoencoderKLWanConfig(
            base_dim=8, z_dim=4, dim_mult=(1, 2, 4, 4), num_res_blocks=1,
            latents_mean=(0.1, -0.2, 0.3, -0.4), latents_std=(2.0, 0.5, 1.5, 1.0),
        )
    )
    z = mx.random.normal((1, 2, 2, 2, 4))
    back = vae.denormalize_latents(vae.normalize_latents(z))
    mx.eval(back)
    assert mx.allclose(back, z, atol=1e-5)


def test_converter_registry_lookup():
    conv = get_converter("AutoencoderKLWan")
    assert conv.source_class == "AutoencoderKLWan"
    with pytest.raises(KeyError):
        get_converter("NopeModel")


def test_assert_matches_detects_drift():
    vae = tiny_vae()
    good = dict(__import__("mlx").utils.tree_flatten(vae.parameters()))
    _assert_matches(vae, good)  # exact match: no raise
    bad = dict(good)
    bad.pop(next(iter(bad)))  # drop one key
    with pytest.raises(ValueError, match="missing"):
        _assert_matches(vae, bad)


def test_converter_base_not_implemented():
    with pytest.raises(NotImplementedError):
        Converter().build_config({})
