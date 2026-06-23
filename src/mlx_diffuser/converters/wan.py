"""Converters for the WAN 2.1 family (diffusers format).

The MLX modules deliberately mirror the diffusers module names, so weight
conversion is almost identity: transpose conv kernels to channels-last and squeeze
the RMSNorm ``gamma`` vectors. The ``resample.1`` index lands naturally because
``WanResample.resample`` is a 2-element ``[op, conv]`` list.
"""

from __future__ import annotations

import mlx.core as mx

from ..models.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanConfig
from .base import Converter, convert_conv_weight, register_converter


@register_converter
class WanVAEConverter(Converter):
    source_class = "AutoencoderKLWan"

    def build_config(self, hf_config: dict) -> AutoencoderKLWanConfig:
        return AutoencoderKLWanConfig(
            base_dim=hf_config["base_dim"],
            z_dim=hf_config["z_dim"],
            dim_mult=tuple(hf_config["dim_mult"]),
            num_res_blocks=hf_config["num_res_blocks"],
            attn_scales=tuple(hf_config.get("attn_scales", [])),
            temperal_downsample=tuple(hf_config["temperal_downsample"]),
            latents_mean=tuple(hf_config["latents_mean"]),
            latents_std=tuple(hf_config["latents_std"]),
        )

    def build_model(self, hf_config: dict) -> AutoencoderKLWan:
        return AutoencoderKLWan(self.build_config(hf_config))

    def convert_weights(self, weights: dict[str, mx.array], hf_config: dict) -> dict[str, mx.array]:
        out: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key.endswith(".gamma"):
                value = value.reshape(-1)
            elif key.endswith(".weight") and value.ndim in (4, 5):
                value = convert_conv_weight(value)
            out[key] = value
        return out
