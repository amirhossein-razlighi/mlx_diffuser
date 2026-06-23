"""Converters for the WAN 2.1 family (diffusers format).

The MLX modules deliberately mirror the diffusers module names, so weight
conversion is almost identity: transpose conv kernels to channels-last and squeeze
the RMSNorm ``gamma`` vectors. The ``resample.1`` index lands naturally because
``WanResample.resample`` is a 2-element ``[op, conv]`` list.
"""

from __future__ import annotations

import mlx.core as mx

from ..models.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanConfig
from ..models.wan_transformer import WanTransformer3DModel, WanTransformerConfig
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


@register_converter
class WanTransformerConverter(Converter):
    source_class = "WanTransformer3DModel"

    def build_config(self, hf_config: dict) -> WanTransformerConfig:
        return WanTransformerConfig(
            patch_size=tuple(hf_config["patch_size"]),
            num_attention_heads=hf_config["num_attention_heads"],
            attention_head_dim=hf_config["attention_head_dim"],
            in_channels=hf_config["in_channels"],
            out_channels=hf_config["out_channels"],
            text_dim=hf_config["text_dim"],
            freq_dim=hf_config["freq_dim"],
            ffn_dim=hf_config["ffn_dim"],
            num_layers=hf_config["num_layers"],
            cross_attn_norm=hf_config["cross_attn_norm"],
            eps=hf_config["eps"],
            rope_max_seq_len=hf_config.get("rope_max_seq_len", 1024),
        )

    def build_model(self, hf_config: dict) -> WanTransformer3DModel:
        return WanTransformer3DModel(self.build_config(hf_config))

    def convert_weights(self, weights: dict[str, mx.array], hf_config: dict) -> dict[str, mx.array]:
        # Only the patch-embedding conv kernel needs reordering; Linear weights are
        # (out, in) in both frameworks, and scale_shift_table / norms map identically.
        out: dict[str, mx.array] = {}
        for key, value in weights.items():
            if key == "patch_embedding.weight":
                value = convert_conv_weight(value)
            out[key] = value
        return out
