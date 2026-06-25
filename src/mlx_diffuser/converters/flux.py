"""Converters for the FLUX.1 components (diffusers / transformers format).

The transformer and T5 encoder are both pure-Linear models, so conversion is almost an
identity remap onto the channels-last parameter tree. The only fix-ups are: dropping
T5's tied ``encoder.embed_tokens`` duplicate (we keep ``shared``), and (for the VAE)
the shared :class:`~mlx_diffuser.converters.sdxl.SDVAEConverter`, which already reads
``shift_factor`` / ``use_quant_conv`` from the config. CLIP reuses the SDXL converter.
"""

from __future__ import annotations

import mlx.core as mx

from ..models.flux_transformer import FluxConfig, FluxTransformer2DModel
from ..models.t5 import T5Config, T5EncoderModel
from .base import Converter, register_converter


@register_converter
class FluxTransformerConverter(Converter):
    source_class = "FluxTransformer2DModel"

    def build_config(self, hf_config: dict) -> FluxConfig:
        return FluxConfig(
            patch_size=hf_config.get("patch_size", 1),
            in_channels=hf_config["in_channels"],
            out_channels=hf_config.get("out_channels"),
            num_layers=hf_config["num_layers"],
            num_single_layers=hf_config["num_single_layers"],
            attention_head_dim=hf_config["attention_head_dim"],
            num_attention_heads=hf_config["num_attention_heads"],
            joint_attention_dim=hf_config["joint_attention_dim"],
            pooled_projection_dim=hf_config["pooled_projection_dim"],
            guidance_embeds=hf_config.get("guidance_embeds", False),
            axes_dims_rope=tuple(hf_config.get("axes_dims_rope", (16, 56, 56))),
        )

    def build_model(self, hf_config: dict) -> FluxTransformer2DModel:
        return FluxTransformer2DModel(self.build_config(hf_config))

    def convert_weights(self, weights: dict[str, mx.array], hf_config: dict) -> dict[str, mx.array]:
        # All parameters are Linear/RMSNorm/Embedding (2-D or 1-D) — identity remap.
        return dict(weights)


@register_converter
class T5EncoderConverter(Converter):
    source_class = "T5EncoderModel"

    def build_config(self, hf_config: dict) -> T5Config:
        return T5Config(
            vocab_size=hf_config["vocab_size"],
            d_model=hf_config["d_model"],
            d_kv=hf_config["d_kv"],
            d_ff=hf_config["d_ff"],
            num_layers=hf_config["num_layers"],
            num_heads=hf_config["num_heads"],
            relative_attention_num_buckets=hf_config.get("relative_attention_num_buckets", 32),
            relative_attention_max_distance=hf_config.get("relative_attention_max_distance", 128),
            layer_norm_epsilon=hf_config.get("layer_norm_epsilon", 1e-6),
        )

    def build_model(self, hf_config: dict) -> T5EncoderModel:
        return T5EncoderModel(self.build_config(hf_config))

    def convert_weights(self, weights: dict[str, mx.array], hf_config: dict) -> dict[str, mx.array]:
        # T5 ties the input embedding: the checkpoint carries both `shared.weight` and a
        # duplicate `encoder.embed_tokens.weight`. Keep one as `shared.weight`, drop the rest.
        out = {k: v for k, v in weights.items() if k != "encoder.embed_tokens.weight"}
        if "shared.weight" not in out and "encoder.embed_tokens.weight" in weights:
            out["shared.weight"] = weights["encoder.embed_tokens.weight"]
        return out
