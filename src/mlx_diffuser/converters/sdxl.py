"""Converters for the Stable Diffusion XL components (diffusers/transformers format).

Each converter turns one component folder of an ``stable-diffusion-xl-base`` checkpoint
into the matching MLX model. The CLIP text encoders convert nearly identity (module
names mirror transformers); the VAE and UNet need conv kernels reordered to
channels-last.
"""

from __future__ import annotations

import mlx.core as mx

from ..models.clip_text import CLIPTextConfig, CLIPTextModel
from .base import Converter, register_converter


class _CLIPConverter(Converter):
    with_projection = False

    def build_config(self, hf_config: dict) -> CLIPTextConfig:
        return CLIPTextConfig(
            vocab_size=hf_config["vocab_size"],
            hidden_size=hf_config["hidden_size"],
            intermediate_size=hf_config["intermediate_size"],
            num_hidden_layers=hf_config["num_hidden_layers"],
            num_attention_heads=hf_config["num_attention_heads"],
            max_position_embeddings=hf_config["max_position_embeddings"],
            hidden_act=hf_config["hidden_act"],
            projection_dim=hf_config.get("projection_dim", hf_config["hidden_size"]),
            layer_norm_eps=hf_config.get("layer_norm_eps", 1e-5),
            with_projection=self.with_projection,
        )

    def build_model(self, hf_config: dict) -> CLIPTextModel:
        return CLIPTextModel(self.build_config(hf_config))

    def convert_weights(self, weights: dict[str, mx.array], hf_config: dict) -> dict[str, mx.array]:
        # Identity, minus registered buffers (position_ids) that aren't parameters.
        return {k: v for k, v in weights.items() if not k.endswith("position_ids")}


@register_converter
class CLIPTextModelConverter(_CLIPConverter):
    source_class = "CLIPTextModel"
    with_projection = False


@register_converter
class CLIPTextModelWithProjectionConverter(_CLIPConverter):
    source_class = "CLIPTextModelWithProjection"
    with_projection = True
