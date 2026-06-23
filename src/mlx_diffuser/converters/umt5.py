"""Converter for the umT5 text encoder (transformers ``UMT5EncoderModel``).

The MLX module names mirror the transformers checkpoint, and every tensor is a
Linear/Embedding/RMSNorm weight whose layout already matches MLX — so conversion
is pure identity. The encoder is large (5.6B params); pass ``quantize=4`` to
``convert`` to load it in ~3 GB.
"""

from __future__ import annotations

import mlx.core as mx

from ..models.umt5 import UMT5Config, UMT5EncoderModel
from .base import Converter, register_converter


@register_converter
class UMT5Converter(Converter):
    source_class = "UMT5EncoderModel"

    def build_config(self, hf_config: dict) -> UMT5Config:
        return UMT5Config(
            vocab_size=hf_config["vocab_size"],
            d_model=hf_config["d_model"],
            d_kv=hf_config["d_kv"],
            d_ff=hf_config["d_ff"],
            num_layers=hf_config["num_layers"],
            num_heads=hf_config["num_heads"],
            relative_attention_num_buckets=hf_config["relative_attention_num_buckets"],
            relative_attention_max_distance=hf_config["relative_attention_max_distance"],
            layer_norm_epsilon=hf_config["layer_norm_epsilon"],
        )

    def build_model(self, hf_config: dict) -> UMT5EncoderModel:
        return UMT5EncoderModel(self.build_config(hf_config))

    def convert_weights(self, weights: dict[str, mx.array], hf_config: dict) -> dict[str, mx.array]:
        # All weights (Linear, Embedding, RMSNorm) already match the MLX layout.
        return dict(weights)
