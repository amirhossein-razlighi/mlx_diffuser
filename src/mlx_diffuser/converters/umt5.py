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

    # umT5-xxl shape (the WAN text encoder). Used as defaults when a single-file
    # checkpoint ships without a config.json.
    _FIELDS = (
        "vocab_size",
        "d_model",
        "d_kv",
        "d_ff",
        "num_layers",
        "num_heads",
        "relative_attention_num_buckets",
        "relative_attention_max_distance",
        "layer_norm_epsilon",
    )

    def build_config(self, hf_config: dict) -> UMT5Config:
        overrides = {f: hf_config[f] for f in self._FIELDS if f in hf_config}
        return UMT5Config(**overrides)

    def build_model(self, hf_config: dict) -> UMT5EncoderModel:
        return UMT5EncoderModel(self.build_config(hf_config))

    def convert_weights(self, weights: dict[str, mx.array], hf_config: dict) -> dict[str, mx.array]:
        # All model weights (Linear/Embedding/RMSNorm) already match the MLX layout.
        # Single-file community checkpoints may bundle the tokenizer (``spiece_model``)
        # and metadata — keep only the encoder's parameter tensors.
        return {k: v for k, v in weights.items() if k.startswith(("shared.", "encoder."))}
