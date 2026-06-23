"""Converter registry and shared tensor helpers.

A :class:`Converter` knows how to turn one external component (a diffusers
subfolder such as ``vae/`` or ``transformer/``) into one of our models: it
translates the ``config.json`` into our :class:`~mlx_diffuser.configuration.Config`
and remaps the weight tensors onto the model's channels-last parameter tree.

The heavy lifting is a *build-and-fill* strategy: we instantiate the target model,
read its expected ``(key -> shape)`` tree, and assert every key is filled with a
matching shape. A converter that drifts from the architecture fails loudly here
rather than silently producing noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx.utils import tree_flatten

if TYPE_CHECKING:
    from ..modeling import ModelMixin

__all__ = [
    "Converter",
    "convert_conv_weight",
    "get_converter",
    "load_safetensors_folder",
    "register_converter",
]


def convert_conv_weight(w: mx.array) -> mx.array:
    """Reorder a PyTorch conv kernel to MLX channels-last layout.

    Conv2d ``(O, I, kH, kW) -> (O, kH, kW, I)``; Conv3d ``(O, I, kT, kH, kW) ->
    (O, kT, kH, kW, I)``. Other ranks are returned unchanged.
    """
    if w.ndim == 4:
        return w.transpose(0, 2, 3, 1)
    if w.ndim == 5:
        return w.transpose(0, 2, 3, 4, 1)
    return w


def load_safetensors_folder(folder: str | Path) -> dict[str, mx.array]:
    """Load and merge every ``*.safetensors`` shard in ``folder`` into one dict.

    MLX reads safetensors natively, so no PyTorch dependency is needed to convert
    weights. Sharded checkpoints (``*-00001-of-000NN.safetensors``) are merged.
    """
    folder = Path(folder)
    shards = sorted(folder.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files in {folder}")
    merged: dict[str, mx.array] = {}
    for shard in shards:
        merged.update(mx.load(str(shard)))  # type: ignore[arg-type]  # safetensors -> dict
    return merged


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


class Converter:
    """Base class for one component converter (one diffusers subfolder).

    Subclasses set :attr:`source_class` (the diffusers ``_class_name`` they accept)
    and implement :meth:`build_config` and :meth:`convert_weights`.
    """

    #: diffusers ``_class_name`` this converter consumes.
    source_class: str

    def build_config(self, hf_config: dict):
        """Translate a diffusers ``config.json`` dict into our ``Config``."""
        raise NotImplementedError

    def build_model(self, hf_config: dict) -> ModelMixin:
        """Instantiate the (empty) target model from a diffusers config."""
        raise NotImplementedError

    def convert_weights(self, weights: dict[str, mx.array], hf_config: dict) -> dict[str, mx.array]:
        """Map a diffusers state dict onto our model's parameter keys."""
        raise NotImplementedError

    def convert(self, source_folder: str | Path) -> ModelMixin:
        """Load a diffusers subfolder and return a populated model.

        Builds the target model, converts the weights, then loads them strictly so
        any missing/extra/mis-shaped tensor raises immediately.
        """
        source_folder = Path(source_folder)
        hf_config = _load_json(source_folder / "config.json")
        weights = load_safetensors_folder(source_folder)

        model = self.build_model(hf_config)
        converted = self.convert_weights(weights, hf_config)
        _assert_matches(model, converted)
        model.load_weights(list(converted.items()), strict=True)
        mx.eval(model.parameters())
        return model


def _assert_matches(model: ModelMixin, converted: dict[str, mx.array]) -> None:
    """Check the converted dict exactly covers the model's parameter tree."""
    expected = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}  # type: ignore[union-attr, str-unpack]
    got = {k: tuple(v.shape) for k, v in converted.items()}
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    mismatched = sorted(k for k in expected.keys() & got.keys() if expected[k] != got[k])
    if missing or extra or mismatched:
        lines = []
        if missing:
            lines.append(f"missing {len(missing)} keys, e.g. {missing[:5]}")
        if extra:
            lines.append(f"extra {len(extra)} keys, e.g. {extra[:5]}")
        if mismatched:
            ex = [(k, expected[k], got[k]) for k in mismatched[:5]]
            lines.append(f"shape mismatch on {len(mismatched)}, e.g. {ex}")
        raise ValueError("Converted weights do not match the model:\n  " + "\n  ".join(lines))


_REGISTRY: dict[str, Converter] = {}


def register_converter(converter_cls: type[Converter]) -> type[Converter]:
    """Class decorator: register a converter under its ``source_class``."""
    _REGISTRY[converter_cls.source_class] = converter_cls()
    return converter_cls


def get_converter(source_class: str) -> Converter:
    """Look up a registered converter by diffusers ``_class_name``."""
    if source_class not in _REGISTRY:
        raise KeyError(f"No converter for '{source_class}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[source_class]
