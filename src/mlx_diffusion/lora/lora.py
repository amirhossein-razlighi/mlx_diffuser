"""LoRA: low-rank adapters for parameter-efficient fine-tuning.

``inject_lora`` swaps matching ``nn.Linear`` layers for :class:`LoRALinear` (base
frozen, low-rank ``A``/``B`` trainable). Adapters can be trained with the normal
DiffusionTrainer, saved/loaded standalone, or merged back into dense weights for
zero-overhead inference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterator

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

DEFAULT_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"


class LoRALinear(nn.Module):
    """Wraps a (frozen) ``nn.Linear`` with a trainable low-rank update."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        out_features, in_features = linear.weight.shape
        self.rank = rank
        self.scale = alpha / rank
        bound = 1.0 / math.sqrt(in_features)
        self.lora_a = mx.random.uniform(low=-bound, high=bound, shape=(rank, in_features))
        self.lora_b = mx.zeros((out_features, rank))  # zero-init => identity at start
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        base = self.linear(x)
        z = self.dropout(x) if self.dropout is not None else x
        delta = (z @ self.lora_a.T) @ self.lora_b.T
        return base + self.scale * delta

    def fused(self) -> nn.Linear:
        """Return a dense ``nn.Linear`` with the adapter merged in."""
        out_features, in_features = self.linear.weight.shape
        has_bias = "bias" in self.linear
        merged = nn.Linear(in_features, out_features, bias=has_bias)
        merged.weight = self.linear.weight + self.scale * (self.lora_b @ self.lora_a)
        if has_bias:
            merged.bias = self.linear.bias
        return merged


def _replace_modules(module: nn.Module, predicate: Callable, make: Callable) -> None:
    for key, value in list(module.items()):
        if isinstance(value, nn.Module) and predicate(key, value):
            setattr(module, key, make(value))
        elif isinstance(value, nn.Module):
            _replace_modules(value, predicate, make)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, nn.Module):
                    _replace_modules(item, predicate, make)


def _iter_lora(module: nn.Module) -> Iterator[LoRALinear]:
    for value in module.values():
        if isinstance(value, LoRALinear):
            yield value
        elif isinstance(value, nn.Module):
            yield from _iter_lora(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, nn.Module):
                    yield from _iter_lora(item)


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
) -> int:
    """Replace target ``nn.Linear`` layers with LoRA adapters and freeze the base.

    Returns the number of layers adapted. After this, ``trainable_parameters``
    contains only the adapter weights.
    """
    count = 0

    def predicate(key: str, value: nn.Module) -> bool:
        return isinstance(value, nn.Linear) and key in targets

    def make(linear: nn.Linear) -> LoRALinear:
        nonlocal count
        count += 1
        return LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)

    _replace_modules(model, predicate, make)
    model.freeze()
    for lora in _iter_lora(model):
        lora.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    return count


def merge_lora(model: nn.Module) -> nn.Module:
    """Fuse all adapters into dense layers in place; returns the model."""
    _replace_modules(
        model,
        predicate=lambda key, value: isinstance(value, LoRALinear),
        make=lambda lora: lora.fused(),
    )
    model.unfreeze()
    return model


def lora_state_dict(model: nn.Module) -> dict[str, mx.array]:
    """The adapter weights (the model's trainable parameters after injection)."""
    return dict(tree_flatten(model.trainable_parameters()))


def save_lora(
    model: nn.Module,
    save_directory: str | Path,
    *,
    rank: int,
    alpha: float,
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS,
    dropout: float = 0.0,
) -> Path:
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    config = {"rank": rank, "alpha": alpha, "targets": list(targets), "dropout": dropout}
    (save_directory / ADAPTER_CONFIG_NAME).write_text(json.dumps(config, indent=2, sort_keys=True))
    mx.save_safetensors(str(save_directory / ADAPTER_WEIGHTS_NAME), lora_state_dict(model))
    return save_directory


def load_lora(model: nn.Module, path: str | Path) -> int:
    """Inject adapters per ``adapter_config.json`` and load their weights."""
    path = Path(path)
    config = json.loads((path / ADAPTER_CONFIG_NAME).read_text())
    count = inject_lora(
        model,
        rank=config["rank"],
        alpha=config["alpha"],
        dropout=config.get("dropout", 0.0),
        targets=tuple(config["targets"]),
    )
    weights = mx.load(str(path / ADAPTER_WEIGHTS_NAME))
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return count
