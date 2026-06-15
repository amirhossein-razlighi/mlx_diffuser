"""DiffusionPipeline: a thin container that wires components together for inference.

A pipeline is just a set of named components (models + a scheduler, optionally a
VAE/conditioner) plus a ``__call__``. Persistence mirrors diffusers: a
``model_index.json`` records each component's subfolder and class, and every
component is saved into its own subfolder. There is deliberately no deep class
hierarchy — concrete pipelines subclass this and implement ``__call__``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..hub import resolve
from ..modeling import ModelMixin
from ..schedulers import Scheduler, load_scheduler

MODEL_INDEX_NAME = "model_index.json"

#: class name -> ModelMixin subclass (filled by register_models()).
MODEL_REGISTRY: dict[str, type[ModelMixin]] = {}
#: class name -> DiffusionPipeline subclass.
PIPELINE_REGISTRY: dict[str, type[DiffusionPipeline]] = {}


def register_models(*classes: type[ModelMixin]) -> None:
    for c in classes:
        MODEL_REGISTRY[c.__name__] = c


def register_pipeline(cls: type[DiffusionPipeline]) -> type[DiffusionPipeline]:
    PIPELINE_REGISTRY[cls.__name__] = cls
    return cls


def _load_component(class_name: str, path: Path, dtype, quantize) -> Any:
    if class_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[class_name].from_pretrained(path, dtype=dtype, quantize=quantize)
    # Otherwise assume a scheduler (selected from its own config tag).
    return load_scheduler(path)


class DiffusionPipeline:
    #: Names of attributes that are persisted as components.
    _component_names: tuple[str, ...] = ()

    def __init__(self, **components: Any):
        for name, value in components.items():
            setattr(self, name, value)

    # --- persistence ------------------------------------------------------
    def save_pretrained(self, save_directory: str | Path) -> Path:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        index: dict[str, Any] = {"_class_name": type(self).__name__, "_components": {}}
        for name in self._component_names:
            comp = getattr(self, name)
            comp.save_pretrained(save_directory / name)
            index["_components"][name] = type(comp).__name__
        (save_directory / MODEL_INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=True))
        return save_directory

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *,
        dtype: str | mx.Dtype | None = None,
        quantize: int | None = None,
        revision: str | None = None,
    ) -> DiffusionPipeline:
        local = resolve(path_or_repo_id, revision=revision)
        index = json.loads((local / MODEL_INDEX_NAME).read_text())

        target: type[DiffusionPipeline] = cls
        if cls is DiffusionPipeline:  # dispatch to the concrete pipeline
            resolved = PIPELINE_REGISTRY.get(index["_class_name"])
            if resolved is None:
                raise ValueError(f"Unknown pipeline {index['_class_name']!r}. Is it registered?")
            target = resolved

        components = {
            name: _load_component(class_name, local / name, dtype, quantize)
            for name, class_name in index["_components"].items()
        }
        return target(**components)

    # --- helpers for subclasses ------------------------------------------
    @staticmethod
    def _resolve_key(seed: int | None, key: mx.array | None) -> mx.array:
        if key is not None:
            return key
        return mx.random.key(seed if seed is not None else 0)

    @staticmethod
    def classifier_free_guidance(cond: mx.array, uncond: mx.array, scale: float) -> mx.array:
        return uncond + scale * (cond - uncond)

    def denoising_loop(
        self,
        scheduler: Scheduler,
        latents: mx.array,
        predict: Callable[[mx.array, mx.array], mx.array],
        key: mx.array,
        progress: bool = False,
    ) -> mx.array:
        """Run the reverse process, calling ``predict(scaled_latents, t)`` per step.

        ``predict`` returns the (already guidance-combined) model output. Evaluation
        is forced once per step to keep the lazy graph bounded.
        """
        steps = scheduler.timesteps
        assert steps is not None, "call scheduler.set_timesteps() before sampling"
        if progress:
            from ..utils import get_logger

            get_logger().info("sampling %d steps", len(steps))
        for t in steps:
            scaled = scheduler.scale_model_input(latents, t)
            model_output = predict(scaled, t)
            key, step_key = mx.random.split(key)
            latents = scheduler.step(model_output, t, latents, key=step_key)
            mx.eval(latents)
        return latents
