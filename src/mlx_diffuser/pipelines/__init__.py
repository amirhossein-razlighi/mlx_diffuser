"""Inference pipelines: containers that wire components into a single ``__call__``."""

from ..models import AutoencoderKL, DiT, UNet2D
from .base import (
    DiffusionPipeline,
    register_models,
    register_pipeline,
)
from .class_conditional import ClassConditionalPipeline

# Register the models that pipelines can load by class name.
register_models(DiT, UNet2D, AutoencoderKL)

__all__ = [
    "DiffusionPipeline",
    "ClassConditionalPipeline",
    "register_models",
    "register_pipeline",
]
