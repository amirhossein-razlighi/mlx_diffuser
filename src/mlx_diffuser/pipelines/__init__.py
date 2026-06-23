"""Inference pipelines: containers that wire components into a single ``__call__``."""

from ..models import AutoencoderKL, AutoencoderKLVideo, DiT, UNet2D, VideoDiT
from .base import (
    DiffusionPipeline,
    register_models,
    register_pipeline,
)
from .class_conditional import ClassConditionalPipeline
from .text_to_video import TextToVideoPipeline

# Register the models that pipelines can load by class name.
register_models(DiT, UNet2D, AutoencoderKL, VideoDiT, AutoencoderKLVideo)

__all__ = [
    "DiffusionPipeline",
    "ClassConditionalPipeline",
    "TextToVideoPipeline",
    "register_models",
    "register_pipeline",
]
