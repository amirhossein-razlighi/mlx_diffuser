"""Inference pipelines: containers that wire components into a single ``__call__``."""

from ..models import (
    AutoencoderKL,
    AutoencoderKLVideo,
    AutoencoderKLWan,
    DiT,
    UNet2D,
    VideoDiT,
    WanTransformer3DModel,
)
from .base import (
    DiffusionPipeline,
    register_models,
    register_pipeline,
)
from .class_conditional import ClassConditionalPipeline
from .sdxl import StableDiffusionXLPipeline
from .text_to_video import TextToVideoPipeline
from .wan import WanPipeline

# Register the models that pipelines can load by class name.
register_models(
    DiT,
    UNet2D,
    AutoencoderKL,
    VideoDiT,
    AutoencoderKLVideo,
    AutoencoderKLWan,
    WanTransformer3DModel,
)

__all__ = [
    "DiffusionPipeline",
    "ClassConditionalPipeline",
    "TextToVideoPipeline",
    "WanPipeline",
    "StableDiffusionXLPipeline",
    "register_models",
    "register_pipeline",
]
