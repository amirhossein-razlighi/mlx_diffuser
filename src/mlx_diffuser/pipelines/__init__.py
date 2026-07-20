"""Inference pipelines: containers that wire components into a single ``__call__``."""

from ..models import (
    AutoencoderKL,
    AutoencoderKLVideo,
    AutoencoderKLWan,
    DINOv2Model,
    DiT,
    TrellisGaussianDecoder,
    TrellisSLatFlowModel,
    TrellisSparseStructureDecoder,
    TrellisSparseStructureFlowModel,
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
from .flux import FluxPipeline
from .ltx2 import LTX2Pipeline
from .sdxl import StableDiffusionXLPipeline
from .text_to_video import TextToVideoPipeline
from .trellis import TrellisImageTo3DPipeline, TrellisPipelineOutput
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
    DINOv2Model,
    TrellisSparseStructureFlowModel,
    TrellisSparseStructureDecoder,
    TrellisSLatFlowModel,
    TrellisGaussianDecoder,
)

__all__ = [
    "DiffusionPipeline",
    "ClassConditionalPipeline",
    "TextToVideoPipeline",
    "WanPipeline",
    "StableDiffusionXLPipeline",
    "FluxPipeline",
    "LTX2Pipeline",
    "TrellisImageTo3DPipeline",
    "TrellisPipelineOutput",
    "register_models",
    "register_pipeline",
]
