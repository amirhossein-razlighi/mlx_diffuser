"""mlx-diffuser: diffusion & flow models on Apple silicon, powered by MLX."""

from __future__ import annotations

from .configuration import Config
from .lora import inject_lora, load_lora, merge_lora, save_lora
from .modeling import ModelMixin
from .models import (
    AutoencoderKL,
    AutoencoderKLConfig,
    AutoencoderKLVideo,
    AutoencoderKLVideoConfig,
    AutoencoderKLWan,
    AutoencoderKLWanConfig,
    DiT,
    DiTConfig,
    UNet2D,
    UNet2DConfig,
    VideoDiT,
    VideoDiTConfig,
    WanTransformer3DModel,
    WanTransformerConfig,
)
from .models.umt5 import UMT5Config, UMT5EncoderModel
from .perf import compile_model, memory_report, set_memory_limit
from .pipelines import (
    ClassConditionalPipeline,
    DiffusionPipeline,
    FluxPipeline,
    LTX2Pipeline,
    StableDiffusionXLPipeline,
    TextToVideoPipeline,
    WanPipeline,
)
from .quantization import quantize_module
from .schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerScheduler,
    Scheduler,
    load_scheduler,
)
from .training import EMA, DiffusionTrainer
from .utils import as_dtype, get_logger, seed_everything, to_array, to_pil
from .version import __version__

__all__ = [
    "Config",
    "ModelMixin",
    "DiT",
    "DiTConfig",
    "UNet2D",
    "UNet2DConfig",
    "AutoencoderKL",
    "AutoencoderKLConfig",
    "VideoDiT",
    "VideoDiTConfig",
    "AutoencoderKLVideo",
    "AutoencoderKLVideoConfig",
    "WanTransformer3DModel",
    "WanTransformerConfig",
    "AutoencoderKLWan",
    "AutoencoderKLWanConfig",
    "UMT5EncoderModel",
    "UMT5Config",
    "DiffusionPipeline",
    "ClassConditionalPipeline",
    "TextToVideoPipeline",
    "WanPipeline",
    "StableDiffusionXLPipeline",
    "FluxPipeline",
    "LTX2Pipeline",
    "DiffusionTrainer",
    "EMA",
    "inject_lora",
    "merge_lora",
    "save_lora",
    "load_lora",
    "compile_model",
    "memory_report",
    "set_memory_limit",
    "quantize_module",
    "Scheduler",
    "DDPMScheduler",
    "DDIMScheduler",
    "EulerDiscreteScheduler",
    "FlowMatchEulerScheduler",
    "load_scheduler",
    "as_dtype",
    "get_logger",
    "seed_everything",
    "to_array",
    "to_pil",
    "__version__",
]
