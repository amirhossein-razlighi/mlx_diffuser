"""mlx-diffusion: diffusion & flow models on Apple silicon, powered by MLX."""

from __future__ import annotations

from .configuration import Config
from .modeling import ModelMixin
from .models import (
    AutoencoderKL,
    AutoencoderKLConfig,
    DiT,
    DiTConfig,
    UNet2D,
    UNet2DConfig,
)
from .pipelines import ClassConditionalPipeline, DiffusionPipeline
from .quantization import quantize_module
from .lora import inject_lora, load_lora, merge_lora, save_lora
from .perf import compile_model, memory_report, set_memory_limit
from .training import DiffusionTrainer, EMA
from .schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerScheduler,
    Scheduler,
    load_scheduler,
)
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
    "DiffusionPipeline",
    "ClassConditionalPipeline",
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
