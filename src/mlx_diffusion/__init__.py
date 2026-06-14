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
from .quantization import quantize_module
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
