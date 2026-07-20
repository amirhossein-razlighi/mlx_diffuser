"""Convert external (PyTorch / diffusers) checkpoints into mlx-diffuser models.

External weights are laid out for PyTorch: NCHW/​OIHW conv kernels, and module
names from the original implementation. A *converter* maps that key-space onto one
of our channels-last ``ModelMixin`` networks and rewrites the tensors (e.g. conv
kernels ``(O, I, *k) -> (O, *k, I)``). Converters are registered by source
``_class_name`` so a whole diffusers pipeline folder can be converted in one call.
"""

from __future__ import annotations

from .base import (
    Converter,
    convert_conv_weight,
    get_converter,
    load_safetensors_folder,
    register_converter,
)
from .trellis import (
    convert_trellis_checkpoint,
    convert_trellis_dense_components,
    download_and_convert_trellis,
)

__all__ = [
    "Converter",
    "convert_conv_weight",
    "convert_trellis_checkpoint",
    "convert_trellis_dense_components",
    "download_and_convert_trellis",
    "get_converter",
    "load_safetensors_folder",
    "register_converter",
]

# Import side-effect: register the built-in converters.
from . import flux, ltx2, sdxl, umt5, wan  # noqa: E402,F401
