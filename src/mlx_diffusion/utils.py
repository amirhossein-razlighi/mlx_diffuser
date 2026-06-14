"""Small shared utilities: dtypes, seeding, logging, image <-> array."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np
    from PIL import Image

__all__ = [
    "DTYPES",
    "as_dtype",
    "get_logger",
    "seed_everything",
    "to_pil",
    "to_array",
]

# Canonical string <-> mlx dtype mapping used in configs and CLI args.
DTYPES: dict[str, mx.Dtype] = {
    "float32": mx.float32,
    "fp32": mx.float32,
    "float16": mx.float16,
    "fp16": mx.float16,
    "bfloat16": mx.bfloat16,
    "bf16": mx.bfloat16,
}


def as_dtype(dtype: str | mx.Dtype | None) -> mx.Dtype | None:
    """Resolve a dtype given as a string alias or an ``mx.Dtype``."""
    if dtype is None or isinstance(dtype, mx.Dtype):
        return dtype
    try:
        return DTYPES[dtype.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Unknown dtype {dtype!r}. Choose from {sorted(DTYPES)}."
        ) from exc


_LOG_LEVEL = os.environ.get("MLX_DIFFUSION_LOG", "INFO").upper()


def get_logger(name: str = "mlx_diffusion") -> logging.Logger:
    """Return a configured library logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(_LOG_LEVEL)
        logger.propagate = False
    return logger


def seed_everything(seed: int) -> mx.array:
    """Seed MLX's global RNG and return a fresh key for explicit use."""
    mx.random.seed(seed)
    return mx.random.key(seed)


def to_array(image: "Image.Image | np.ndarray", dtype: mx.Dtype = mx.float32) -> mx.array:
    """Convert a PIL image or HWC uint8/float array to a CHW-free [-1, 1] array.

    Returns shape ``(H, W, C)`` in ``[-1, 1]`` (MLX/`channels-last` convention).
    """
    import numpy as np

    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 127.5 - 1.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return mx.array(arr).astype(dtype)


def to_pil(array: mx.array) -> "Image.Image":
    """Convert a single ``(H, W, C)`` array in ``[-1, 1]`` to a PIL image."""
    import numpy as np
    from PIL import Image

    arr = mx.clip((array + 1.0) * 127.5 + 0.5, 0, 255).astype(mx.uint8)
    np_arr = np.array(arr)
    if np_arr.shape[-1] == 1:
        np_arr = np_arr[..., 0]
    return Image.fromarray(np_arr)
