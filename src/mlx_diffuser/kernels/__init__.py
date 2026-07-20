"""Optional custom Metal kernels with portable MLX fallbacks."""

from .sparse import sparse_conv3d

__all__ = ["sparse_conv3d"]
