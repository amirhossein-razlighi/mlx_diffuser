"""Networks: config-driven nn.Modules that predict the diffusion/flow target."""

from .dit import DiT, DiTConfig

__all__ = ["DiT", "DiTConfig"]
