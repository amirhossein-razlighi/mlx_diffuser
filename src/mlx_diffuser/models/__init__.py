"""Networks: config-driven nn.Modules that predict the diffusion/flow target."""

from .autoencoder_kl import AutoencoderKL, AutoencoderKLConfig, DiagonalGaussian
from .autoencoder_kl_video import AutoencoderKLVideo, AutoencoderKLVideoConfig
from .autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanConfig
from .dit import DiT, DiTConfig
from .unet2d import UNet2D, UNet2DConfig
from .video_dit import VideoDiT, VideoDiTConfig

__all__ = [
    "DiT",
    "DiTConfig",
    "UNet2D",
    "UNet2DConfig",
    "AutoencoderKL",
    "AutoencoderKLConfig",
    "DiagonalGaussian",
    "VideoDiT",
    "VideoDiTConfig",
    "AutoencoderKLVideo",
    "AutoencoderKLVideoConfig",
    "AutoencoderKLWan",
    "AutoencoderKLWanConfig",
]
