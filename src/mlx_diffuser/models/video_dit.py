"""Video Diffusion Transformer (the backbone of LTX-Video and the WAN series).

A spatiotemporal DiT: a video latent ``(B, T, H, W, C)`` is patchified over all
three axes into tokens, processed by transformer blocks that combine adaLN-Zero
timestep conditioning, 3D-RoPE self-attention, and cross-attention to text, then
unpatchified back into a latent. It pairs with the flow-matching scheduler and the
3D video VAE to form a text-to-video pipeline.

The ``*_config`` classmethods return presets whose shapes mirror the published
LTX-Video / WAN architectures, so you can instantiate, benchmark, and quantize
those model sizes from scratch. Loading the official pretrained weights is a
separate step (a checkpoint converter) and is not included here.

Tensors are channels-last: input/output are ``(B, T, H, W, C)``.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx

from ..configuration import Config
from ..layers.blocks import FinalLayer
from ..layers.embeddings import TimestepEmbedder
from ..layers.rope import rope_3d_freqs
from ..layers.video_blocks import PatchEmbed3D, VideoDiTBlock
from ..modeling import ModelMixin


@dataclasses.dataclass
class VideoDiTConfig(Config):
    in_channels: int = 16  # latent channels of the video VAE
    out_channels: int | None = None
    patch_size: tuple[int, int, int] = (1, 2, 2)  # (time, height, width)
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    cross_attn_dim: int | None = 4096  # text-embedding dim (T5); None => unconditional
    rope_theta: float = 10000.0
    qk_norm: bool = True

    @property
    def resolved_out_channels(self) -> int:
        return self.out_channels if self.out_channels is not None else self.in_channels

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    # --- presets (architecture shapes of well-known video models) ---------
    @classmethod
    def wan_t2v_1_3b(cls) -> VideoDiTConfig:
        """WAN 2.1 text-to-video 1.3B shape (16-channel latents, umT5 conditioning)."""
        return cls(
            in_channels=16,
            patch_size=(1, 2, 2),
            hidden_size=1536,
            depth=30,
            num_heads=12,
            mlp_ratio=8960 / 1536,
            cross_attn_dim=4096,
        )

    @classmethod
    def wan_t2v_14b(cls) -> VideoDiTConfig:
        """WAN 2.1 text-to-video 14B shape."""
        return cls(
            in_channels=16,
            patch_size=(1, 2, 2),
            hidden_size=5120,
            depth=40,
            num_heads=40,
            mlp_ratio=13824 / 5120,
            cross_attn_dim=4096,
        )

    @classmethod
    def ltx_video(cls) -> VideoDiTConfig:
        """LTX-Video shape (128-channel highly-compressed latents, T5-XXL conditioning)."""
        return cls(
            in_channels=128,
            patch_size=(1, 1, 1),
            hidden_size=2048,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            cross_attn_dim=4096,
        )


class VideoDiT(ModelMixin[VideoDiTConfig]):
    config_class = VideoDiTConfig

    def __init__(self, config: VideoDiTConfig):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if config.head_dim % 2 != 0:
            raise ValueError("head_dim (hidden_size / num_heads) must be even for RoPE.")

        self.patch_embed = PatchEmbed3D(config.in_channels, config.hidden_size, config.patch_size)
        self.t_embed = TimestepEmbedder(config.hidden_size)
        self.blocks = [
            VideoDiTBlock(
                config.hidden_size,
                config.num_heads,
                config.mlp_ratio,
                cross_attn_dim=config.cross_attn_dim,
                qk_norm=config.qk_norm,
            )
            for _ in range(config.depth)
        ]
        pt, ph, pw = config.patch_size
        patch_dim = pt * ph * pw * config.resolved_out_channels
        self.final = FinalLayer(config.hidden_size, patch_dim)

    def _unpatchify(self, x: mx.array, t: int, h: int, w: int) -> mx.array:
        pt, ph, pw = self.config.patch_size
        c = self.config.resolved_out_channels
        b = x.shape[0]
        x = x.reshape(b, t, h, w, pt, ph, pw, c)
        x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)  # (b, t, pt, h, ph, w, pw, c)
        return x.reshape(b, t * pt, h * ph, w * pw, c)

    def __call__(
        self,
        x: mx.array,
        t: mx.array,
        context: mx.array | None = None,
        *,
        training: bool = False,
    ) -> mx.array:
        """Predict the flow/noise target for video latents ``x`` at timestep ``t``.

        Args:
            x: video latents ``(B, T, H, W, C)``.
            t: per-sample timestep / sigma ``(B,)``.
            context: per-token text embeddings ``(B, L, cross_attn_dim)`` for
                cross-attention, or ``None`` for unconditional.
        """
        tokens, (gt, gh, gw) = self.patch_embed(x)
        cos, sin = rope_3d_freqs(self.config.head_dim, gt, gh, gw, self.config.rope_theta)
        rope = (cos[None, None].astype(tokens.dtype), sin[None, None].astype(tokens.dtype))

        c = self.t_embed(t)
        for block in self.blocks:
            tokens = block(tokens, c, rope, context=context)

        tokens = self.final(tokens, c)
        return self._unpatchify(tokens, gt, gh, gw)
