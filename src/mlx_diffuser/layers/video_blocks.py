"""Building blocks for video diffusion transformers (LTX-Video / WAN style).

Tokens are spatiotemporal: a video ``(B, T, H, W, C)`` is patchified over all
three axes into a flat token sequence, processed by transformer blocks that mix
adaLN-Zero time conditioning, RoPE self-attention, and (optional) cross-attention
to text, then unpatchified back to a video.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .attention import Attention
from .blocks import FeedForward
from .normalization import AdaLNModulation, modulate


class PatchEmbed3D(nn.Module):
    """Patchify ``(B, T, H, W, C)`` into tokens via a strided 3D conv.

    ``patch_size`` is ``(pt, ph, pw)``. Returns ``(B, N, hidden)`` tokens plus the
    token grid ``(t, h, w)`` so positions and unpatchify can be reconstructed.
    """

    def __init__(self, in_channels: int, hidden_size: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> tuple[mx.array, tuple[int, int, int]]:
        x = self.proj(x)
        b, t, h, w, c = x.shape
        return x.reshape(b, t * h * w, c), (t, h, w)


class VideoDiTBlock(nn.Module):
    """Transformer block: adaLN-Zero self-attention (+RoPE), text cross-attention, MLP.

    Conditioning ``c`` (timestep, optionally pooled text) drives adaLN shift/scale/
    gate for the self-attention and MLP sublayers. When ``cross_attn_dim`` is set,
    a gated cross-attention to ``context`` (per-token text embeddings) is inserted;
    its gate is zero-initialized so an untrained block is the identity.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        cross_attn_dim: int | None = None,
        cond_dim: int | None = None,
        qk_norm: bool = True,
    ):
        super().__init__()
        cond_dim = cond_dim or dim
        self.norm1 = nn.LayerNorm(dim, affine=False)
        self.attn = Attention(dim, num_heads, qk_norm=qk_norm)
        self.modulation = AdaLNModulation(cond_dim, dim, n=6)

        if cross_attn_dim is not None:
            self.norm_cross = nn.LayerNorm(dim, affine=False)
            self.cross_attn = Attention(dim, num_heads, context_dim=cross_attn_dim, qk_norm=qk_norm)
            self.cross_gate = mx.zeros((dim,))  # zero-init -> identity at start
        else:
            self.norm_cross = None
            self.cross_attn = None

        self.norm2 = nn.LayerNorm(dim, affine=False)
        self.mlp = FeedForward(dim, mlp_ratio)

    def __call__(
        self,
        x: mx.array,
        c: mx.array,
        rope: tuple[mx.array, mx.array],
        context: mx.array | None = None,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(c)
        x = x + gate_msa[:, None, :] * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope=rope
        )
        if self.cross_attn is not None and self.norm_cross is not None and context is not None:
            x = x + self.cross_gate * self.cross_attn(self.norm_cross(x), context=context)
        x = x + gate_mlp[:, None, :] * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
