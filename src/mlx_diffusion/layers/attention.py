"""Multi-head attention built on MLX's fused scaled-dot-product kernel."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class Attention(nn.Module):
    """Multi-head attention (self- or cross-attention).

    Channels-last tokens ``(B, T, dim)``. When ``context`` is given, keys/values
    come from it (cross-attention); otherwise from ``x`` (self-attention).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        context_dim: int | None = None,
        qkv_bias: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        context_dim = context_dim or dim

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else None

    def _split_heads(self, x: mx.array) -> mx.array:
        b, t, _ = x.shape
        return x.reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, context: mx.array | None = None, mask: mx.array | None = None) -> mx.array:
        kv = x if context is None else context
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(kv))
        v = self._split_heads(self.v_proj(kv))

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        b, _, t, _ = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(b, t, self.num_heads * self.head_dim)
        return self.out_proj(out)
