"""Multi-head attention built on MLX's fused scaled-dot-product kernel."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotary embedding on ``(B, heads, T, head_dim)`` with ``(.., T, head_dim)`` cos/sin."""
    return x * cos + _rotate_half(x) * sin


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

    def __call__(
        self,
        x: mx.array,
        context: mx.array | None = None,
        mask: mx.array | None = None,
        rope: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        """Attend over ``x`` (``(B, T, dim)``).

        ``context`` switches to cross-attention. ``rope`` is an optional
        ``(cos, sin)`` pair broadcastable to ``(B, heads, T, head_dim)`` that
        rotates the queries/keys before the dot product (self-attention only —
        position has no meaning across the query/context boundary).
        """
        kv = x if context is None else context
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(kv))
        v = self._split_heads(self.v_proj(kv))

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope is not None and context is None:
            cos, sin = rope
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        b, _, t, _ = out.shape
        out = out.transpose(0, 2, 1, 3).reshape(b, t, self.num_heads * self.head_dim)
        return self.out_proj(out)
