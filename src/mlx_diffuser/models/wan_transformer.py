"""WanTransformer3DModel: a faithful MLX port of WAN 2.1's text-to-video DiT.

Mirrors diffusers' ``WanTransformer3DModel`` module-for-module so the official
weights load via the converter. A video latent ``(B, T, H, W, C)`` is patch-embedded
with a strided 3D conv into tokens, processed by transformer blocks that combine
adaLN modulation (a shared time projection plus a per-block ``scale_shift_table``),
interleaved 3D-RoPE self-attention, and cross-attention to text, then unpatchified.

Normalizations follow the reference's FP32 policy: the modulation arithmetic and
LayerNorms are computed in float32 and cast back, which matters when running in
bf16. Tensors are channels-last.
"""

from __future__ import annotations

import dataclasses
import math

import mlx.core as mx
import mlx.nn as nn

from ..configuration import Config
from ..modeling import ModelMixin


@dataclasses.dataclass
class WanTransformerConfig(Config):
    patch_size: tuple[int, int, int] = (1, 2, 2)
    num_attention_heads: int = 12
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 8960
    num_layers: int = 30
    cross_attn_norm: bool = True
    eps: float = 1e-6
    rope_max_seq_len: int = 1024
    theta: float = 10000.0

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @classmethod
    def wan_t2v_1_3b(cls) -> WanTransformerConfig:
        return cls()  # defaults match the 1.3B model


# --- embeddings & rope -------------------------------------------------------


def _sinusoidal_timestep(t: mx.array, dim: int, max_period: float = 10000.0) -> mx.array:
    """diffusers ``get_timestep_embedding`` with flip_sin_to_cos, shift 0 -> (B, dim)."""
    half = dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
    emb = t.astype(mx.float32)[:, None] * mx.exp(exponent)[None]
    return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)


class WanRotaryPosEmbed:
    """Interleaved (GPT-J style) 3D RoPE; per-axis pair dims sum to ``head_dim``.

    Returns ``(cos, sin)`` of shape ``(1, num_tokens, 1, head_dim // 2)`` for the
    token grid ``(f, h, w)`` in row-major order — the order patch tokens are
    flattened in. Cached per grid; the frequencies are constants, not parameters.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        h_dim = w_dim = 2 * (head_dim // 6)
        t_dim = head_dim - h_dim - w_dim
        self.dims = (t_dim, h_dim, w_dim)
        self.theta = theta
        self._cache: dict[tuple[int, int, int], tuple[mx.array, mx.array]] = {}

    def _axis_inv_freq(self, dim: int) -> mx.array:
        return 1.0 / (self.theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, frames: int, height: int, width: int) -> tuple[mx.array, mx.array]:
        key = (frames, height, width)
        if key in self._cache:
            return self._cache[key]
        invf = [self._axis_inv_freq(d) for d in self.dims]
        f = mx.arange(frames, dtype=mx.float32)[:, None] * invf[0][None]  # (F, t_dim/2)
        h = mx.arange(height, dtype=mx.float32)[:, None] * invf[1][None]  # (H, h_dim/2)
        w = mx.arange(width, dtype=mx.float32)[:, None] * invf[2][None]  # (W, w_dim/2)
        # broadcast to (F, H, W, dim/2) per axis then concat -> (F,H,W, head_dim/2)
        F_, H_, W_ = frames, height, width
        fa = mx.broadcast_to(f[:, None, None, :], (F_, H_, W_, f.shape[-1]))
        ha = mx.broadcast_to(h[None, :, None, :], (F_, H_, W_, h.shape[-1]))
        wa = mx.broadcast_to(w[None, None, :, :], (F_, H_, W_, w.shape[-1]))
        ang = mx.concatenate([fa, ha, wa], axis=-1).reshape(F_ * H_ * W_, -1)
        cos = mx.cos(ang)[None, :, None, :]
        sin = mx.sin(ang)[None, :, None, :]
        self._cache[key] = (cos, sin)
        return cos, sin


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Interleaved RoPE on ``(B, L, heads, head_dim)`` with ``(1, L, 1, head_dim/2)``."""
    xr = x.reshape(*x.shape[:-1], -1, 2)
    x_even, x_odd = xr[..., 0], xr[..., 1]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return mx.stack([out_even, out_odd], axis=-1).reshape(x.shape)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class PixArtTextProjection(nn.Module):
    """Two-layer text projection with tanh-approx GELU (diffusers PixArtAlpha)."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.gelu_approx(self.linear_1(x)))


class WanConditionEmbedder(nn.Module):
    def __init__(self, cfg: WanTransformerConfig):
        super().__init__()
        d = cfg.inner_dim
        self.time_embedder = TimestepEmbedding(cfg.freq_dim, d)
        self.time_proj = nn.Linear(d, d * 6)
        self.text_embedder = PixArtTextProjection(cfg.text_dim, d)
        self._freq_dim = cfg.freq_dim

    def __call__(
        self, timestep: mx.array, context: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        proj = _sinusoidal_timestep(timestep, self._freq_dim)
        temb = self.time_embedder(proj)  # (B, d)
        timestep_proj = self.time_proj(nn.silu(temb))  # (B, 6d)
        context = self.text_embedder(context)
        return temb, timestep_proj, context


# --- attention & block -------------------------------------------------------


class WanAttention(nn.Module):
    """Multi-head attention with qk-RMSNorm across the full dim; optional RoPE."""

    def __init__(self, dim: int, heads: int, eps: float, cross_attention: bool):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.cross_attention = cross_attention
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = [nn.Linear(dim, dim)]  # diffusers ModuleList [Linear, Dropout]
        self.norm_q = nn.RMSNorm(dim, eps=eps)
        self.norm_k = nn.RMSNorm(dim, eps=eps)

    def __call__(self, x, context=None, rope=None) -> mx.array:
        kv = x if context is None else context
        b, lq = x.shape[0], x.shape[1]
        q = self.norm_q(self.to_q(x))
        k = self.norm_k(self.to_k(kv))
        v = self.to_v(kv)
        q = q.reshape(b, lq, self.heads, self.head_dim)
        k = k.reshape(b, kv.shape[1], self.heads, self.head_dim)
        v = v.reshape(b, kv.shape[1], self.heads, self.head_dim)
        if rope is not None:
            q = _apply_rope(q, *rope)
            k = _apply_rope(k, *rope)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
        out = out.transpose(0, 2, 1, 3).reshape(b, lq, self.heads * self.head_dim)
        return self.to_out[0](out)


class WanFeedForward(nn.Module):
    """diffusers FeedForward(gelu-approximate): net.0.proj -> gelu_tanh -> net.2."""

    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.net = [_GELUProj(dim, ffn_dim), _Identity(), nn.Linear(ffn_dim, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.net[2](self.net[0](x))


class _GELUProj(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu_approx(self.proj(x))


class _Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


def _layernorm_f32(x: mx.array, eps: float) -> mx.array:
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.var(xf, axis=-1, keepdims=True)
    return (xf - mean) * mx.rsqrt(var + eps)


class WanTransformerBlock(nn.Module):
    def __init__(self, cfg: WanTransformerConfig):
        super().__init__()
        d, heads, eps = cfg.inner_dim, cfg.num_attention_heads, cfg.eps
        self.attn1 = WanAttention(d, heads, eps, cross_attention=False)
        self.attn2 = WanAttention(d, heads, eps, cross_attention=True)
        self.norm2 = nn.LayerNorm(d, eps=eps) if cfg.cross_attn_norm else _Identity()
        self.ffn = WanFeedForward(d, cfg.ffn_dim)
        self.scale_shift_table = mx.zeros((1, 6, d))
        self.eps = eps

    def __call__(self, x, context, timestep_proj, rope) -> mx.array:
        # timestep_proj: (B, 6, d); broadcast modulation over tokens
        mod = self.scale_shift_table + timestep_proj.astype(mx.float32)
        shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = (
            mod[:, i : i + 1] for i in range(6)
        )
        # 1. self-attention
        norm_x = _layernorm_f32(x, self.eps) * (1 + scale_msa) + shift_msa
        x = x.astype(mx.float32) + self.attn1(norm_x.astype(x.dtype), rope=rope) * gate_msa
        x = x.astype(context.dtype)
        # 2. cross-attention
        norm_x = self.norm2(x)
        x = x + self.attn2(norm_x, context=context)
        # 3. feed-forward
        norm_x = _layernorm_f32(x, self.eps) * (1 + c_scale) + c_shift
        x = x.astype(mx.float32) + self.ffn(norm_x.astype(x.dtype)).astype(mx.float32) * c_gate
        return x.astype(context.dtype)


class WanTransformer3DModel(ModelMixin[WanTransformerConfig]):
    """WAN 2.1 text-to-video diffusion transformer. Channels-last ``(B, T, H, W, C)``."""

    config_class = WanTransformerConfig

    def __init__(self, config: WanTransformerConfig):
        super().__init__()
        self.config = config
        d = config.inner_dim
        self.rope = WanRotaryPosEmbed(config.attention_head_dim, config.theta)
        self.patch_embedding = nn.Conv3d(
            config.in_channels, d, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.condition_embedder = WanConditionEmbedder(config)
        self.blocks = [WanTransformerBlock(config) for _ in range(config.num_layers)]
        self.norm_out = nn.LayerNorm(d, eps=config.eps, affine=False)
        self.proj_out = nn.Linear(d, config.out_channels * math.prod(config.patch_size))
        self.scale_shift_table = mx.zeros((1, 2, d))

    def __call__(self, x: mx.array, timestep: mx.array, context: mx.array) -> mx.array:
        """Predict the flow target for latents ``x`` ``(B, T, H, W, C)`` at ``timestep``.

        ``context`` is per-token text embeddings ``(B, L, text_dim)`` (umT5).
        """
        b, t, h, w, _ = x.shape
        pt, ph, pw = self.config.patch_size
        gt, gh, gw = t // pt, h // ph, w // pw

        tokens = self.patch_embedding(x)  # (B, gt, gh, gw, d)
        tokens = tokens.reshape(b, gt * gh * gw, -1)
        rope = self.rope(gt, gh, gw)

        temb, timestep_proj, context = self.condition_embedder(timestep, context)
        timestep_proj = timestep_proj.reshape(b, 6, -1)

        for block in self.blocks:
            tokens = block(tokens, context, timestep_proj, rope)

        mod = self.scale_shift_table + temb.astype(mx.float32)[:, None]  # (B, 2, d)
        shift, scale = mod[:, 0:1], mod[:, 1:2]
        tokens = _layernorm_f32(tokens, self.config.eps) * (1 + scale) + shift
        tokens = self.proj_out(tokens.astype(x.dtype))

        # unpatchify -> (B, T, H, W, C_out)
        c = self.config.out_channels
        tokens = tokens.reshape(b, gt, gh, gw, pt, ph, pw, c)
        tokens = tokens.transpose(0, 1, 4, 2, 5, 3, 6, 7)
        return tokens.reshape(b, gt * pt, gh * ph, gw * pw, c)
