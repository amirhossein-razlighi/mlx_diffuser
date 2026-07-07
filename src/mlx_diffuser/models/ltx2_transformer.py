"""LTX2Transformer3DModel: a faithful MLX port of LTX-2's audio-video DiT.

Mirrors diffusers' ``LTX2VideoTransformer3DModel`` module-for-module so the
official weights load via the converter. LTX-2 denoises a *joint* state — a
video token stream (dim 4096) and an audio token stream (dim 2048) — in 48
blocks that each run per-modality self-attention, text cross-attention, and
bidirectional audio<->video cross-attention, all modulated by per-block
``scale_shift_table`` parameters added to shared timestep projections.

LTX-2.3 specifics reproduced here: "split" RoPE (rotate-half with per-head
frequency bands), gated attention (a per-head sigmoid gate computed from the
pre-attention hidden states), adaLN modulation of the text cross-attention
(9 modulation params per block plus global prompt scale/shift tables), and
patch-boundary-midpoint positional coordinates scaled to seconds/pixels.

RoPE tables are computed in float64 via numpy (MLX's GPU arrays are float32,
but the reference uses double precision for the frequencies) — they are cached
per grid, so this costs nothing per step.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..caching import FirstBlockCache
from ..configuration import Config
from ..modeling import ModelMixin
from .wan_transformer import _sinusoidal_timestep


@dataclasses.dataclass
class LTX2TransformerConfig(Config):
    # video stream
    in_channels: int = 128
    out_channels: int = 128
    patch_size: int = 1
    patch_size_t: int = 1
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    cross_attention_dim: int = 4096
    vae_scale_factors: tuple[int, int, int] = (8, 32, 32)
    pos_embed_max_pos: int = 20
    base_height: int = 2048
    base_width: int = 2048
    # audio stream
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    audio_cross_attention_dim: int = 2048
    audio_scale_factor: int = 4
    audio_pos_embed_max_pos: int = 20
    audio_sampling_rate: int = 16000
    audio_hop_length: int = 160
    # shared
    num_layers: int = 48
    caption_channels: int = 3840
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    causal_offset: int = 1
    timestep_scale_multiplier: float = 1000.0
    cross_attn_timestep_scale_multiplier: float = 1000.0
    rope_type: str = "split"  # "split" (LTX-2.3) or "interleaved" (LTX-2.0)
    gated_attn: bool = True
    cross_attn_mod: bool = True  # 9 modulation params + prompt adaLN (LTX-2.3)
    use_prompt_embeddings: bool = False  # 2.0 projects text in the transformer

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def audio_inner_dim(self) -> int:
        return self.audio_num_attention_heads * self.audio_attention_head_dim

    @classmethod
    def ltx_2_3_22b(cls) -> LTX2TransformerConfig:
        return cls()  # defaults match the LTX-2.3 22B checkpoint


# --- rotary embeddings --------------------------------------------------------


def _rope_freqs_1d(grid: np.ndarray, dim: int, num_pos_dims: int, theta: float) -> np.ndarray:
    """The LTX-2 frequency ramp in float64: ``(grid*2-1) * theta**linspace * pi/2``.

    ``grid`` is ``(B, T, num_pos_dims)`` of positions already normalized to the
    base range; returns raw angles ``(B, T, num_pos_dims * (dim // (2*num_pos_dims)))``.
    """
    steps = dim // (2 * num_pos_dims)
    pow_indices = theta ** np.linspace(0.0, 1.0, steps, dtype=np.float64)
    freqs = pow_indices * np.pi / 2.0
    ang = (grid[..., None] * 2 - 1) * freqs  # (B, T, dims, steps)
    return ang.transpose(0, 1, 3, 2).reshape(grid.shape[0], grid.shape[1], -1)


def _rope_tables(
    ang: np.ndarray, dim: int, heads: int, rope_type: str
) -> tuple[mx.array, mx.array]:
    """Turn raw angles into (cos, sin) tables, padded to ``dim`` and head-split.

    "interleaved": tables ``(B, T, dim)`` with each angle repeated twice, padding
    (cos 1 / sin 0) *prepended*. "split": tables ``(B, heads, T, dim/2/heads)``
    where the ``dim/2`` frequencies are dealt across heads in order.
    """
    cos, sin = np.cos(ang), np.sin(ang)
    if rope_type == "interleaved":
        cos = np.repeat(cos, 2, axis=-1)
        sin = np.repeat(sin, 2, axis=-1)
        pad = dim - cos.shape[-1]
        if pad:
            cos = np.concatenate([np.ones_like(cos[..., :pad]), cos], axis=-1)
            sin = np.concatenate([np.zeros_like(sin[..., :pad]), sin], axis=-1)
    else:  # split
        pad = dim // 2 - cos.shape[-1]
        if pad:
            cos = np.concatenate([np.ones_like(cos[..., :pad]), cos], axis=-1)
            sin = np.concatenate([np.zeros_like(sin[..., :pad]), sin], axis=-1)
        b, t = cos.shape[:2]
        cos = cos.reshape(b, t, heads, -1).swapaxes(1, 2)
        sin = sin.reshape(b, t, heads, -1).swapaxes(1, 2)
    return mx.array(cos.astype(np.float32)), mx.array(sin.astype(np.float32))


def _apply_rope_interleaved(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Interleaved RoPE on ``(B, T, dim)`` with ``(B, T, dim)`` tables."""
    xr = x.reshape(*x.shape[:-1], -1, 2)
    x_real, x_imag = xr[..., 0], xr[..., 1]
    rot = mx.stack([-x_imag, x_real], axis=-1).reshape(x.shape)
    return (x.astype(mx.float32) * cos + rot.astype(mx.float32) * sin).astype(x.dtype)


def _apply_rope_split(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Split RoPE: rotate-half per head with per-head tables ``(B, H, T, r)``.

    ``x`` is ``(B, T, dim)``; each head's channels are split into two halves that
    form the rotation pairs.
    """
    _, heads, t, r = cos.shape
    b = x.shape[0]
    xh = x.reshape(b, t, heads, 2, r).astype(mx.float32)
    x1, x2 = xh[..., 0, :], xh[..., 1, :]  # (B, T, H, r)
    cos_t = cos.transpose(0, 2, 1, 3)  # (B, T, H, r)
    sin_t = sin.transpose(0, 2, 1, 3)
    out1 = x1 * cos_t - x2 * sin_t
    out2 = x2 * cos_t + x1 * sin_t
    return mx.stack([out1, out2], axis=-2).reshape(x.shape).astype(x.dtype)


def _apply_rope(x: mx.array, rope, rope_type: str) -> mx.array:
    if rope is None:
        return x
    if rope_type == "split":
        return _apply_rope_split(x, *rope)
    return _apply_rope_interleaved(x, *rope)


class LTX2RotaryPosEmbed:
    """Video/audio RoPE for LTX-2: patch-midpoint coords in seconds/pixels.

    ``prepare_video_coords`` returns ``(B, 3, T)`` midpoints (time in seconds —
    frame index scaled by the VAE stride, causal-offset, divided by fps — and
    pixel-space row/col centers). ``prepare_audio_coords`` returns ``(B, 1, T)``
    midpoints in seconds of the mel-latent grid. ``__call__`` converts
    coordinates into (cos, sin) tables.
    """

    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        rope_type: str,
        theta: float,
        max_positions: tuple[float, ...],
        causal_offset: int = 1,
    ):
        self.dim = dim
        self.heads = heads
        self.rope_type = rope_type
        self.theta = theta
        self.max_positions = max_positions
        self.causal_offset = causal_offset

    def prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        *,
        scale_factors: tuple[int, int, int],
        fps: float,
    ) -> np.ndarray:
        st, sh, sw = scale_factors
        grid = np.stack(
            np.meshgrid(
                np.arange(num_frames, dtype=np.float64),
                np.arange(height, dtype=np.float64),
                np.arange(width, dtype=np.float64),
                indexing="ij",
            ),
            axis=0,
        ).reshape(3, -1)  # (3, T) patch starts (patch sizes are 1)
        starts = grid * np.array([st, sh, sw], dtype=np.float64)[:, None]
        ends = (grid + 1) * np.array([st, sh, sw], dtype=np.float64)[:, None]
        for c in (starts, ends):
            c[0] = np.clip(c[0] + self.causal_offset - st, 0, None) / fps
        mid = (starts + ends) / 2.0
        return np.broadcast_to(mid[None], (batch_size, 3, mid.shape[-1]))

    def prepare_audio_coords(
        self,
        batch_size: int,
        num_frames: int,
        *,
        scale_factor: int,
        hop_length: int,
        sampling_rate: int,
    ) -> np.ndarray:
        grid = np.arange(num_frames, dtype=np.float64)
        to_secs = lambda g: (  # noqa: E731
            np.clip(g * scale_factor + self.causal_offset - scale_factor, 0, None)
            * hop_length
            / sampling_rate
        )
        mid = (to_secs(grid) + to_secs(grid + 1)) / 2.0
        return np.broadcast_to(mid[None, None], (batch_size, 1, num_frames))

    def __call__(self, coords: np.ndarray) -> tuple[mx.array, mx.array]:
        num_pos_dims = coords.shape[1]
        grid = np.stack(
            [coords[:, i] / self.max_positions[i] for i in range(num_pos_dims)], axis=-1
        )
        ang = _rope_freqs_1d(grid, self.dim, num_pos_dims, self.theta)
        return _rope_tables(ang, self.dim, self.heads, self.rope_type)


# --- shared modules -----------------------------------------------------------


class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class _PixArtTimeEmbed(nn.Module):
    """PixArtAlphaCombinedTimestepSizeEmbeddings without additional conditions."""

    def __init__(self, dim: int):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, dim)

    def __call__(self, t: mx.array) -> mx.array:
        return self.timestep_embedder(_sinusoidal_timestep(t, 256))


class LTX2AdaLNSingle(nn.Module):
    """Timestep embed + one linear producing ``num_mod_params`` modulation vectors."""

    def __init__(self, dim: int, num_mod_params: int):
        super().__init__()
        self.emb = _PixArtTimeEmbed(dim)
        self.linear = nn.Linear(dim, num_mod_params * dim)

    def __call__(self, t: mx.array) -> tuple[mx.array, mx.array]:
        embedded = self.emb(t)
        return self.linear(nn.silu(embedded)), embedded


def _rmsnorm_f32(x: mx.array, eps: float) -> mx.array:
    xf = x.astype(mx.float32)
    return xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)


class LTX2Attention(nn.Module):
    """LTX-2 attention: qk-RMSNorm across the full inner dim, RoPE applied
    pre-head-split, optional per-head sigmoid gating, separate q/k RoPE for the
    audio<->video cross-attention."""

    def __init__(
        self,
        query_dim: int,
        *,
        heads: int,
        dim_head: int,
        cross_attention_dim: int | None = None,
        gated: bool = False,
        rope_type: str = "split",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        inner = heads * dim_head
        kv_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.rope_type = rope_type
        self.norm_q = nn.RMSNorm(inner, eps=eps)
        self.norm_k = nn.RMSNorm(inner, eps=eps)
        self.to_q = nn.Linear(query_dim, inner)
        self.to_k = nn.Linear(kv_dim, inner)
        self.to_v = nn.Linear(kv_dim, inner)
        self.to_out = [nn.Linear(inner, query_dim)]
        self.to_gate_logits = nn.Linear(query_dim, heads) if gated else None

    def __call__(
        self,
        x: mx.array,
        context: mx.array | None = None,
        *,
        q_rope=None,
        k_rope=None,
        mask: mx.array | None = None,
    ) -> mx.array:
        kv = x if context is None else context
        b, lq = x.shape[0], x.shape[1]
        gate_logits = self.to_gate_logits(x) if self.to_gate_logits is not None else None

        q = self.norm_q(self.to_q(x))
        k = self.norm_k(self.to_k(kv))
        v = self.to_v(kv)
        if q_rope is not None:
            q = _apply_rope(q, q_rope, self.rope_type)
            k = _apply_rope(k, k_rope if k_rope is not None else q_rope, self.rope_type)

        q = q.reshape(b, lq, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, kv.shape[1], self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, kv.shape[1], self.heads, self.head_dim).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=None if mask is None else mask.astype(v.dtype)
        )
        out = out.transpose(0, 2, 1, 3)  # (B, L, H, hd)
        if gate_logits is not None:
            out = out * (2.0 * mx.sigmoid(gate_logits))[..., None]
        return self.to_out[0](out.reshape(b, lq, -1))


class LTX2FeedForward(nn.Module):
    """diffusers FeedForward(gelu-approximate): net.0.proj -> gelu_tanh -> net.2."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = [_GELUProj(dim, dim * 4), _Identity(), nn.Linear(dim * 4, dim)]

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


def _mod_params(table: mx.array, temb: mx.array, n: int) -> list[mx.array]:
    """``table (n, d)`` + ``temb (B, T, n*d)`` -> ``n`` tensors of ``(B, T, d)``."""
    b, t = temb.shape[0], temb.shape[1]
    vals = table[None, None].astype(mx.float32) + temb.astype(mx.float32).reshape(b, t, n, -1)
    return [vals[:, :, i] for i in range(n)]


# --- transformer block --------------------------------------------------------


class LTX2TransformerBlock(nn.Module):
    def __init__(self, cfg: LTX2TransformerConfig):
        super().__init__()
        d, ad = cfg.inner_dim, cfg.audio_inner_dim
        vh, vhd = cfg.num_attention_heads, cfg.attention_head_dim
        ah, ahd = cfg.audio_num_attention_heads, cfg.audio_attention_head_dim
        eps, rt, gated = cfg.norm_eps, cfg.rope_type, cfg.gated_attn
        self.eps = eps
        self.cross_attn_mod = cfg.cross_attn_mod

        # 1. self-attention (video, audio)
        self.attn1 = LTX2Attention(d, heads=vh, dim_head=vhd, gated=gated, rope_type=rt, eps=eps)
        self.audio_attn1 = LTX2Attention(
            ad, heads=ah, dim_head=ahd, gated=gated, rope_type=rt, eps=eps
        )
        # 2. text cross-attention (video, audio)
        self.attn2 = LTX2Attention(
            d,
            heads=vh,
            dim_head=vhd,
            cross_attention_dim=cfg.cross_attention_dim,
            gated=gated,
            rope_type=rt,
            eps=eps,
        )
        self.audio_attn2 = LTX2Attention(
            ad,
            heads=ah,
            dim_head=ahd,
            cross_attention_dim=cfg.audio_cross_attention_dim,
            gated=gated,
            rope_type=rt,
            eps=eps,
        )
        # 3. audio->video and video->audio cross-attention (audio-sized heads)
        self.audio_to_video_attn = LTX2Attention(
            d, heads=ah, dim_head=ahd, cross_attention_dim=ad, gated=gated, rope_type=rt, eps=eps
        )
        self.video_to_audio_attn = LTX2Attention(
            ad, heads=ah, dim_head=ahd, cross_attention_dim=d, gated=gated, rope_type=rt, eps=eps
        )
        # 4. feed-forward
        self.ff = LTX2FeedForward(d)
        self.audio_ff = LTX2FeedForward(ad)
        # 5. per-block modulation tables
        n_mod = 9 if cfg.cross_attn_mod else 6
        self.scale_shift_table = mx.zeros((n_mod, d))
        self.audio_scale_shift_table = mx.zeros((n_mod, ad))
        if cfg.cross_attn_mod:
            self.prompt_scale_shift_table = mx.zeros((2, d))
            self.audio_prompt_scale_shift_table = mx.zeros((2, ad))
        self.video_a2v_cross_attn_scale_shift_table = mx.zeros((5, d))
        self.audio_a2v_cross_attn_scale_shift_table = mx.zeros((5, ad))

    def __call__(
        self,
        x: mx.array,
        audio: mx.array,
        text: mx.array,
        audio_text: mx.array,
        *,
        temb: mx.array,
        temb_audio: mx.array,
        temb_ca: mx.array,
        temb_ca_audio: mx.array,
        temb_ca_gate: mx.array,
        temb_ca_audio_gate: mx.array,
        temb_prompt: mx.array | None,
        temb_prompt_audio: mx.array | None,
        video_rope,
        audio_rope,
        ca_video_rope,
        ca_audio_rope,
        text_mask: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        n_mod = self.scale_shift_table.shape[0]
        vmod = _mod_params(self.scale_shift_table, temb, n_mod)
        amod = _mod_params(self.audio_scale_shift_table, temb_audio, n_mod)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = vmod[:6]
        a_shift_msa, a_scale_msa, a_gate_msa, a_shift_mlp, a_scale_mlp, a_gate_mlp = amod[:6]

        # 1. self-attention
        norm_x = _rmsnorm_f32(x, self.eps) * (1 + scale_msa) + shift_msa
        x = (
            x.astype(mx.float32)
            + self.attn1(norm_x.astype(text.dtype), q_rope=video_rope) * gate_msa
        ).astype(text.dtype)
        norm_a = _rmsnorm_f32(audio, self.eps) * (1 + a_scale_msa) + a_shift_msa
        audio = (
            audio.astype(mx.float32)
            + self.audio_attn1(norm_a.astype(text.dtype), q_rope=audio_rope) * a_gate_msa
        ).astype(text.dtype)

        # 2. text cross-attention
        norm_x = _rmsnorm_f32(x, self.eps)
        norm_a = _rmsnorm_f32(audio, self.eps)
        if self.cross_attn_mod:
            assert temb_prompt is not None and temb_prompt_audio is not None
            shift_q, scale_q, gate_q = vmod[6:9]
            a_shift_q, a_scale_q, a_gate_q = amod[6:9]
            shift_kv, scale_kv = _mod_params(self.prompt_scale_shift_table, temb_prompt, 2)
            a_shift_kv, a_scale_kv = _mod_params(
                self.audio_prompt_scale_shift_table, temb_prompt_audio, 2
            )
            norm_x = norm_x * (1 + scale_q) + shift_q
            norm_a = norm_a * (1 + a_scale_q) + a_shift_q
            text = (text.astype(mx.float32) * (1 + scale_kv) + shift_kv).astype(text.dtype)
            audio_text = (audio_text.astype(mx.float32) * (1 + a_scale_kv) + a_shift_kv).astype(
                audio_text.dtype
            )
        attn = self.attn2(norm_x.astype(text.dtype), context=text, mask=text_mask)
        if self.cross_attn_mod:
            attn = attn * gate_q
        x = (x.astype(mx.float32) + attn).astype(text.dtype)
        attn = self.audio_attn2(norm_a.astype(text.dtype), context=audio_text, mask=text_mask)
        if self.cross_attn_mod:
            attn = attn * a_gate_q
        audio = (audio.astype(mx.float32) + attn).astype(text.dtype)

        # 3. audio<->video cross-attention
        norm_x = _rmsnorm_f32(x, self.eps)
        norm_a = _rmsnorm_f32(audio, self.eps)
        v_ca = _mod_params(self.video_a2v_cross_attn_scale_shift_table[:4], temb_ca, 4)
        v_gate = _mod_params(self.video_a2v_cross_attn_scale_shift_table[4:], temb_ca_gate, 1)[0]
        a_ca = _mod_params(self.audio_a2v_cross_attn_scale_shift_table[:4], temb_ca_audio, 4)
        a_gate = _mod_params(
            self.audio_a2v_cross_attn_scale_shift_table[4:], temb_ca_audio_gate, 1
        )[0]
        v_a2v_scale, v_a2v_shift, v_v2a_scale, v_v2a_shift = v_ca
        a_a2v_scale, a_a2v_shift, a_v2a_scale, a_v2a_shift = a_ca

        mod_x = (norm_x * (1 + v_a2v_scale) + v_a2v_shift).astype(text.dtype)
        mod_a = (norm_a * (1 + a_a2v_scale) + a_a2v_shift).astype(text.dtype)
        a2v = self.audio_to_video_attn(
            mod_x, context=mod_a, q_rope=ca_video_rope, k_rope=ca_audio_rope
        )
        x = (x.astype(mx.float32) + v_gate * a2v).astype(text.dtype)

        mod_x = (norm_x * (1 + v_v2a_scale) + v_v2a_shift).astype(text.dtype)
        mod_a = (norm_a * (1 + a_v2a_scale) + a_v2a_shift).astype(text.dtype)
        v2a = self.video_to_audio_attn(
            mod_a, context=mod_x, q_rope=ca_audio_rope, k_rope=ca_video_rope
        )
        audio = (audio.astype(mx.float32) + a_gate * v2a).astype(text.dtype)

        # 4. feed-forward
        norm_x = _rmsnorm_f32(x, self.eps) * (1 + scale_mlp) + shift_mlp
        x = (
            x.astype(mx.float32) + self.ff(norm_x.astype(text.dtype)).astype(mx.float32) * gate_mlp
        ).astype(text.dtype)
        norm_a = _rmsnorm_f32(audio, self.eps) * (1 + a_scale_mlp) + a_shift_mlp
        audio = (
            audio.astype(mx.float32)
            + self.audio_ff(norm_a.astype(text.dtype)).astype(mx.float32) * a_gate_mlp
        ).astype(text.dtype)
        return x, audio


# --- model --------------------------------------------------------------------


class LTX2Transformer3DModel(ModelMixin[LTX2TransformerConfig]):
    """LTX-2 audio-video diffusion transformer over packed token sequences.

    Inputs are packed latents: video ``(B, F*H*W, in_channels)`` (patch sizes are
    1 for LTX-2) and audio ``(B, L_a, audio_in_channels)``; text streams come
    from :class:`~mlx_diffuser.models.ltx2_connectors.LTX2TextConnectors`.
    """

    config_class = LTX2TransformerConfig

    def __init__(self, config: LTX2TransformerConfig):
        super().__init__()
        self.config = config
        cfg = config
        d, ad = cfg.inner_dim, cfg.audio_inner_dim

        self.proj_in = nn.Linear(cfg.in_channels, d)
        self.audio_proj_in = nn.Linear(cfg.audio_in_channels, ad)

        n_mod = 9 if cfg.cross_attn_mod else 6
        self.time_embed = LTX2AdaLNSingle(d, n_mod)
        self.audio_time_embed = LTX2AdaLNSingle(ad, n_mod)
        self.av_cross_attn_video_scale_shift = LTX2AdaLNSingle(d, 4)
        self.av_cross_attn_audio_scale_shift = LTX2AdaLNSingle(ad, 4)
        self.av_cross_attn_video_a2v_gate = LTX2AdaLNSingle(d, 1)
        self.av_cross_attn_audio_v2a_gate = LTX2AdaLNSingle(ad, 1)
        if cfg.cross_attn_mod:
            self.prompt_adaln = LTX2AdaLNSingle(d, 2)
            self.audio_prompt_adaln = LTX2AdaLNSingle(ad, 2)
        self.scale_shift_table = mx.zeros((2, d))
        self.audio_scale_shift_table = mx.zeros((2, ad))

        self.rope = LTX2RotaryPosEmbed(
            d,
            heads=cfg.num_attention_heads,
            rope_type=cfg.rope_type,
            theta=cfg.rope_theta,
            max_positions=(cfg.pos_embed_max_pos, cfg.base_height, cfg.base_width),
            causal_offset=cfg.causal_offset,
        )
        self.audio_rope = LTX2RotaryPosEmbed(
            ad,
            heads=cfg.audio_num_attention_heads,
            rope_type=cfg.rope_type,
            theta=cfg.rope_theta,
            max_positions=(cfg.audio_pos_embed_max_pos,),
            causal_offset=cfg.causal_offset,
        )
        ca_max = max(cfg.pos_embed_max_pos, cfg.audio_pos_embed_max_pos)
        self.cross_attn_rope = LTX2RotaryPosEmbed(
            cfg.audio_cross_attention_dim,
            heads=cfg.num_attention_heads,
            rope_type=cfg.rope_type,
            theta=cfg.rope_theta,
            max_positions=(ca_max,),
            causal_offset=cfg.causal_offset,
        )
        self.cross_attn_audio_rope = LTX2RotaryPosEmbed(
            cfg.audio_cross_attention_dim,
            heads=cfg.audio_num_attention_heads,
            rope_type=cfg.rope_type,
            theta=cfg.rope_theta,
            max_positions=(ca_max,),
            causal_offset=cfg.causal_offset,
        )

        self.transformer_blocks = [LTX2TransformerBlock(cfg) for _ in range(cfg.num_layers)]

        # norm_out / audio_norm_out are affine-free LayerNorms (see _layernorm_f32).
        self.proj_out = nn.Linear(d, cfg.out_channels)
        self.audio_proj_out = nn.Linear(ad, cfg.audio_out_channels)

    def prepare_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        audio_num_frames: int,
        fps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Precompute the (constant per run) video/audio positional coordinates."""
        cfg = self.config
        video = self.rope.prepare_video_coords(
            batch_size, num_frames, height, width, scale_factors=cfg.vae_scale_factors, fps=fps
        )
        audio = self.audio_rope.prepare_audio_coords(
            batch_size,
            audio_num_frames,
            scale_factor=cfg.audio_scale_factor,
            hop_length=cfg.audio_hop_length,
            sampling_rate=cfg.audio_sampling_rate,
        )
        return video, audio

    def __call__(
        self,
        hidden_states: mx.array,
        audio_hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        audio_encoder_hidden_states: mx.array,
        timestep: mx.array,
        video_coords: np.ndarray,
        audio_coords: np.ndarray,
        encoder_attention_mask: mx.array | None = None,
        cache: FirstBlockCache | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Denoise one step. ``timestep`` is ``(B,)`` and already scaled by 1000.

        Returns the (video, audio) velocity predictions with the input shapes.
        """
        cfg = self.config

        text_mask = None
        if encoder_attention_mask is not None:
            m = encoder_attention_mask.astype(mx.float32)
            text_mask = ((1.0 - m) * -10000.0)[:, None, None, :]  # (B, 1, 1, L)

        video_rope = self.rope(video_coords)
        audio_rope = self.audio_rope(audio_coords)
        ca_video_rope = self.cross_attn_rope(video_coords[:, 0:1])
        ca_audio_rope = self.cross_attn_audio_rope(audio_coords[:, 0:1])

        x = self.proj_in(hidden_states)
        audio = self.audio_proj_in(audio_hidden_states)

        t = timestep.astype(mx.float32)
        gate_scale = cfg.cross_attn_timestep_scale_multiplier / cfg.timestep_scale_multiplier
        temb, embedded = self.time_embed(t)
        temb_audio, audio_embedded = self.audio_time_embed(t)
        temb, embedded = temb[:, None], embedded[:, None]  # (B, 1, n*d) / (B, 1, d)
        temb_audio, audio_embedded = temb_audio[:, None], audio_embedded[:, None]
        temb_ca = self.av_cross_attn_video_scale_shift(t)[0][:, None]
        temb_ca_gate = self.av_cross_attn_video_a2v_gate(t * gate_scale)[0][:, None]
        temb_ca_audio = self.av_cross_attn_audio_scale_shift(t)[0][:, None]
        temb_ca_audio_gate = self.av_cross_attn_audio_v2a_gate(t * gate_scale)[0][:, None]
        temb_prompt = temb_prompt_audio = None
        if cfg.cross_attn_mod:
            temb_prompt = self.prompt_adaln(t)[0][:, None]
            temb_prompt_audio = self.audio_prompt_adaln(t)[0][:, None]

        block_kwargs = dict(
            temb=temb,
            temb_audio=temb_audio,
            temb_ca=temb_ca,
            temb_ca_audio=temb_ca_audio,
            temb_ca_gate=temb_ca_gate,
            temb_ca_audio_gate=temb_ca_audio_gate,
            temb_prompt=temb_prompt,
            temb_prompt_audio=temb_prompt_audio,
            video_rope=video_rope,
            audio_rope=audio_rope,
            ca_video_rope=ca_video_rope,
            ca_audio_rope=ca_audio_rope,
            text_mask=text_mask,
        )
        x, audio = self._run_blocks(
            x, audio, encoder_hidden_states, audio_encoder_hidden_states, block_kwargs, cache
        )

        mod = (
            self.scale_shift_table[None, None].astype(mx.float32)
            + embedded.astype(mx.float32)[:, :, None]
        )
        shift, scale = mod[:, :, 0], mod[:, :, 1]
        x = _layernorm_f32(x, 1e-6) * (1 + scale) + shift
        out = self.proj_out(x.astype(hidden_states.dtype))

        amod = (
            self.audio_scale_shift_table[None, None].astype(mx.float32)
            + audio_embedded.astype(mx.float32)[:, :, None]
        )
        a_shift, a_scale = amod[:, :, 0], amod[:, :, 1]
        audio = _layernorm_f32(audio, 1e-6) * (1 + a_scale) + a_shift
        audio_out = self.audio_proj_out(audio.astype(audio_hidden_states.dtype))
        return out, audio_out

    def _run_blocks(self, x, audio, text, audio_text, block_kwargs, cache):
        """Run the blocks, optionally reusing the First-Block Cache (both streams)."""
        if cache is None:
            for block in self.transformer_blocks:
                x, audio = block(x, audio, text, audio_text, **block_kwargs)
            return x, audio

        first_x, first_a = self.transformer_blocks[0](x, audio, text, audio_text, **block_kwargs)
        if cache.should_reuse(first_x - x):
            res_x, res_a = cache.residual
            return first_x + res_x, first_a + res_a
        hx, ha = first_x, first_a
        for block in self.transformer_blocks[1:]:
            hx, ha = block(hx, ha, text, audio_text, **block_kwargs)
        cache.residual = (hx - first_x, ha - first_a)
        return hx, ha


def _layernorm_f32(x: mx.array, eps: float) -> mx.array:
    xf = x.astype(mx.float32)
    mean = mx.mean(xf, axis=-1, keepdims=True)
    var = mx.var(xf, axis=-1, keepdims=True)
    return (xf - mean) * mx.rsqrt(var + eps)
