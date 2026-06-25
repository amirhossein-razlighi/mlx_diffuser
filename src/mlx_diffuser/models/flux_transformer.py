"""FluxTransformer2DModel: a faithful MLX port of FLUX.1's text-to-image MMDiT.

Mirrors diffusers' ``FluxTransformer2DModel`` module-for-module so the official FLUX.1
(schnell / dev) weights load via the converter. Packed image latents ``(B, L_img, 64)``
and T5 text tokens ``(B, L_txt, 4096)`` are embedded to a shared ``inner_dim`` and
processed by two stacks:

* **double-stream blocks** — image and text keep separate residual streams, each with
  its own adaLN-Zero modulation and projections, but attend *jointly* (their q/k/v are
  concatenated for one attention) so information mixes both ways;
* **single-stream blocks** — image and text are concatenated into one sequence and run
  through a fused attention+MLP block (FLUX's parallel-transformer design).

Positions use a 3-axis rotary embedding (a constant time axis plus the latent row/col),
applied with the interleaved (GPT-J) convention shared with WAN. adaLN modulation is
gated (adaLN-Zero), and qk-RMSNorm stabilizes attention. Tensors are channels-last.
"""

from __future__ import annotations

import dataclasses
import math

import mlx.core as mx
import mlx.nn as nn

from ..caching import FirstBlockCache
from ..configuration import Config
from ..modeling import ModelMixin


@dataclasses.dataclass
class FluxConfig(Config):
    patch_size: int = 1
    in_channels: int = 64
    out_channels: int | None = None
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = False
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def out_ch(self) -> int:
        return self.out_channels if self.out_channels is not None else self.in_channels

    @classmethod
    def flux_schnell(cls) -> FluxConfig:
        return cls(guidance_embeds=False)

    @classmethod
    def flux_dev(cls) -> FluxConfig:
        return cls(guidance_embeds=True)


# --- embeddings & rope -------------------------------------------------------


def _sinusoidal_timestep(t: mx.array, dim: int = 256, max_period: float = 10000.0) -> mx.array:
    """diffusers ``Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)``."""
    half = dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
    emb = t.astype(mx.float32)[:, None] * mx.exp(exponent)[None]
    return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)


def flux_rope(
    ids: mx.array, axes_dim: tuple[int, ...], theta: float = 10000.0
) -> tuple[mx.array, mx.array]:
    """3-axis FLUX rotary embedding.

    ``ids`` is ``(L, n_axes)`` position indices; for each axis we build half-dim rotary
    frequencies and concatenate, giving ``(cos, sin)`` of shape ``(1, L, 1, head_dim/2)``
    ready for the interleaved :func:`apply_rope` (sequence at axis 1, heads at axis 2).
    """
    cos_parts, sin_parts = [], []
    for i, dim in enumerate(axes_dim):
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))  # (dim/2,)
        ang = ids[:, i : i + 1].astype(mx.float32) * inv_freq[None]  # (L, dim/2)
        cos_parts.append(mx.cos(ang))
        sin_parts.append(mx.sin(ang))
    cos = mx.concatenate(cos_parts, axis=-1)[None, :, None, :]
    sin = mx.concatenate(sin_parts, axis=-1)[None, :, None, :]
    return cos, sin


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Interleaved (GPT-J) RoPE on ``(B, L, heads, head_dim)`` with half-dim ``cos``/``sin``."""
    xr = x.reshape(*x.shape[:-1], -1, 2)
    x_even, x_odd = xr[..., 0], xr[..., 1]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return mx.stack([out_even, out_odd], axis=-1).reshape(x.shape)


class _MLPEmbed(nn.Module):
    """diffusers ``TimestepEmbedding`` / ``PixArtAlphaTextProjection``: lin -> act -> lin."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class CombinedTimestepTextEmbed(nn.Module):
    """temb = timestep_embedder(sinusoid(t)) [+ guidance_embedder(sinusoid(g))] + text_embedder(pooled)."""

    def __init__(self, cfg: FluxConfig):
        super().__init__()
        d = cfg.inner_dim
        self.timestep_embedder = _MLPEmbed(256, d)
        self.guidance = cfg.guidance_embeds
        if cfg.guidance_embeds:
            self.guidance_embedder = _MLPEmbed(256, d)
        self.text_embedder = _MLPEmbed(cfg.pooled_projection_dim, d)

    def __call__(self, timestep: mx.array, pooled: mx.array, guidance: mx.array | None) -> mx.array:
        temb = self.timestep_embedder(_sinusoidal_timestep(timestep).astype(pooled.dtype))
        if self.guidance:
            assert guidance is not None, "guidance-distilled FLUX (dev) requires a guidance scale"
            temb = temb + self.guidance_embedder(
                _sinusoidal_timestep(guidance).astype(pooled.dtype)
            )
        return temb + self.text_embedder(pooled)


# --- attention ---------------------------------------------------------------


class FluxAttention(nn.Module):
    """Joint attention with qk-RMSNorm and RoPE.

    In double-stream blocks the text (``encoder``) and image streams have separate
    projections (``add_*`` for text, ``to_*`` for image); their q/k/v are concatenated
    (text first) for a single attention and split back afterwards. In single-stream
    blocks (``pre_only``) there is just one stream and the block does the output proj.
    """

    def __init__(self, dim: int, heads: int, head_dim: int, eps: float, pre_only: bool):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.pre_only = pre_only
        inner = heads * head_dim
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_q = nn.Linear(dim, inner)
        self.to_k = nn.Linear(dim, inner)
        self.to_v = nn.Linear(dim, inner)
        if not pre_only:
            self.to_out = [nn.Linear(inner, dim)]  # diffusers ModuleList [Linear, Dropout]
            self.norm_added_q = nn.RMSNorm(head_dim, eps=eps)
            self.norm_added_k = nn.RMSNorm(head_dim, eps=eps)
            self.add_q_proj = nn.Linear(dim, inner)
            self.add_k_proj = nn.Linear(dim, inner)
            self.add_v_proj = nn.Linear(dim, inner)
            self.to_add_out = nn.Linear(inner, dim)

    def _split_heads(self, x: mx.array) -> mx.array:
        b, length, _ = x.shape
        return x.reshape(b, length, self.heads, self.head_dim)

    def __call__(self, x, rope, encoder=None):
        b = x.shape[0]
        q = self.norm_q(self._split_heads(self.to_q(x)))
        k = self.norm_k(self._split_heads(self.to_k(x)))
        v = self._split_heads(self.to_v(x))
        if encoder is not None:  # double stream: text q/k/v prepended
            eq = self.norm_added_q(self._split_heads(self.add_q_proj(encoder)))
            ek = self.norm_added_k(self._split_heads(self.add_k_proj(encoder)))
            ev = self._split_heads(self.add_v_proj(encoder))
            q = mx.concatenate([eq, q], axis=1)
            k = mx.concatenate([ek, k], axis=1)
            v = mx.concatenate([ev, v], axis=1)
        q = apply_rope(q, *rope)
        k = apply_rope(k, *rope)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
        out = out.transpose(0, 2, 1, 3).reshape(b, -1, self.heads * self.head_dim)
        if encoder is None:
            return out  # single stream: block handles the output projection
        txt_len = encoder.shape[1]
        enc_out, img_out = out[:, :txt_len], out[:, txt_len:]
        return self.to_out[0](img_out), self.to_add_out(enc_out)


# --- feed-forward & norms ----------------------------------------------------


class _GELU(nn.Module):
    """diffusers ``GELU(approximate='tanh')``: a Linear then tanh-approx GELU."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu_approx(self.proj(x))


class _Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class FeedForward(nn.Module):
    """diffusers ``FeedForward(activation_fn='gelu-approximate')``: net.0 -> net.2."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = dim * mult
        self.net = [_GELU(dim, inner), _Identity(), nn.Linear(inner, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.net[2](self.net[0](x))


class _AdaLNZero(nn.Module):
    """adaLN-Zero (6 params): returns the modulated input plus the four mlp/gate signals."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 6 * dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6, affine=False)

    def __call__(self, x, emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            self.linear(nn.silu(emb)), 6, axis=-1
        )
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class _AdaLNZeroSingle(nn.Module):
    """adaLN-Zero for single-stream blocks (3 params): modulated input plus the gate."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 3 * dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6, affine=False)

    def __call__(self, x, emb):
        shift, scale, gate = mx.split(self.linear(nn.silu(emb)), 3, axis=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None], gate


class _AdaLNContinuous(nn.Module):
    """diffusers ``AdaLayerNormContinuous`` (chunk order scale, shift)."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 2 * dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6, affine=False)

    def __call__(self, x, emb):
        scale, shift = mx.split(self.linear(nn.silu(emb)), 2, axis=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


# --- blocks ------------------------------------------------------------------


class FluxTransformerBlock(nn.Module):
    """Double-stream block: image + text keep separate streams but attend jointly."""

    def __init__(self, cfg: FluxConfig):
        super().__init__()
        d = cfg.inner_dim
        self.norm1 = _AdaLNZero(d)
        self.norm1_context = _AdaLNZero(d)
        self.attn = FluxAttention(
            d, cfg.num_attention_heads, cfg.attention_head_dim, 1e-6, pre_only=False
        )
        self.norm2 = nn.LayerNorm(d, eps=1e-6, affine=False)
        self.ff = FeedForward(d)
        self.norm2_context = nn.LayerNorm(d, eps=1e-6, affine=False)
        self.ff_context = FeedForward(d)

    def __call__(self, hidden, encoder, temb, rope):
        norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden, temb)
        norm_e, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder, temb)

        attn_out, ctx_attn_out = self.attn(norm_h, rope, encoder=norm_e)

        hidden = hidden + gate_msa[:, None] * attn_out
        norm_h = self.norm2(hidden) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden = hidden + gate_mlp[:, None] * self.ff(norm_h)

        encoder = encoder + c_gate_msa[:, None] * ctx_attn_out
        norm_e = self.norm2_context(encoder) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        encoder = encoder + c_gate_mlp[:, None] * self.ff_context(norm_e)
        return encoder, hidden


class FluxSingleTransformerBlock(nn.Module):
    """Single-stream block: one fused attention + MLP over the concatenated sequence."""

    def __init__(self, cfg: FluxConfig):
        super().__init__()
        d = cfg.inner_dim
        self.mlp_hidden_dim = d * 4
        self.norm = _AdaLNZeroSingle(d)
        self.proj_mlp = nn.Linear(d, self.mlp_hidden_dim)
        self.proj_out = nn.Linear(d + self.mlp_hidden_dim, d)
        self.attn = FluxAttention(
            d, cfg.num_attention_heads, cfg.attention_head_dim, 1e-6, pre_only=True
        )

    def __call__(self, hidden, encoder, temb, rope):
        txt_len = encoder.shape[1]
        x = mx.concatenate([encoder, hidden], axis=1)
        residual = x
        norm_x, gate = self.norm(x, temb)
        mlp = nn.gelu_approx(self.proj_mlp(norm_x))
        attn_out = self.attn(norm_x, rope)
        x = mx.concatenate([attn_out, mlp], axis=-1)
        x = residual + gate[:, None] * self.proj_out(x)
        return x[:, :txt_len], x[:, txt_len:]


class FluxTransformer2DModel(ModelMixin[FluxConfig]):
    """FLUX.1 text-to-image diffusion transformer (MMDiT). Channels-last latents."""

    config_class = FluxConfig

    def __init__(self, config: FluxConfig):
        super().__init__()
        self.config = config
        d = config.inner_dim
        self.time_text_embed = CombinedTimestepTextEmbed(config)
        self.context_embedder = nn.Linear(config.joint_attention_dim, d)
        self.x_embedder = nn.Linear(config.in_channels, d)
        self.transformer_blocks = [FluxTransformerBlock(config) for _ in range(config.num_layers)]
        self.single_transformer_blocks = [
            FluxSingleTransformerBlock(config) for _ in range(config.num_single_layers)
        ]
        self.norm_out = _AdaLNContinuous(d)
        self.proj_out = nn.Linear(d, config.patch_size**2 * config.out_ch)

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        pooled_projections: mx.array,
        img_ids: mx.array,
        txt_ids: mx.array,
        guidance: mx.array | None = None,
        cache: FirstBlockCache | None = None,
    ) -> mx.array:
        """Predict the flow velocity for packed latents ``hidden_states`` ``(B, L_img, 64)``.

        ``timestep`` is the flow time in ``[0, 1]`` (scaled by 1000 internally, matching
        diffusers). ``encoder_hidden_states`` are T5 tokens ``(B, L_txt, 4096)``;
        ``pooled_projections`` is the CLIP pooled embed ``(B, 768)``. ``img_ids`` /
        ``txt_ids`` are the ``(L, 3)`` rotary position ids. ``guidance`` (FLUX-dev only)
        is the distilled guidance scale.
        """
        hidden = self.x_embedder(hidden_states)
        timestep = timestep * 1000
        guidance = guidance * 1000 if guidance is not None else None
        temb = self.time_text_embed(timestep, pooled_projections, guidance)
        encoder = self.context_embedder(encoder_hidden_states)

        ids = mx.concatenate([txt_ids, img_ids], axis=0)
        rope = flux_rope(ids, self.config.axes_dims_rope)

        encoder, hidden = self._run_blocks(hidden, encoder, temb, rope, cache)

        hidden = self.norm_out(hidden, temb)
        return self.proj_out(hidden)

    def _run_blocks(self, hidden, encoder, temb, rope, cache):
        """Run double then single blocks, optionally reusing the First-Block Cache.

        The cache signal is the first double block's contribution to the image stream;
        on a hit we add the cached residual of every remaining block instead of running
        them. Text-stream updates from the skipped blocks do not affect the output (only
        the image stream is read out), so caching the image residual alone is exact.
        """
        if cache is None or not cache.enabled:
            for block in self.transformer_blocks:
                encoder, hidden = block(hidden, encoder, temb, rope)
            for block in self.single_transformer_blocks:
                encoder, hidden = block(hidden, encoder, temb, rope)
            return encoder, hidden

        first_in = hidden
        encoder, hidden = self.transformer_blocks[0](hidden, encoder, temb, rope)
        if cache.should_reuse(hidden - first_in):
            return encoder, first_in + cache.residual
        after_first = hidden
        for block in self.transformer_blocks[1:]:
            encoder, hidden = block(hidden, encoder, temb, rope)
        for block in self.single_transformer_blocks:
            encoder, hidden = block(hidden, encoder, temb, rope)
        cache.residual = hidden - after_first
        return encoder, hidden
