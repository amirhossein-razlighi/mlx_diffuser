"""LTX2TextConnectors: adapts Gemma-3 hidden states for the LTX-2 transformer.

Mirrors diffusers' ``LTX2TextConnectors``. LTX-2 conditions on *all* Gemma-3
hidden states (49 per token for the 12B model): they are RMS-normalized per
token/layer, flattened, projected per modality (LTX-2.3's
``per_modality_projections``) and refined by a small 1D transformer per stream
whose padding positions are replaced with **learnable registers** — a bank of
128 learned "null text" tokens tiled across the padded tail. After the
connectors, every position is a valid key for the DiT's text cross-attention,
so no attention mask is needed downstream.
"""

from __future__ import annotations

import dataclasses
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..configuration import Config
from ..modeling import ModelMixin
from .ltx2_transformer import LTX2Attention, LTX2FeedForward, _rmsnorm_f32, _rope_tables


@dataclasses.dataclass
class LTX2ConnectorsConfig(Config):
    caption_channels: int = 3840
    text_proj_in_factor: int = 49  # Gemma-3-12B: embeddings + 48 layers
    video_hidden_dim: int = 4096
    audio_hidden_dim: int = 2048
    video_num_attention_heads: int = 32
    video_attention_head_dim: int = 128
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    num_layers: int = 8
    num_learnable_registers: int = 128
    rope_base_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_type: str = "split"
    gated_attn: bool = True
    proj_bias: bool = True
    eps: float = 1e-6

    @classmethod
    def ltx_2_3_22b(cls) -> LTX2ConnectorsConfig:
        return cls()  # defaults match the LTX-2.3 22B checkpoint


class LTX2ConnectorBlock(nn.Module):
    """Pre-RMSNorm attention + feed-forward block (no modulation)."""

    def __init__(self, dim: int, heads: int, head_dim: int, cfg: LTX2ConnectorsConfig):
        super().__init__()
        self.attn1 = LTX2Attention(
            dim,
            heads=heads,
            dim_head=head_dim,
            gated=cfg.gated_attn,
            rope_type=cfg.rope_type,
            eps=cfg.eps,
        )
        self.ff = LTX2FeedForward(dim)
        self.eps = cfg.eps

    def __call__(self, x: mx.array, rope) -> mx.array:
        h = _rmsnorm_f32(x, self.eps).astype(x.dtype)
        x = x + self.attn1(h, q_rope=rope)
        h = _rmsnorm_f32(x, self.eps).astype(x.dtype)
        return x + self.ff(h)


class LTX2ConnectorTransformer1d(nn.Module):
    """A small 1D transformer with learnable registers replacing padding."""

    def __init__(self, heads: int, head_dim: int, cfg: LTX2ConnectorsConfig):
        super().__init__()
        self.inner_dim = heads * head_dim
        self.heads = heads
        self.cfg = cfg
        self.learnable_registers = mx.zeros((cfg.num_learnable_registers, self.inner_dim))
        self.transformer_blocks = [
            LTX2ConnectorBlock(self.inner_dim, heads, head_dim, cfg) for _ in range(cfg.num_layers)
        ]

    def _rope(self, seq_len: int) -> tuple[mx.array, mx.array]:
        grid = (np.arange(seq_len, dtype=np.float64) / self.cfg.rope_base_seq_len)[None, :, None]
        steps = self.inner_dim // 2
        pow_indices = self.cfg.rope_theta ** np.linspace(0.0, 1.0, steps, dtype=np.float64)
        ang = (grid * 2 - 1) * (pow_indices * np.pi / 2.0)
        ang = ang.reshape(1, seq_len, -1)
        return _rope_tables(ang, self.inner_dim, self.heads, self.cfg.rope_type)

    def __call__(self, x: mx.array, attention_mask: mx.array) -> mx.array:
        """Run the connector; ``attention_mask`` is the binary (B, L) prompt mask.

        Padding positions (left padding) are replaced by the tiled register bank,
        after which every position is valid, so the blocks run unmasked.
        """
        b, seq_len, _ = x.shape
        n_reg = self.cfg.num_learnable_registers
        if seq_len % n_reg != 0:
            raise ValueError(f"sequence length {seq_len} must be a multiple of {n_reg} registers")
        registers = mx.tile(self.learnable_registers, (seq_len // n_reg, 1))

        # Compact each row's valid tokens to the front, then fill the tail with
        # registers. (Prompts are left-padded, so this is a left-shift per row.)
        rows = []
        n_valid = np.array(attention_mask.sum(axis=-1))
        for i in range(b):
            valid = int(n_valid[i])
            content = x[i, seq_len - valid :] if valid else x[i, :0]
            rows.append(mx.concatenate([content, registers[valid:]], axis=0))
        x = mx.stack(rows, axis=0)

        rope = self._rope(seq_len)
        for block in self.transformer_blocks:
            x = block(x, rope)
        return _rmsnorm_f32(x, self.cfg.eps).astype(x.dtype)


class LTX2TextConnectors(ModelMixin[LTX2ConnectorsConfig]):
    """Per-modality text feature extractors (LTX-2.3 layout)."""

    config_class = LTX2ConnectorsConfig

    def __init__(self, config: LTX2ConnectorsConfig):
        super().__init__()
        self.config = config
        in_dim = config.caption_channels * config.text_proj_in_factor
        self.video_text_proj_in = nn.Linear(in_dim, config.video_hidden_dim, bias=config.proj_bias)
        self.audio_text_proj_in = nn.Linear(in_dim, config.audio_hidden_dim, bias=config.proj_bias)
        self.video_connector = LTX2ConnectorTransformer1d(
            config.video_num_attention_heads, config.video_attention_head_dim, config
        )
        self.audio_connector = LTX2ConnectorTransformer1d(
            config.audio_num_attention_heads, config.audio_attention_head_dim, config
        )

    def __call__(
        self, text_hidden_states: mx.array, attention_mask: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Project stacked Gemma states ``(B, L, D, layers)`` for both streams.

        Returns ``(video_text (B, L, video_dim), audio_text (B, L, audio_dim))``.
        After the register replacement every position is valid, so no text
        attention mask is needed by the transformer.
        """
        cfg = self.config
        # Per-token, per-layer RMS norm over the caption channel dim (float32).
        xf = text_hidden_states.astype(mx.float32)
        normed = xf * mx.rsqrt(mx.mean(xf * xf, axis=2, keepdims=True) + 1e-6)
        normed = normed.reshape(*normed.shape[:2], -1)  # (B, L, D*layers), channel-major
        normed = normed * attention_mask.astype(mx.float32)[..., None]
        normed = normed.astype(text_hidden_states.dtype)

        video_in = normed * math.sqrt(cfg.video_hidden_dim / cfg.caption_channels)
        audio_in = normed * math.sqrt(cfg.audio_hidden_dim / cfg.caption_channels)
        video = self.video_connector(self.video_text_proj_in(video_in), attention_mask)
        audio = self.audio_connector(self.audio_text_proj_in(audio_in), attention_mask)
        return video, audio
