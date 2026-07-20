"""DINOv2 with register tokens, ported to MLX for TRELLIS conditioning."""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn

from ..configuration import Config
from ..layers.trellis import LayerNorm32
from ..modeling import ModelMixin


@dataclasses.dataclass
class DINOv2Config(Config):
    image_size: int = 518
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    mlp_ratio: int = 4
    num_register_tokens: int = 4
    layer_norm_eps: float = 1e-6
    layerscale_value: float = 1.0
    qkv_bias: bool = True

    @classmethod
    def vitl14_registers(cls) -> DINOv2Config:
        return cls()


class _PatchEmbeddings(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.projection(x)
        return x.reshape(x.shape[0], -1, x.shape[-1])


class _DINOEmbeddings(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        patches_per_axis = config.image_size // config.patch_size
        self.image_size = config.image_size
        self.patch_embeddings = _PatchEmbeddings(config)
        self.cls_token = mx.zeros((1, 1, config.hidden_size))
        self.mask_token = mx.zeros((1, config.hidden_size))
        self.position_embeddings = mx.zeros(
            (1, 1 + patches_per_axis**2, config.hidden_size)
        )
        self.register_tokens = mx.zeros(
            (1, config.num_register_tokens, config.hidden_size)
        )

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 4 or x.shape[1:3] != (self.image_size, self.image_size):
            raise ValueError(
                f"DINOv2 input must have shape (B, {self.image_size}, {self.image_size}, 3), "
                f"got {x.shape}"
            )
        patches = self.patch_embeddings(x)
        cls = mx.broadcast_to(self.cls_token, (x.shape[0], 1, self.cls_token.shape[-1]))
        tokens = mx.concatenate([cls, patches], axis=1) + self.position_embeddings
        registers = mx.broadcast_to(
            self.register_tokens,
            (x.shape[0], self.register_tokens.shape[1], self.register_tokens.shape[2]),
        )
        return mx.concatenate([tokens[:, :1], registers, tokens[:, 1:]], axis=1)


class _DINOSelfAttention(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)

    def _heads(self, x: mx.array) -> mx.array:
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

    def __call__(self, x: mx.array) -> mx.array:
        q, k, v = self._heads(self.query(x)), self._heads(self.key(x)), self._heads(self.value(x))
        h = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
        return h.transpose(0, 2, 1, 3).reshape(x.shape)


class _DINOAttentionOutput(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.dense(x)


class _DINOAttention(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.attention = _DINOSelfAttention(config)
        self.output = _DINOAttentionOutput(config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.output(self.attention(x))


class _DINOLayerScale(nn.Module):
    def __init__(self, hidden_size: int, value: float):
        super().__init__()
        self.lambda1 = mx.full((hidden_size,), value)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.lambda1


class _DINOMLP(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        hidden = config.hidden_size * config.mlp_ratio
        self.fc1 = nn.Linear(config.hidden_size, hidden)
        self.fc2 = nn.Linear(hidden, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class _DINOEncoderLayer(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, affine=True
        )
        self.attention = _DINOAttention(config)
        self.layer_scale1 = _DINOLayerScale(config.hidden_size, config.layerscale_value)
        self.norm2 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, affine=True
        )
        self.mlp = _DINOMLP(config)
        self.layer_scale2 = _DINOLayerScale(config.hidden_size, config.layerscale_value)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class _DINOEncoder(nn.Module):
    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.layer = [_DINOEncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x: mx.array, *, low_memory: bool = False) -> mx.array:
        for layer in self.layer:
            x = layer(x)
            if low_memory:
                mx.eval(x)
        return x


class DINOv2Model(ModelMixin[DINOv2Config]):
    """Hugging Face-compatible DINOv2 register-token vision transformer."""

    config_class = DINOv2Config

    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.config = config
        self.embeddings = _DINOEmbeddings(config)
        self.encoder = _DINOEncoder(config)
        self.layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, affine=True
        )

    def __call__(
        self,
        pixel_values: mx.array,
        *,
        return_prenorm: bool = False,
        low_memory: bool = False,
    ) -> mx.array:
        hidden = self.encoder(self.embeddings(pixel_values), low_memory=low_memory)
        return hidden if return_prenorm else self.layernorm(hidden)

    def trellis_conditioning(self, pixel_values: mx.array, *, low_memory: bool = False) -> mx.array:
        """Return the plain-normalized pre-final-norm tokens used to train TRELLIS."""

        hidden = self(pixel_values, return_prenorm=True, low_memory=low_memory)
        return LayerNorm32(self.config.hidden_size, affine=False, eps=1e-5)(hidden)


__all__ = ["DINOv2Config", "DINOv2Model"]
