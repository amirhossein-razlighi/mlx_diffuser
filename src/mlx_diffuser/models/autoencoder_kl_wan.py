"""AutoencoderKLWan: a faithful MLX port of WAN 2.1's causal 3D video VAE.

This mirrors diffusers' ``AutoencoderKLWan`` (the ``is_residual=False`` / WAN-2.1
configuration) module-for-module so the official weights load directly via the
converter. Tensors are channels-last ``(B, T, H, W, C)``.

WAN's VAE is *causal in time* and processes a clip in temporal chunks, carrying a
small feature cache (the last two frames before each kernel-3 temporal conv)
across chunk boundaries. This is what makes the very first frame map to the first
latent frame and groups of four frames map to one latent frame each — so the cache
machinery is reproduced exactly here, not approximated.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn

from ..configuration import Config
from ..modeling import ModelMixin
from .autoencoder_kl import DiagonalGaussian

CACHE_T = 2

# Default per-channel latent statistics (WAN 2.1, z_dim=16).
_LATENTS_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]  # fmt: skip
_LATENTS_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]  # fmt: skip


@dataclasses.dataclass
class AutoencoderKLWanConfig(Config):
    base_dim: int = 96
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    in_channels: int = 3
    out_channels: int = 3
    latents_mean: tuple[float, ...] = tuple(_LATENTS_MEAN)
    latents_std: tuple[float, ...] = tuple(_LATENTS_STD)

    @property
    def temperal_upsample(self) -> tuple[bool, ...]:
        return tuple(reversed(self.temperal_downsample))


# --- primitives --------------------------------------------------------------


class WanCausalConv3d(nn.Conv3d):
    """3D conv with causal time padding and a streaming feature cache.

    Padding is applied manually (the parent runs with ``padding=0``): time is
    padded ``2 * pad_t`` on the past side only, height/width symmetrically. When a
    ``cache_x`` (the tail of the previous chunk) is supplied, it is prepended and
    the left time-padding is reduced accordingly — reproducing WAN's causal
    streaming across chunk boundaries.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        ks = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        st = (stride,) * 3 if isinstance(stride, int) else tuple(stride)
        pad = (padding,) * 3 if isinstance(padding, int) else tuple(padding)
        super().__init__(in_channels, out_channels, ks, stride=st, padding=0)
        self._pad = pad

    def __call__(self, x: mx.array, cache_x: mx.array | None = None) -> mx.array:
        pt, ph, pw = self._pad
        pad_t = 2 * pt
        if cache_x is not None and pad_t > 0:
            x = mx.concatenate([cache_x, x], axis=1)
            pad_t -= cache_x.shape[1]
        x = mx.pad(x, [(0, 0), (pad_t, 0), (ph, ph), (pw, pw), (0, 0)])
        return super().__call__(x)


class WanRMSNorm(nn.Module):
    """L2 channel normalization scaled by ``sqrt(dim)`` (== RMSNorm), channels-last."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = mx.ones((dim,))
        self._scale = dim**0.5

    def __call__(self, x: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        norm = mx.rsqrt(mx.sum(xf * xf, axis=-1, keepdims=True) + 1e-12)
        return (xf * norm * self._scale).astype(x.dtype) * self.gamma


def _spatial_pad_right(x: mx.array) -> mx.array:
    """ZeroPad2d((0,1,0,1)) on channels-last ``(N, H, W, C)`` (pad bottom/right)."""
    return mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])


class WanResample(nn.Module):
    """Spatial (and optional temporal) up/downsampling, matching diffusers keys.

    ``resample`` is a 2-element list ``[op, conv]`` so the convolution's weight key
    is ``resample.1.*`` exactly as in the diffusers checkpoint (``op`` at index 0
    is the parameter-free pad/upsample).
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int | None = None):
        super().__init__()
        self.mode = mode
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode in ("upsample2d", "upsample3d"):
            up = nn.Upsample(scale_factor=2.0, mode="nearest")
            self.resample = [up, nn.Conv2d(dim, upsample_out_dim, 3, padding=1)]
            if mode == "upsample3d":
                self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode in ("downsample2d", "downsample3d"):
            self.resample = [_SpatialPad(), nn.Conv2d(dim, dim, 3, stride=2)]
            if mode == "downsample3d":
                self.time_conv = WanCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1))
        else:
            self.resample = [_Identity(), _Identity()]

    def _spatial(self, x: mx.array) -> mx.array:
        # x: (B, T, H, W, C) -> per-frame 2D op -> (B, T, H', W', C)
        b, t, h, w, c = x.shape
        x = x.reshape(b * t, h, w, c)
        x = self.resample[1](self.resample[0](x))
        return x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3])

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, -CACHE_T:]
                if cache_x.shape[1] < 2 and feat_cache[idx] != "Rep":
                    cache_x = mx.concatenate([feat_cache[idx][:, -1:], cache_x], axis=1)
                if cache_x.shape[1] < 2 and feat_cache[idx] == "Rep":
                    cache_x = mx.concatenate([mx.zeros_like(cache_x), cache_x], axis=1)
                if feat_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                # split doubled channels into a doubled time axis
                b, t, h, w, c = x.shape
                x = x.reshape(b, t, h, w, 2, c // 2)
                x = mx.stack([x[..., 0, :], x[..., 1, :]], axis=2)
                x = x.reshape(b, t * 2, h, w, c // 2)

        x = self._spatial(x)

        if self.mode == "downsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x
                feat_idx[0] += 1
            else:
                cache_x = x[:, -1:]
                x = self.time_conv(mx.concatenate([feat_cache[idx][:, -1:], x], axis=1))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x


class _Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class _SpatialPad(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return _spatial_pad_right(x)


def _cached_conv(conv: WanCausalConv3d, x: mx.array, feat_cache, feat_idx) -> mx.array:
    """Run a causal conv, reading/writing its slot in the streaming feature cache."""
    if feat_cache is None:
        return conv(x)
    idx = feat_idx[0]
    cache_x = x[:, -CACHE_T:]
    if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
        cache_x = mx.concatenate([feat_cache[idx][:, -1:], cache_x], axis=1)
    out = conv(x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return out


class WanResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm1 = WanRMSNorm(in_dim)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMSNorm(out_dim)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else _Identity()
        )

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        h = self.conv_shortcut(x)
        x = nn.silu(self.norm1(x))
        x = _cached_conv(self.conv1, x, feat_cache, feat_idx)
        x = nn.silu(self.norm2(x))
        x = _cached_conv(self.conv2, x, feat_cache, feat_idx)
        return x + h


class WanAttentionBlock(nn.Module):
    """Single-head spatial self-attention applied per frame (1x1 conv qkv/proj)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = WanRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        b, t, h, w, c = x.shape
        y = x.reshape(b * t, h, w, c)
        y = self.norm(y)
        qkv = self.to_qkv(y)  # (b*t, h, w, 3c)
        qkv = qkv.reshape(b * t, h * w, 3 * c)
        q, k, v = mx.split(qkv, 3, axis=-1)
        # single head: (b*t, 1, hw, c)
        q, k, v = (z[:, None] for z in (q, k, v))
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=c**-0.5)
        out = out[:, 0].reshape(b * t, h, w, c)
        out = self.proj(out).reshape(b, t, h, w, c)
        return out + identity


class WanMidBlock(nn.Module):
    def __init__(self, dim: int, num_layers: int = 1):
        super().__init__()
        self.resnets = [WanResidualBlock(dim, dim)]
        self.attentions: list[WanAttentionBlock] = []
        for _ in range(num_layers):
            self.attentions.append(WanAttentionBlock(dim))
            self.resnets.append(WanResidualBlock(dim, dim))

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        x = self.resnets[0](x, feat_cache, feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:], strict=True):
            x = attn(x)
            x = resnet(x, feat_cache, feat_idx)
        return x


class WanUpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, upsample_mode):
        super().__init__()
        self.resnets: list[WanResidualBlock] = []
        cur = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(WanResidualBlock(cur, out_dim))
            cur = out_dim
        self.upsamplers = (
            [WanResample(out_dim, mode=upsample_mode)] if upsample_mode is not None else None
        )

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, feat_cache, feat_idx)
        return x


# --- encoder / decoder -------------------------------------------------------


class WanEncoder3d(nn.Module):
    def __init__(self, cfg: AutoencoderKLWanConfig):
        super().__init__()
        dim, dim_mult = cfg.base_dim, list(cfg.dim_mult)
        dims = [dim * u for u in [1] + dim_mult]
        n = len(dim_mult)

        self.conv_in = WanCausalConv3d(cfg.in_channels, dims[0], 3, padding=1)
        self.down_blocks: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            cur = in_dim
            for _ in range(cfg.num_res_blocks):
                self.down_blocks.append(WanResidualBlock(cur, out_dim))
                cur = out_dim
            if i != n - 1:
                mode = "downsample3d" if cfg.temperal_downsample[i] else "downsample2d"
                self.down_blocks.append(WanResample(out_dim, mode=mode))
        self.mid_block = WanMidBlock(dims[-1])
        self.norm_out = WanRMSNorm(dims[-1])
        self.conv_out = WanCausalConv3d(dims[-1], cfg.z_dim * 2, 3, padding=1)

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        x = _cached_conv(self.conv_in, x, feat_cache, feat_idx)
        for layer in self.down_blocks:
            x = layer(x, feat_cache, feat_idx)
        x = self.mid_block(x, feat_cache, feat_idx)
        x = nn.silu(self.norm_out(x))
        x = _cached_conv(self.conv_out, x, feat_cache, feat_idx)
        return x


class WanDecoder3d(nn.Module):
    def __init__(self, cfg: AutoencoderKLWanConfig):
        super().__init__()
        dim, dim_mult = cfg.base_dim, list(cfg.dim_mult)
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        n = len(dim_mult)
        tup = cfg.temperal_upsample

        self.conv_in = WanCausalConv3d(cfg.z_dim, dims[0], 3, padding=1)
        self.mid_block = WanMidBlock(dims[0])
        self.up_blocks: list[WanUpBlock] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            if i > 0:
                in_dim = in_dim // 2
            up_flag = i != n - 1
            mode = None
            if up_flag:
                mode = "upsample3d" if tup[i] else "upsample2d"
            self.up_blocks.append(WanUpBlock(in_dim, out_dim, cfg.num_res_blocks, mode))
        self.norm_out = WanRMSNorm(dims[-1])
        self.conv_out = WanCausalConv3d(dims[-1], cfg.out_channels, 3, padding=1)

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        x = _cached_conv(self.conv_in, x, feat_cache, feat_idx)
        x = self.mid_block(x, feat_cache, feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)
        x = nn.silu(self.norm_out(x))
        x = _cached_conv(self.conv_out, x, feat_cache, feat_idx)
        return x


def _count_causal_convs(module: nn.Module) -> int:
    return sum(isinstance(m, WanCausalConv3d) for _, m in _iter_modules(module))


def _iter_modules(module: nn.Module):
    yield "", module
    for name, child in module.children().items():
        if isinstance(child, nn.Module):
            for sub, m in _iter_modules(child):
                yield f"{name}.{sub}" if sub else name, m
        elif isinstance(child, list):
            for i, item in enumerate(child):
                if isinstance(item, nn.Module):
                    for sub, m in _iter_modules(item):
                        yield f"{name}.{i}.{sub}" if sub else f"{name}.{i}", m


class AutoencoderKLWan(ModelMixin[AutoencoderKLWanConfig]):
    """WAN 2.1 causal 3D VAE. Channels-last ``(B, T, H, W, C)``."""

    config_class = AutoencoderKLWanConfig

    def __init__(self, config: AutoencoderKLWanConfig):
        super().__init__()
        self.config = config
        self.encoder = WanEncoder3d(config)
        self.quant_conv = WanCausalConv3d(config.z_dim * 2, config.z_dim * 2, 1)
        self.post_quant_conv = WanCausalConv3d(config.z_dim, config.z_dim, 1)
        self.decoder = WanDecoder3d(config)
        self._enc_conv_num = _count_causal_convs(self.encoder)
        self._dec_conv_num = _count_causal_convs(self.decoder)

    # --- latent normalization helpers (used by the diffusion pipeline) -------
    @property
    def latents_mean(self) -> mx.array:
        return mx.array(self.config.latents_mean).reshape(1, 1, 1, 1, -1)

    @property
    def latents_std(self) -> mx.array:
        return mx.array(self.config.latents_std).reshape(1, 1, 1, 1, -1)

    def normalize_latents(self, z: mx.array) -> mx.array:
        """Pixel-latent -> normalized latent space the transformer operates in."""
        return (z - self.latents_mean) / self.latents_std

    def denormalize_latents(self, z: mx.array) -> mx.array:
        return z * self.latents_std + self.latents_mean

    # --- encode / decode -----------------------------------------------------
    def encode(self, x: mx.array) -> DiagonalGaussian:
        """Encode ``(B, T, H, W, C)`` video into a latent ``DiagonalGaussian``.

        The clip is processed causally in temporal chunks (frame 0 alone, then
        groups of 4), accumulating ``1 + (T-1)//4`` latent frames.
        """
        num_frames = x.shape[1]
        feat_cache = [None] * self._enc_conv_num
        feat_idx = [0]
        iters = 1 + (num_frames - 1) // 4
        outs = []
        for i in range(iters):
            feat_idx[0] = 0
            chunk = x[:, :1] if i == 0 else x[:, 1 + 4 * (i - 1) : 1 + 4 * i]
            outs.append(self.encoder(chunk, feat_cache, feat_idx))
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)
        return DiagonalGaussian(self.quant_conv(out))

    def decode(self, z: mx.array) -> mx.array:
        """Decode a latent ``(B, T, H, W, C)`` back to pixels in ``[-1, 1]``."""
        num_frames = z.shape[1]
        x = self.post_quant_conv(z)
        feat_cache = [None] * self._dec_conv_num
        feat_idx = [0]
        outs = []
        for i in range(num_frames):
            feat_idx[0] = 0
            outs.append(self.decoder(x[:, i : i + 1], feat_cache, feat_idx))
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)
        return mx.clip(out, -1.0, 1.0)

    def __call__(
        self, x: mx.array, key: mx.array | None = None
    ) -> tuple[mx.array, DiagonalGaussian]:
        posterior = self.encode(x)
        return self.decode(posterior.mode()), posterior
