"""LTX-2 audio decode stack: the audio VAE decoder and the BigVGAN vocoder.

LTX-2 denoises audio latents jointly with the video; this module turns those
latents into a waveform. Two models, mirroring ltx-core:

* :class:`LTX2AudioDecoder` — the decoder half of the audio VAE. Latent tokens
  ``(B, T, 128)`` (8 channels x 16 mel bins, 25 tokens/s) are decoded into a
  stereo log-mel spectrogram ``(B, 4T-3, 64, 2)`` at 100 frames/s. VQGAN-style
  conv stack with parameter-free pixel norms; convolutions are causal along
  time (zero padding on the left only).
* :class:`LTX2Vocoder` — ltx-core's ``VocoderWithBWE``: a BigVGAN-v2 generator
  (snake-beta activations with anti-aliased up/down resampling) synthesizes a
  16 kHz stereo waveform from the mel spectrogram, then a bandwidth-extension
  generator predicts a 48 kHz residual from the 16 kHz output's own mel (via a
  causal STFT whose bases ship in the checkpoint), added to a sinc-resampled
  skip. All anti-aliasing filters are loaded from the checkpoint.

Both models are small (~90 MB / ~260 MB bf16) and must run in **float32**:
bfloat16 accumulation across the vocoder's 100+ sequential convolutions
audibly degrades the spectrum (see ltx-core's vocoder notes). Tensors are
channels-last: spectrograms ``(B, T, mel, C)``, waveforms ``(B, T, C)``.
"""

from __future__ import annotations

import dataclasses
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..configuration import Config
from ..modeling import ModelMixin


def _pixel_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


# --- audio VAE decoder ----------------------------------------------------------


@dataclasses.dataclass
class LTX2AudioDecoderConfig(Config):
    z_channels: int = 8
    out_channels: int = 2  # stereo
    base_channels: int = 128
    ch_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    mel_bins: int = 64
    latent_downsample_factor: int = 4

    @classmethod
    def ltx_2_3(cls) -> LTX2AudioDecoderConfig:
        return cls()  # defaults match the LTX-2.3 checkpoint


class AudioCausalConv2d(nn.Module):
    """Conv over (time, mel): causal zero padding in time, symmetric in mel."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.time_pad = kernel_size - 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(0, kernel_size // 2))

    def __call__(self, x: mx.array) -> mx.array:  # (B, T, M, C)
        if self.time_pad:
            x = mx.pad(x, [(0, 0), (self.time_pad, 0), (0, 0), (0, 0)])
        return self.conv(x)


class AudioResnetBlock(nn.Module):
    """pixel-norm -> silu -> conv, twice; 1x1 shortcut when channels change."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = AudioCausalConv2d(in_channels, out_channels)
        self.conv2 = AudioCausalConv2d(out_channels, out_channels)
        if in_channels != out_channels:
            self.nin_shortcut = AudioCausalConv2d(in_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nn.silu(_pixel_norm(x)))
        h = self.conv2(nn.silu(_pixel_norm(h)))
        if "nin_shortcut" in self:
            x = self.nin_shortcut(x)
        return x + h


class AudioUpsample(nn.Module):
    """Nearest 2x in (time, mel) + conv; drops the first time row (causal undo)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = AudioCausalConv2d(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        b, t, m, c = x.shape
        x = mx.broadcast_to(x[:, :, None, :, None, :], (b, t, 2, m, 2, c))
        x = x.reshape(b, 2 * t, 2 * m, c)
        return self.conv(x)[:, 1:]


class _AudioUpStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, upsample: bool):
        super().__init__()
        self.block = [
            AudioResnetBlock(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_blocks)
        ]
        if upsample:
            self.upsample = AudioUpsample(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.block:
            x = block(x)
        if "upsample" in self:
            x = self.upsample(x)
        return x


class _AudioMidBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block_1 = AudioResnetBlock(channels, channels)
        self.block_2 = AudioResnetBlock(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        return self.block_2(self.block_1(x))


class LTX2AudioDecoder(ModelMixin[LTX2AudioDecoderConfig]):
    """Audio VAE decoder: latent tokens ``(B, T, 128)`` -> mel ``(B, 4T-3, 64, 2)``."""

    config_class = LTX2AudioDecoderConfig

    def __init__(self, config: LTX2AudioDecoderConfig):
        super().__init__()
        self.config = config
        ch = config.base_channels
        top = ch * config.ch_mult[-1]
        self.conv_in = AudioCausalConv2d(config.z_channels, top)
        self.mid = _AudioMidBlock(top)
        # up[level]: run from the highest level down to 0; level 0 has no upsample.
        self.up = []
        block_in = top
        for level, mult in enumerate(config.ch_mult):
            block_out = ch * mult
            # walked in reversed order, so up[-1] sees `top` input channels
            self.up.append(
                _AudioUpStage(
                    in_channels=block_in
                    if level == len(config.ch_mult) - 1
                    else ch * config.ch_mult[level + 1],
                    out_channels=block_out,
                    num_blocks=config.num_res_blocks + 1,
                    upsample=level != 0,
                )
            )
        self.conv_out = AudioCausalConv2d(ch * config.ch_mult[0], config.out_channels)
        packed = config.z_channels * config.mel_bins // config.latent_downsample_factor
        self.latents_mean = mx.zeros((packed,))
        self.latents_std = mx.ones((packed,))

    def decode(self, tokens: mx.array) -> mx.array:
        """Decode packed audio latent tokens ``(B, T, 128)`` to a mel spectrogram.

        Returns ``(B, 4T-3, mel_bins, 2)`` (log-mel, stereo, 100 frames/s).
        """
        cfg = self.config
        z = tokens * self.latents_std + self.latents_mean
        b, t, _ = z.shape
        latent_bins = cfg.mel_bins // cfg.latent_downsample_factor
        # packed channel index is (c, mel): unpack then move channels last
        z = z.reshape(b, t, cfg.z_channels, latent_bins).transpose(0, 1, 3, 2)

        h = self.mid(self.conv_in(z))
        for stage in reversed(self.up):
            h = stage(h)
        h = self.conv_out(nn.silu(_pixel_norm(h)))
        target_t = t * cfg.latent_downsample_factor - (cfg.latent_downsample_factor - 1)
        return h[:, :target_t, : cfg.mel_bins, : cfg.out_channels]


# --- BigVGAN vocoder with bandwidth extension -------------------------------------


@dataclasses.dataclass
class LTX2VocoderConfig(Config):
    mel_bins: int = 64
    # main generator: mel (100 frames/s) -> 16 kHz waveform (x160)
    upsample_rates: tuple[int, ...] = (5, 2, 2, 2, 2, 2)
    upsample_kernel_sizes: tuple[int, ...] = (11, 4, 4, 4, 4, 4)
    upsample_initial_channel: int = 1536
    # bandwidth extension: mel of the 16 kHz output -> 48 kHz residual (x240)
    bwe_upsample_rates: tuple[int, ...] = (6, 5, 2, 2, 2)
    bwe_upsample_kernel_sizes: tuple[int, ...] = (12, 11, 4, 4, 4)
    bwe_upsample_initial_channel: int = 512
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilations: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    input_sampling_rate: int = 16000
    output_sampling_rate: int = 48000
    stft_n_fft: int = 512
    stft_hop_length: int = 80

    @classmethod
    def ltx_2_3(cls) -> LTX2VocoderConfig:
        return cls()  # defaults match the LTX-2.3 checkpoint


def _per_channel_conv(x: mx.array, filt: mx.array, *, stride: int, transpose: bool) -> mx.array:
    """Apply the same 1-channel filter to every channel of ``x`` (B, L, C)."""
    b, length, c = x.shape
    y = x.transpose(0, 2, 1).reshape(b * c, length, 1)
    w = filt.reshape(1, -1, 1)  # (C_out=1, K, C_in=1)
    y = mx.conv_transpose1d(y, w, stride=stride) if transpose else mx.conv1d(y, w, stride=stride)
    return y.reshape(b, c, -1).transpose(0, 2, 1)


class SnakeBeta(nn.Module):
    """BigVGAN-v2 snake activation: ``x + sin^2(alpha x) / beta`` (log-scale params)."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha)
        beta = mx.exp(self.beta)
        return x + mx.sin(x * alpha) ** 2 / (beta + 1e-9)


class _AntiAliasUpsample(nn.Module):
    """2x sinc upsampling (kaiser filter loaded from the checkpoint)."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.ratio = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.filter = mx.zeros((1, 1, kernel_size))

    def __call__(self, x: mx.array) -> mx.array:  # (B, L, C)
        x = mx.pad(x, [(0, 0), (self.pad, self.pad), (0, 0)], mode="edge")
        y = self.ratio * _per_channel_conv(x, self.filter, stride=self.ratio, transpose=True)
        return y[:, self.pad_left : y.shape[1] - self.pad_right]


class _LowPassFilter(nn.Module):
    def __init__(self, kernel_size: int = 12, stride: int = 2):
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.filter = mx.zeros((1, 1, kernel_size))

    def __call__(self, x: mx.array) -> mx.array:  # (B, L, C)
        x = mx.pad(x, [(0, 0), (self.pad_left, self.pad_right), (0, 0)], mode="edge")
        return _per_channel_conv(x, self.filter, stride=self.stride, transpose=False)


class _AntiAliasDownsample(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.lowpass = _LowPassFilter(kernel_size, stride=ratio)

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class Activation1d(nn.Module):
    """Anti-aliased activation: 2x upsample -> snake -> 2x downsample."""

    def __init__(self, channels: int):
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample = _AntiAliasUpsample()
        self.downsample = _AntiAliasDownsample()

    def __call__(self, x: mx.array) -> mx.array:
        return self.downsample(self.act(self.upsample(x)))


class AMPBlock(nn.Module):
    """BigVGAN residual block: (act -> dilated conv -> act -> conv) x 3."""

    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...]):
        super().__init__()
        self.convs1 = [
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=d,
                padding=(kernel_size * d - d) // 2,
            )
            for d in dilations
        ]
        self.convs2 = [
            nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
            for _ in dilations
        ]
        self.acts1 = [Activation1d(channels) for _ in dilations]
        self.acts2 = [Activation1d(channels) for _ in dilations]

    def __call__(self, x: mx.array) -> mx.array:
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2, strict=True):
            x = x + c2(a2(c1(a1(x))))
        return x


class BigVGAN(nn.Module):
    """BigVGAN-v2 generator: stereo mel ``(B, T, mel, 2)`` -> waveform ``(B, L, 2)``."""

    def __init__(
        self,
        cfg: LTX2VocoderConfig,
        rates: tuple[int, ...],
        kernel_sizes: tuple[int, ...],
        initial_channel: int,
        *,
        final_clamp: bool,
    ):
        super().__init__()
        self.final_clamp = final_clamp
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.conv_pre = nn.Conv1d(2 * cfg.mel_bins, initial_channel, 7, padding=3)
        self.ups = [
            nn.ConvTranspose1d(
                initial_channel // 2**i,
                initial_channel // 2 ** (i + 1),
                k,
                stride=r,
                padding=(k - r) // 2,
            )
            for i, (r, k) in enumerate(zip(rates, kernel_sizes, strict=True))
        ]
        self.resblocks = [
            AMPBlock(initial_channel // 2 ** (i + 1), k, d)
            for i in range(len(rates))
            for k, d in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilations, strict=True)
        ]
        final_channels = initial_channel // 2 ** len(rates)
        self.act_post = Activation1d(final_channels)
        self.conv_post = nn.Conv1d(final_channels, 2, 7, padding=3, bias=False)

    def __call__(self, mel: mx.array) -> mx.array:
        b, t, m, s = mel.shape
        x = mel.transpose(0, 1, 3, 2).reshape(b, t, s * m)  # stereo-major channel packing
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = up(x)
            blocks = self.resblocks[i * self.num_kernels : (i + 1) * self.num_kernels]
            acc = blocks[0](x)
            for block in blocks[1:]:
                acc = acc + block(x)
            x = acc / self.num_kernels
        x = self.conv_post(self.act_post(x))
        return mx.clip(x, -1.0, 1.0) if self.final_clamp else x


class _STFTFn(nn.Module):
    """Causal STFT as a strided conv with checkpoint-stored DFT x window bases."""

    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.hop_length = hop_length
        self.left_pad = n_fft - hop_length
        self.forward_basis = mx.zeros((2 * (n_fft // 2 + 1), 1, n_fft))

    def magnitude(self, y: mx.array) -> mx.array:  # (B, L) -> (B, frames, n_freqs)
        y = mx.pad(y[..., None], [(0, 0), (self.left_pad, 0), (0, 0)])
        spec = mx.conv1d(y, self.forward_basis.transpose(0, 2, 1), stride=self.hop_length)
        n_freqs = spec.shape[-1] // 2
        real, imag = spec[..., :n_freqs], spec[..., n_freqs:]
        return mx.sqrt(real * real + imag * imag)


class MelSTFT(nn.Module):
    """Causal log-mel spectrogram whose bases are loaded from the checkpoint."""

    def __init__(self, n_fft: int, hop_length: int, n_mels: int):
        super().__init__()
        self.stft_fn = _STFTFn(n_fft, hop_length)
        self.mel_basis = mx.zeros((n_mels, n_fft // 2 + 1))

    def __call__(self, y: mx.array) -> mx.array:  # (B, L) -> (B, frames, n_mels)
        mel = self.stft_fn.magnitude(y) @ self.mel_basis.T
        return mx.log(mx.maximum(mel, 1e-5))


def _hann_resample_filter(ratio: int) -> tuple[np.ndarray, int, int, int]:
    """Hann-windowed sinc filter matching torchaudio's resampler (ltx-core BWE skip)."""
    rolloff, width = 0.99, math.ceil(6 / 0.99)
    kernel_size = 2 * width * ratio + 1
    time = (np.arange(kernel_size, dtype=np.float64) / ratio - width) * rolloff
    window = np.cos(np.clip(time, -6, 6) * math.pi / 12) ** 2
    filt = np.sinc(time) * window * rolloff / ratio
    return filt.astype(np.float32), width, 2 * width * ratio, kernel_size - ratio


class LTX2Vocoder(ModelMixin[LTX2VocoderConfig]):
    """Mel ``(B, T, mel, 2)`` -> 48 kHz stereo waveform ``(B, L, 2)`` in [-1, 1].

    Run in float32 (load with ``dtype=mx.float32``).
    """

    config_class = LTX2VocoderConfig

    def __init__(self, config: LTX2VocoderConfig):
        super().__init__()
        self.config = config
        self.vocoder = BigVGAN(
            config,
            config.upsample_rates,
            config.upsample_kernel_sizes,
            config.upsample_initial_channel,
            final_clamp=True,
        )
        self.bwe_generator = BigVGAN(
            config,
            config.bwe_upsample_rates,
            config.bwe_upsample_kernel_sizes,
            config.bwe_upsample_initial_channel,
            final_clamp=False,
        )
        self.mel_stft = MelSTFT(config.stft_n_fft, config.stft_hop_length, config.mel_bins)
        ratio = config.output_sampling_rate // config.input_sampling_rate
        filt, pad, pad_left, pad_right = _hann_resample_filter(ratio)
        self._ratio = ratio
        self._resample = (mx.array(filt), pad, pad_left, pad_right)

    def _resample_skip(self, x: mx.array) -> mx.array:  # (B, L, C) 16 kHz -> 48 kHz
        filt, pad, pad_left, pad_right = self._resample
        x = mx.pad(x, [(0, 0), (pad, pad), (0, 0)], mode="edge")
        y = self._ratio * _per_channel_conv(x, filt, stride=self._ratio, transpose=True)
        return y[:, pad_left : y.shape[1] - pad_right]

    def __call__(self, mel: mx.array) -> mx.array:
        cfg = self.config
        x = self.vocoder(mel)  # (B, L16, 2)
        b, length, channels = x.shape
        out_length = length * cfg.output_sampling_rate // cfg.input_sampling_rate

        remainder = length % cfg.stft_hop_length
        if remainder:
            x = mx.pad(x, [(0, 0), (0, cfg.stft_hop_length - remainder), (0, 0)])

        # BWE: log-mel of the 16 kHz output -> 48 kHz residual + resampled skip.
        flat = x.transpose(0, 2, 1).reshape(b * channels, -1)
        mel16 = self.mel_stft(flat)  # (B*C, frames, mel)
        mel16 = mel16.reshape(b, channels, *mel16.shape[1:]).transpose(0, 2, 3, 1)
        residual = self.bwe_generator(mel16)
        skip = self._resample_skip(x)
        return mx.clip(residual + skip, -1.0, 1.0)[:, :out_length]
