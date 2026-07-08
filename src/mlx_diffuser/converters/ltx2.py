"""LTX-2.3 checkpoint conversion: stream the 46 GB single file into 4-bit MLX.

The official LTX-2.3 release is one ~46 GB safetensors bundle (transformer +
connectors + video VAE + audio stack) plus a separate fp32 Gemma-3-12B text
encoder (~48 GB) — together far larger than both the free disk and the RAM of
the machines this targets. So this converter never materializes the originals:

* the single file is read **remotely** over HTTP range requests, tensor ranges
  coalesced into large fetches, each tensor quantized/cast the moment it
  arrives and flushed to sharded MLX safetensors;
* the Gemma shards are downloaded one at a time, converted, and deleted before
  the next shard is fetched (peak extra disk ~= one 5 GB shard).

Output is an MLX-native checkpoint folder (per component: ``config.json``,
optional ``quantization.json``, sharded weights) that
``ModelMixin.from_pretrained`` loads directly — quantizing the skeleton first,
so nothing bigger than the quantized model is ever resident.

Only the decode path is converted: the video and audio VAE *encoders* are
skipped (text-to-video never encodes pixels or waveforms).
"""

from __future__ import annotations

import json
import shutil
import struct
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from ..modeling import ModelMixin
from ..utils import get_logger
from .base import convert_conv_weight

logger = get_logger()

LTX_REPO = "Lightricks/LTX-2.3"
LTX_FILE = "ltx-2.3-22b-distilled.safetensors"
GEMMA_REPO = "Lightricks/LTX-2"

_DTYPES = {"BF16": mx.bfloat16, "F32": mx.float32, "F16": mx.float16}


class RemoteSafetensors:
    """Random access to a safetensors file over HTTP range requests."""

    def __init__(self, url: str, max_fetch_bytes: int = 256 * 1024**2):
        self.url = url
        self.max_fetch_bytes = max_fetch_bytes
        head = self._get(0, 8)
        header_len = struct.unpack("<Q", head)[0]
        self.header: dict = json.loads(self._get(8, header_len))
        self.metadata: dict = self.header.pop("__metadata__", {})
        self.base = 8 + header_len

    def _get(self, start: int, length: int, retries: int = 6) -> bytes:
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{start + length - 1}"}
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == retries - 1:
                    raise
                wait = 2.0 * (attempt + 1)
                logger.warning("range fetch failed (%s); retrying in %.0fs", exc, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def keys(self) -> list[str]:
        return list(self.header)

    def _to_array(self, key: str, buf: bytes) -> mx.array:
        import numpy as np

        info = self.header[key]
        if info["dtype"] == "BF16":
            raw = mx.array(np.frombuffer(buf, dtype=np.uint16))
            return raw.view(mx.bfloat16).reshape(info["shape"])
        np_dtype = {"F32": np.float32, "F16": np.float16}[info["dtype"]]
        return mx.array(np.frombuffer(buf, dtype=np_dtype)).reshape(info["shape"])

    def iter_tensors(self, keys: list[str]):
        """Yield ``(key, mx.array)`` fetching adjacent tensors in coalesced ranges."""
        keys = sorted(keys, key=lambda k: self.header[k]["data_offsets"][0])
        batch: list[str] = []
        batch_start = batch_end = 0
        for key in keys:
            s, e = self.header[key]["data_offsets"]
            contiguous = batch and s == batch_end
            if batch and (not contiguous or e - batch_start > self.max_fetch_bytes):
                yield from self._emit(batch, batch_start, batch_end)
                batch = []
            if not batch:
                batch_start = s
            batch.append(key)
            batch_end = e
        if batch:
            yield from self._emit(batch, batch_start, batch_end)

    def _emit(self, batch: list[str], start: int, end: int):
        buf = self._get(self.base + start, end - start)
        for key in batch:
            s, e = self.header[key]["data_offsets"]
            yield key, self._to_array(key, buf[s - start : e - start])


# --- key remapping ------------------------------------------------------------

_TRANSFORMER_RENAMES = [
    ("patchify_proj", "proj_in"),
    ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
    ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
    ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
    ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
    ("prompt_adaln_single", "prompt_adaln"),
    ("adaln_single", "time_embed"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
    ("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table"),
    ("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table"),
]

_CONNECTOR_RENAMES = [
    ("text_embedding_projection.video_aggregate_embed", "video_text_proj_in"),
    ("text_embedding_projection.audio_aggregate_embed", "audio_text_proj_in"),
    ("video_embeddings_connector", "video_connector"),
    ("audio_embeddings_connector", "audio_connector"),
    ("transformer_1d_blocks", "transformer_blocks"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
]

_DM = "model.diffusion_model."


def _rename(key: str, renames: list[tuple[str, str]]) -> str:
    for old, new in renames:
        key = key.replace(old, new)
    return key


def map_transformer_key(key: str) -> str | None:
    if not key.startswith(_DM) or "embeddings_connector" in key:
        return None
    return _rename(key.removeprefix(_DM), _TRANSFORMER_RENAMES)


def map_connectors_key(key: str) -> str | None:
    if key.startswith("text_embedding_projection."):
        return _rename(key, _CONNECTOR_RENAMES)
    if key.startswith(_DM) and "embeddings_connector" in key:
        return _rename(key.removeprefix(_DM), _CONNECTOR_RENAMES)
    return None


def map_vae_key(key: str) -> str | None:
    if key == "vae.per_channel_statistics.mean-of-means":
        return "latents_mean"
    if key == "vae.per_channel_statistics.std-of-means":
        return "latents_std"
    if key.startswith("vae.decoder."):
        return key.removeprefix("vae.decoder.")
    return None  # vae.encoder / other statistics / audio_vae / vocoder


def map_audio_decoder_key(key: str) -> str | None:
    if key == "audio_vae.per_channel_statistics.mean-of-means":
        return "latents_mean"
    if key == "audio_vae.per_channel_statistics.std-of-means":
        return "latents_std"
    if key.startswith("audio_vae.decoder."):
        return key.removeprefix("audio_vae.decoder.")
    return None  # audio_vae.encoder


def map_vocoder_key(key: str) -> str | None:
    # inverse STFT basis is unused (we only analyze, never invert)
    if not key.startswith("vocoder.") or key.endswith("inverse_basis"):
        return None
    return key.removeprefix("vocoder.")


def convert_audio_tensor(key: str, tensor: mx.array) -> mx.array:
    """torch conv kernels -> MLX channels-last layouts (audio stack)."""
    if tensor.ndim == 4:  # Conv2d (Cout, Cin, H, W) -> (Cout, H, W, Cin)
        return tensor.transpose(0, 2, 3, 1)
    if tensor.ndim == 3 and key.endswith(".weight"):
        if ".ups." in key:  # ConvTranspose1d (Cin, Cout, K) -> (Cout, K, Cin)
            return tensor.transpose(1, 2, 0)
        return tensor.transpose(0, 2, 1)  # Conv1d (Cout, Cin, K) -> (Cout, K, Cin)
    return tensor  # anti-alias filters, STFT/mel bases, snake params, biases


def map_gemma_key(key: str) -> str | None:
    prefix = "language_model.model."
    if key.startswith(prefix):
        return key.removeprefix(prefix)
    return None  # vision tower, multi-modal projector, lm_head


# --- streaming component writer -------------------------------------------------


class _ComponentWriter:
    """Quantize/cast incoming tensors against a model skeleton and write shards."""

    def __init__(
        self,
        model: ModelMixin,
        out_dir: Path,
        *,
        quantize: int | None,
        group_size: int = 64,
        dtype: mx.Dtype = mx.bfloat16,
        shard_bytes: int = 2 * 1024**3,
    ):
        from ..quantization import quantize_module

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / ".incomplete").touch()
        self.dtype = dtype
        self.bits = quantize
        self.group_size = group_size
        self.shard_bytes = shard_bytes

        model.config.save(self.out_dir)
        if quantize is not None:
            quantize_module(model, bits=quantize, group_size=group_size)
            (self.out_dir / "quantization.json").write_text(
                json.dumps({"bits": quantize, "group_size": group_size})
            )
        self.expected = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}  # type: ignore[union-attr, str-unpack]
        self.quantized_prefixes = {
            k.removesuffix(".scales") for k in self.expected if k.endswith(".scales")
        }
        self.written: dict[str, tuple[int, ...]] = {}
        self.pending: dict[str, mx.array] = {}
        self.pending_bytes = 0
        self.shard_idx = 0
        del model

    def add(self, name: str, tensor: mx.array) -> None:
        prefix = name.removesuffix(".weight")
        if name.endswith(".weight") and prefix in self.quantized_prefixes:
            w, scales, biases = mx.quantize(
                tensor.astype(self.dtype), group_size=self.group_size, bits=self.bits
            )
            self._stage(
                {f"{prefix}.weight": w, f"{prefix}.scales": scales, f"{prefix}.biases": biases}
            )
            return
        if tensor.ndim == 5:  # conv kernels -> channels-last
            tensor = convert_conv_weight(tensor)
        if tensor.dtype in (mx.bfloat16, mx.float16, mx.float32):
            expected_fp32 = name in self.expected and tensor.dtype == mx.float32
            tensor = tensor if expected_fp32 else tensor.astype(self.dtype)
        self._stage({name: tensor})

    def _stage(self, items: dict[str, mx.array]) -> None:
        for k, v in items.items():
            if k not in self.expected:
                raise KeyError(f"unexpected converted key: {k}")
            if tuple(v.shape) != self.expected[k]:
                raise ValueError(f"{k}: shape {tuple(v.shape)} != expected {self.expected[k]}")
            self.pending[k] = v
            self.written[k] = tuple(v.shape)
            self.pending_bytes += v.nbytes
        if self.pending_bytes >= self.shard_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self.pending:
            return
        self.shard_idx += 1
        path = self.out_dir / f"model-{self.shard_idx:05d}.safetensors"
        mx.eval(list(self.pending.values()))
        mx.save_safetensors(str(path), self.pending)
        logger.info("wrote %s (%.2f GB)", path.name, self.pending_bytes / 1e9)
        self.pending = {}
        self.pending_bytes = 0
        mx.clear_cache()

    def flush(self) -> None:
        self._flush()

    def finish(self) -> None:
        self._flush()
        missing = sorted(set(self.expected) - set(self.written))
        if missing:
            raise ValueError(f"missing {len(missing)} keys, e.g. {missing[:5]}")
        (self.out_dir / ".incomplete").unlink(missing_ok=True)


# --- top-level conversion -------------------------------------------------------


def convert_ltx2_checkpoint(
    out_dir: str | Path,
    *,
    repo: str = LTX_REPO,
    filename: str = LTX_FILE,
    gemma_repo: str = GEMMA_REPO,
    quantize: int = 4,
    quantize_connectors: int = 8,
) -> Path:
    """Stream-convert LTX-2.3 into an MLX checkpoint folder (~20 GB on disk).

    Produces ``transformer/`` (quantized), ``connectors/`` (quantized),
    ``vae_decoder/``, ``audio_decoder/`` and ``vocoder/`` (bf16),
    ``text_encoder/`` (quantized Gemma-3-12B), and ``tokenizer/``. Skips the
    VAE encoders. Needs no PyTorch and never holds a full-precision component
    in RAM or on disk. Already-converted components are skipped, so re-running
    against an existing folder just fills in what's missing.
    """
    from ..models.autoencoder_kl_ltx2 import LTX2VAEDecoderConfig, LTX2VideoDecoder
    from ..models.ltx2_audio import (
        LTX2AudioDecoder,
        LTX2AudioDecoderConfig,
        LTX2Vocoder,
        LTX2VocoderConfig,
    )
    from ..models.ltx2_connectors import LTX2ConnectorsConfig, LTX2TextConnectors
    from ..models.ltx2_transformer import LTX2Transformer3DModel, LTX2TransformerConfig

    out = Path(out_dir)
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    remote = RemoteSafetensors(url)
    all_keys = remote.keys()

    # Small components first so a mapping problem surfaces within seconds, not
    # after streaming 40 GB of transformer weights.
    components: list[tuple[str, ModelMixin, Callable[[str], str | None], int | None]] = [
        (
            "audio_decoder",
            LTX2AudioDecoder(LTX2AudioDecoderConfig.ltx_2_3()),
            map_audio_decoder_key,
            None,
        ),
        ("vocoder", LTX2Vocoder(LTX2VocoderConfig.ltx_2_3()), map_vocoder_key, None),
        ("vae_decoder", LTX2VideoDecoder(LTX2VAEDecoderConfig.ltx_2_3()), map_vae_key, None),
        (
            "connectors",
            LTX2TextConnectors(LTX2ConnectorsConfig.ltx_2_3_22b()),
            map_connectors_key,
            quantize_connectors,
        ),
        (
            "transformer",
            LTX2Transformer3DModel(LTX2TransformerConfig.ltx_2_3_22b()),
            map_transformer_key,
            quantize,
        ),
    ]
    for name, skeleton, key_map, bits in components:
        target = out / name
        if (target / "config.json").exists() and not _incomplete(target):
            logger.info("%s already converted, skipping", name)
            continue
        logger.info("converting %s -> %s", name, target)
        writer = _ComponentWriter(skeleton, target, quantize=bits)
        keys = [k for k in all_keys if key_map(k) is not None]
        total = sum(
            remote.header[k]["data_offsets"][1] - remote.header[k]["data_offsets"][0] for k in keys
        )
        done = 0
        for key, tensor in remote.iter_tensors(keys):
            mapped: str = key_map(key)  # type: ignore[assignment]  # keys pre-filtered
            if name in ("audio_decoder", "vocoder"):
                tensor = convert_audio_tensor(mapped, tensor)
            writer.add(mapped, tensor)
            done += remote.header[key]["data_offsets"][1] - remote.header[key]["data_offsets"][0]
            print(f"  {name}: {done / 1e9:.1f}/{total / 1e9:.1f} GB", end="\r", flush=True)
        print()
        writer.finish()

    _convert_gemma(out / "text_encoder", gemma_repo, quantize)
    _download_tokenizer(out / "tokenizer", gemma_repo)
    return out


def _incomplete(target: Path) -> bool:
    return (target / ".incomplete").exists()


def _convert_gemma(target: Path, repo: str, bits: int) -> None:
    """Convert the fp32 Gemma-3-12B shards one at a time (download -> quantize -> delete)."""
    from huggingface_hub import hf_hub_download

    from ..models.gemma3 import Gemma3Config, Gemma3TextEncoder

    if (target / "config.json").exists() and not _incomplete(target):
        logger.info("text_encoder already converted, skipping")
        return

    tmp = target.parent / "_shards_tmp"
    index_path = hf_hub_download(repo, "text_encoder/model.safetensors.index.json", local_dir=tmp)
    weight_map: dict[str, str] = json.loads(Path(index_path).read_text())["weight_map"]
    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if map_gemma_key(key) is not None:
            shards.setdefault(shard, []).append(key)

    writer = _ComponentWriter(Gemma3TextEncoder(Gemma3Config.gemma3_12b()), target, quantize=bits)
    for i, (shard, keys) in enumerate(sorted(shards.items()), 1):
        logger.info("text_encoder shard %d/%d: %s", i, len(shards), shard)
        # local_dir (not the HF cache) so each ~5 GB fp32 shard can be deleted
        # before the next one downloads — peak extra disk stays ~one shard.
        path = Path(hf_hub_download(repo, f"text_encoder/{shard}", local_dir=tmp))
        weights: dict[str, mx.array] = mx.load(str(path))  # type: ignore[assignment]
        for key in keys:
            writer.add(map_gemma_key(key), weights[key])  # type: ignore[arg-type]  # filtered above
        writer.flush()
        del weights
        mx.clear_cache()
        path.unlink()
    writer.finish()
    shutil.rmtree(tmp, ignore_errors=True)


def _download_tokenizer(target: Path, repo: str) -> None:
    from huggingface_hub import hf_hub_download

    target.mkdir(parents=True, exist_ok=True)
    files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    )
    for fname in files:
        path = hf_hub_download(repo, f"tokenizer/{fname}")
        shutil.copy(path, target / fname)
