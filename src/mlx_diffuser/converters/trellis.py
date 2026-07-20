"""Strict converters for the official Microsoft TRELLIS dense components."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlx.core as mx

from ..models.dinov2 import DINOv2Config, DINOv2Model
from ..models.trellis import (
    TrellisSparseStructureDecoder,
    TrellisSparseStructureDecoderConfig,
    TrellisSparseStructureFlowConfig,
    TrellisSparseStructureFlowModel,
)
from ..models.trellis_gaussian import (
    TrellisGaussianDecoder,
    TrellisGaussianDecoderConfig,
)
from ..models.trellis_slat import TrellisSLatFlowConfig, TrellisSLatFlowModel
from .base import Converter, convert_conv_weight, register_converter


def _component_args(config: dict, component: str) -> dict:
    """Read either a TRELLIS training config wrapper or a flat argument dict."""

    if "models" not in config:
        # Hub components are stored as {"name": ..., "args": {...}}, while tests
        # and direct callers may already provide the inner argument mapping.
        return config.get("args", config)
    try:
        return config["models"][component]["args"]
    except KeyError as exc:
        raise ValueError(f"TRELLIS config has no models.{component}.args section") from exc


@register_converter
class DINOv2WithRegistersConverter(Converter):
    source_class = "Dinov2WithRegistersModel"

    def build_config(self, hf_config: dict) -> DINOv2Config:
        return DINOv2Config(
            image_size=hf_config.get("image_size", 518),
            patch_size=hf_config.get("patch_size", 14),
            num_channels=hf_config.get("num_channels", 3),
            hidden_size=hf_config.get("hidden_size", 1024),
            num_hidden_layers=hf_config.get("num_hidden_layers", 24),
            num_attention_heads=hf_config.get("num_attention_heads", 16),
            mlp_ratio=hf_config.get("mlp_ratio", 4),
            num_register_tokens=hf_config.get("num_register_tokens", 4),
            layer_norm_eps=hf_config.get("layer_norm_eps", 1e-6),
            layerscale_value=hf_config.get("layerscale_value", 1.0),
            qkv_bias=hf_config.get("qkv_bias", True),
        )

    def build_model(self, hf_config: dict) -> DINOv2Model:
        return DINOv2Model(self.build_config(hf_config))

    def convert_weights(self, weights: dict[str, mx.array], hf_config: dict) -> dict[str, mx.array]:
        return {
            key: convert_conv_weight(value) if value.ndim == 4 else value
            for key, value in weights.items()
        }


@register_converter
class TrellisSparseStructureFlowConverter(Converter):
    source_class = "SparseStructureFlowModel"

    def build_config(self, hf_config: dict) -> TrellisSparseStructureFlowConfig:
        args = _component_args(hf_config, "denoiser")
        return TrellisSparseStructureFlowConfig(
            resolution=args.get("resolution", 16),
            in_channels=args.get("in_channels", 8),
            out_channels=args.get("out_channels", 8),
            model_channels=args.get("model_channels", 1024),
            cond_channels=args.get("cond_channels", 1024),
            num_blocks=args.get("num_blocks", 24),
            num_heads=args.get("num_heads", 16),
            mlp_ratio=args.get("mlp_ratio", 4.0),
            patch_size=args.get("patch_size", 1),
            pe_mode=args.get("pe_mode", "ape"),
            use_fp16=args.get("use_fp16", True),
            share_mod=args.get("share_mod", False),
            qk_rms_norm=args.get("qk_rms_norm", True),
            qk_rms_norm_cross=args.get("qk_rms_norm_cross", False),
        )

    def build_model(self, hf_config: dict) -> TrellisSparseStructureFlowModel:
        return TrellisSparseStructureFlowModel(self.build_config(hf_config))

    def convert_weights(self, weights: dict[str, mx.array], hf_config: dict) -> dict[str, mx.array]:
        # MLX and PyTorch Linear weights are both (out, in). Parameter names match.
        return dict(weights)


@register_converter
class TrellisSparseStructureDecoderConverter(Converter):
    source_class = "SparseStructureDecoder"

    def build_config(self, hf_config: dict) -> TrellisSparseStructureDecoderConfig:
        args = _component_args(hf_config, "decoder")
        return TrellisSparseStructureDecoderConfig(
            out_channels=args.get("out_channels", 1),
            latent_channels=args.get("latent_channels", 8),
            num_res_blocks=args.get("num_res_blocks", 2),
            channels=tuple(args.get("channels", (512, 128, 32))),
            num_res_blocks_middle=args.get("num_res_blocks_middle", 2),
            norm_type=args.get("norm_type", "layer"),
            use_fp16=args.get("use_fp16", True),
        )

    def build_model(self, hf_config: dict) -> TrellisSparseStructureDecoder:
        return TrellisSparseStructureDecoder(self.build_config(hf_config))

    def convert_weights(self, weights: dict[str, mx.array], hf_config: dict) -> dict[str, mx.array]:
        return {
            key: convert_conv_weight(value) if value.ndim == 5 else value
            for key, value in weights.items()
        }


@register_converter
class TrellisSLatFlowConverter(Converter):
    source_class = "ElasticSLatFlowModel"

    def build_config(self, hf_config: dict) -> TrellisSLatFlowConfig:
        args = _component_args(hf_config, "denoiser")
        return TrellisSLatFlowConfig(
            resolution=args.get("resolution", 64),
            in_channels=args.get("in_channels", 8),
            out_channels=args.get("out_channels", 8),
            model_channels=args.get("model_channels", 1024),
            cond_channels=args.get("cond_channels", 1024),
            num_blocks=args.get("num_blocks", 24),
            num_heads=args.get("num_heads", 16),
            mlp_ratio=args.get("mlp_ratio", 4.0),
            patch_size=args.get("patch_size", 2),
            num_io_res_blocks=args.get("num_io_res_blocks", 2),
            io_block_channels=tuple(args.get("io_block_channels", (128,))),
            pe_mode=args.get("pe_mode", "ape"),
            use_fp16=args.get("use_fp16", True),
            use_skip_connection=args.get("use_skip_connection", True),
            share_mod=args.get("share_mod", False),
            qk_rms_norm=args.get("qk_rms_norm", True),
            qk_rms_norm_cross=args.get("qk_rms_norm_cross", False),
        )

    def build_model(self, hf_config: dict) -> TrellisSLatFlowModel:
        return TrellisSLatFlowModel(self.build_config(hf_config))

    def convert_weights(self, weights: dict[str, mx.array], hf_config: dict) -> dict[str, mx.array]:
        # Official spconv tensors already use (O, kD, kH, kW, I), which is the
        # persistent layout retained by our custom Metal sparse-conv modules.
        return dict(weights)


@register_converter
class TrellisGaussianDecoderConverter(Converter):
    source_class = "ElasticSLatGaussianDecoder"

    def build_config(self, hf_config: dict) -> TrellisGaussianDecoderConfig:
        args = _component_args(hf_config, "decoder")
        rep = args.get("representation_config", {})
        learning_rates = rep.get("lr", {})
        return TrellisGaussianDecoderConfig(
            resolution=args.get("resolution", 64),
            model_channels=args.get("model_channels", 768),
            latent_channels=args.get("latent_channels", 8),
            num_blocks=args.get("num_blocks", 12),
            num_heads=args.get("num_heads", 12),
            mlp_ratio=args.get("mlp_ratio", 4.0),
            attn_mode=args.get("attn_mode", "swin"),
            window_size=args.get("window_size", 8),
            pe_mode=args.get("pe_mode", "ape"),
            use_fp16=args.get("use_fp16", True),
            qk_rms_norm=args.get("qk_rms_norm", False),
            num_gaussians=rep.get("num_gaussians", 32),
            perturb_offset=rep.get("perturb_offset", True),
            voxel_size=rep.get("voxel_size", 1.5),
            filter_kernel_size_3d=rep.get("3d_filter_kernel_size", 9e-4),
            scaling_bias=rep.get("scaling_bias", 4e-3),
            opacity_bias=rep.get("opacity_bias", 0.1),
            scaling_activation=rep.get("scaling_activation", "softplus"),
            xyz_lr=learning_rates.get("_xyz", 1.0),
            features_lr=learning_rates.get("_features_dc", 1.0),
            opacity_lr=learning_rates.get("_opacity", 1.0),
            scaling_lr=learning_rates.get("_scaling", 1.0),
            rotation_lr=learning_rates.get("_rotation", 0.1),
        )

    def build_model(self, hf_config: dict) -> TrellisGaussianDecoder:
        return TrellisGaussianDecoder(self.build_config(hf_config))

    def convert_weights(self, weights: dict[str, mx.array], hf_config: dict) -> dict[str, mx.array]:
        return dict(weights)


def convert_trellis_dense_components(
    source: str | Path,
    output: str | Path,
    *,
    quantize_flow: int | None = 8,
) -> Path:
    """Convert TRELLIS's occupancy flow and decoder into an MLX checkpoint folder.

    ``source`` may be the root of an official Hub snapshot or its ``ckpts`` folder.
    The default 8-bit flow weights save memory on 16 GB machines while the much
    smaller Conv3D decoder remains mixed FP16/FP32 for fidelity.
    """

    source = Path(source)
    ckpts = source / "ckpts" if (source / "ckpts").is_dir() else source
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    flow_source = ckpts / "ss_flow_img_dit_L_16l8_fp16.safetensors"
    decoder_source = ckpts / "ss_dec_conv3d_16l8_fp16.safetensors"
    for path in (flow_source, decoder_source):
        if not path.exists():
            raise FileNotFoundError(f"missing official TRELLIS component: {path}")

    flow = TrellisSparseStructureFlowConverter().convert(flow_source, quantize=quantize_flow)
    flow.save_pretrained(output / "sparse_structure_flow")
    if quantize_flow is not None:
        (output / "sparse_structure_flow" / "quantization.json").write_text(
            json.dumps({"bits": quantize_flow, "group_size": 64})
        )
    del flow
    mx.clear_cache()

    decoder = TrellisSparseStructureDecoderConverter().convert(decoder_source)
    decoder.save_pretrained(output / "sparse_structure_decoder")
    del decoder
    mx.clear_cache()

    manifest = {
        "_class_name": "TrellisImageTo3DPipeline",
        "format_version": 1,
        "components": {
            "sparse_structure_flow": "TrellisSparseStructureFlowModel",
            "sparse_structure_decoder": "TrellisSparseStructureDecoder",
        },
        "source": "microsoft/TRELLIS-image-large",
        "status": "dense-stage",
    }
    (output / "trellis.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return output


def convert_trellis_checkpoint(
    source: str | Path,
    dino_source: str | Path,
    output: str | Path,
    *,
    quantize_dino: int | None = 8,
    quantize_sparse_structure_flow: int | None = 8,
) -> Path:
    """Convert the complete image-to-Gaussian TRELLIS pipeline for staged MLX use."""

    source = Path(source)
    ckpts = source / "ckpts" if (source / "ckpts").is_dir() else source
    output = convert_trellis_dense_components(
        ckpts,
        output,
        quantize_flow=quantize_sparse_structure_flow,
    )

    dino = DINOv2WithRegistersConverter().convert(
        dino_source,
        dtype=mx.float16,
        quantize=quantize_dino,
    )
    dino.save_pretrained(output / "image_conditioner")
    if quantize_dino is not None:
        (output / "image_conditioner" / "quantization.json").write_text(
            json.dumps({"bits": quantize_dino, "group_size": 64})
        )
    del dino
    mx.clear_cache()

    slat_path = ckpts / "slat_flow_img_dit_L_64l8p2_fp16.safetensors"
    gaussian_path = ckpts / "slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"
    for path in (slat_path, gaussian_path):
        if not path.exists():
            raise FileNotFoundError(f"missing official TRELLIS component: {path}")

    slat = TrellisSLatFlowConverter().convert(slat_path)
    slat.save_pretrained(output / "slat_flow")
    del slat
    mx.clear_cache()

    gaussian = TrellisGaussianDecoderConverter().convert(gaussian_path)
    gaussian.save_pretrained(output / "gaussian_decoder")
    del gaussian
    mx.clear_cache()

    manifest = {
        "_class_name": "TrellisImageTo3DPipeline",
        "format_version": 1,
        "components": {
            "image_conditioner": "DINOv2Model",
            "sparse_structure_flow": "TrellisSparseStructureFlowModel",
            "sparse_structure_decoder": "TrellisSparseStructureDecoder",
            "slat_flow": "TrellisSLatFlowModel",
            "gaussian_decoder": "TrellisGaussianDecoder",
        },
        "source": "microsoft/TRELLIS-image-large",
        "conditioner_source": "facebook/dinov2-with-registers-large",
        "status": "image-to-gaussian",
    }
    (output / "trellis.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return output


def download_and_convert_trellis(
    output: str | Path,
    *,
    quantize_dino: int | None = 8,
    quantize_sparse_structure_flow: int | None = 8,
) -> Path:
    """Download official safetensors and build a staged MLX TRELLIS checkpoint."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("TRELLIS download requires `uv sync --extra trellis`") from exc

    with tempfile.TemporaryDirectory(prefix="mlx-diffuser-trellis-") as temporary_dir:
        temporary = Path(temporary_dir)
        trellis_source = temporary / "trellis"
        dino_source = temporary / "dinov2"
        snapshot_download(
            "microsoft/TRELLIS-image-large",
            local_dir=trellis_source,
            allow_patterns=[
                "ckpts/ss_flow_img_dit_L_16l8_fp16.json",
                "ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors",
                "ckpts/ss_dec_conv3d_16l8_fp16.json",
                "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
                "ckpts/slat_flow_img_dit_L_64l8p2_fp16.json",
                "ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors",
                "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.json",
                "ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors",
            ],
        )
        snapshot_download(
            "facebook/dinov2-with-registers-large",
            local_dir=dino_source,
            allow_patterns=["config.json", "model.safetensors"],
        )
        return convert_trellis_checkpoint(
            trellis_source,
            dino_source,
            output,
            quantize_dino=quantize_dino,
            quantize_sparse_structure_flow=quantize_sparse_structure_flow,
        )


__all__ = [
    "TrellisSparseStructureDecoderConverter",
    "TrellisSparseStructureFlowConverter",
    "TrellisSLatFlowConverter",
    "TrellisGaussianDecoderConverter",
    "convert_trellis_checkpoint",
    "download_and_convert_trellis",
    "convert_trellis_dense_components",
]
