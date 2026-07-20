"""Staged, low-unified-memory TRELLIS image-to-3D inference pipeline."""

from __future__ import annotations

import dataclasses
import gc
import json
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np

from ..layers.sparse import SparseTensor
from ..models.dinov2 import DINOv2Model
from ..models.trellis import (
    TrellisSparseStructureDecoder,
    TrellisSparseStructureFlowModel,
)
from ..models.trellis_gaussian import GaussianSplat3D, TrellisGaussianDecoder
from ..models.trellis_slat import TrellisSLatFlowModel
from ..schedulers.trellis_flow import TrellisFlowEulerSampler

SLAT_MEAN = (
    -2.1687545776367188,
    -0.004347046371549368,
    -0.13352349400520325,
    -0.08418072760105133,
    -0.5271206498146057,
    0.7238689064979553,
    -1.1414450407028198,
    1.2039363384246826,
)
SLAT_STD = (
    2.377650737762451,
    2.386378288269043,
    2.124418020248413,
    2.1748552322387695,
    2.663944721221924,
    2.371192216873169,
    2.6217446327209473,
    2.684523105621338,
)


@dataclasses.dataclass(frozen=True)
class TrellisPipelineOutput:
    coordinates: mx.array
    slat: SparseTensor
    gaussians: tuple[GaussianSplat3D, ...]

    def save_ply(self, path: str | Path, sample: int = 0) -> Path:
        return self.gaussians[sample].save_ply(path)


class TrellisImageTo3DPipeline:
    """Native MLX image-to-Gaussian TRELLIS pipeline.

    ``from_pretrained`` stores component paths and loads only one stage at a time.
    This avoids keeping DINOv2, both large flow transformers, and the decoder
    resident together, reducing peak residency for unified-memory Macs.
    """

    _classes: dict[str, Any] = {
        "image_conditioner": DINOv2Model,
        "sparse_structure_flow": TrellisSparseStructureFlowModel,
        "sparse_structure_decoder": TrellisSparseStructureDecoder,
        "slat_flow": TrellisSLatFlowModel,
        "gaussian_decoder": TrellisGaussianDecoder,
    }

    def __init__(
        self,
        *,
        component_paths: dict[str, Path] | None = None,
        components: dict[str, Any] | None = None,
    ):
        self.component_paths = component_paths or {}
        self._components = components or {}
        self.sampler = TrellisFlowEulerSampler(sigma_min=1e-5)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> TrellisImageTo3DPipeline:
        path = Path(path)
        manifest_path = path / "trellis.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing converted TRELLIS manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        required = set(cls._classes)
        missing = required - set(manifest.get("components", {}))
        if missing:
            raise ValueError(f"TRELLIS checkpoint is missing components: {sorted(missing)}")
        return cls(component_paths={name: path / name for name in required})

    def _load(self, name: str):
        if name not in self._components:
            path = self.component_paths.get(name)
            if path is None:
                raise ValueError(f"component {name!r} was not provided")
            self._components[name] = self._classes[name].from_pretrained(path)
        return self._components[name]

    def _release(self, name: str) -> None:
        if name in self.component_paths:
            self._components.pop(name, None)
            gc.collect()
            mx.clear_cache()

    @staticmethod
    def preprocess_image(
        image: str | Path | Any,
        *,
        remove_background: bool = True,
    ) -> mx.array:
        """Crop an RGBA object, composite on black, and resize to DINOv2's 518 px."""

        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a path or PIL.Image.Image")

        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba, dtype=np.uint8)[..., 3]
        has_alpha = bool(np.any(alpha != 255))
        if not has_alpha and remove_background:
            try:
                import rembg
            except ImportError as exc:
                raise ImportError(
                    "background removal requires `uv add 'rembg[cpu]'`, or pass a transparent PNG"
                ) from exc
            rgba = rembg.remove(image.convert("RGB")).convert("RGBA")
            alpha = np.asarray(rgba, dtype=np.uint8)[..., 3]
            has_alpha = True

        if has_alpha:
            foreground = np.argwhere(alpha > int(0.8 * 255))
            if not len(foreground):
                raise ValueError("input image alpha channel contains no visible foreground")
            top, left = foreground.min(axis=0)
            bottom, right = foreground.max(axis=0)
            center_x, center_y = (left + right) / 2, (top + bottom) / 2
            size = max(right - left, bottom - top) * 1.2
            box = (
                int(center_x - size / 2),
                int(center_y - size / 2),
                int(center_x + size / 2),
                int(center_y + size / 2),
            )
            rgba = rgba.crop(box)
        rgba = rgba.resize((518, 518), Image.Resampling.LANCZOS)
        pixels = np.asarray(rgba, dtype=np.float32) / 255.0
        rgb = pixels[..., :3] * pixels[..., 3:4]
        return mx.array(rgb[None])

    def encode_image(self, image: mx.array, *, low_memory: bool = True) -> mx.array:
        if image.ndim == 3:
            image = image[None]
        if image.shape[1:] != (518, 518, 3):
            raise ValueError(
                f"preprocessed image must have shape (B, 518, 518, 3), got {image.shape}"
            )
        mean = mx.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = mx.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        model = self._load("image_conditioner")
        conditioning = model.trellis_conditioning((image - mean) / std, low_memory=low_memory)
        mx.eval(conditioning)
        self._release("image_conditioner")
        return conditioning

    def sample_sparse_structure(
        self,
        conditioning: mx.array,
        *,
        key: mx.array,
        num_samples: int = 1,
        steps: int = 25,
        low_memory: bool = True,
    ) -> mx.array:
        flow = self._load("sparse_structure_flow")
        if conditioning.shape[0] == 1 and num_samples > 1:
            conditioning = mx.broadcast_to(conditioning, (num_samples, *conditioning.shape[1:]))
        noise = mx.random.normal(
            (
                num_samples,
                flow.config.resolution,
                flow.config.resolution,
                flow.config.resolution,
                flow.config.in_channels,
            ),
            key=key,
        )
        latent = self.sampler.sample(
            flow,
            noise,
            conditioning,
            negative_cond=mx.zeros_like(conditioning),
            steps=steps,
            rescale_t=3.0,
            cfg_strength=5.0,
            cfg_interval=(0.5, 1.0),
        ).samples
        mx.eval(latent)
        del flow
        self._release("sparse_structure_flow")

        decoder = self._load("sparse_structure_decoder")
        coords = decoder.occupied_coordinates(latent)
        mx.eval(coords)
        self._release("sparse_structure_decoder")
        if coords.shape[0] == 0:
            raise RuntimeError("TRELLIS occupancy decoder produced no occupied voxels")
        return coords

    def sample_slat(
        self,
        conditioning: mx.array,
        coords: mx.array,
        *,
        key: mx.array,
        steps: int = 25,
        low_memory: bool = True,
    ) -> SparseTensor:
        flow = self._load("slat_flow")
        batch_size = int(mx.max(coords[:, 0]).item()) + 1
        if conditioning.shape[0] == 1 and batch_size > 1:
            conditioning = mx.broadcast_to(conditioning, (batch_size, *conditioning.shape[1:]))
        noise = SparseTensor(
            mx.random.normal((coords.shape[0], flow.config.in_channels), key=key),
            coords,
            batch_size=batch_size,
            spatial_shape=(flow.config.resolution,) * 3,
        )
        result = self.sampler.sample(
            lambda x, t, cond: flow(x, t, cond, low_memory=low_memory),
            noise,
            conditioning,
            negative_cond=mx.zeros_like(conditioning),
            steps=steps,
            rescale_t=3.0,
            cfg_strength=5.0,
            cfg_interval=(0.5, 1.0),
        ).samples
        mean = mx.array(SLAT_MEAN, dtype=result.dtype).reshape(1, -1)
        std = mx.array(SLAT_STD, dtype=result.dtype).reshape(1, -1)
        result = result * std + mean
        mx.eval(result.features)
        self._release("slat_flow")
        return result

    def decode_gaussians(
        self, slat: SparseTensor, *, low_memory: bool = True
    ) -> tuple[GaussianSplat3D, ...]:
        decoder = self._load("gaussian_decoder")
        gaussians = tuple(decoder(slat, low_memory=low_memory))
        for gaussian in gaussians:
            mx.eval(
                gaussian.xyz_normalized,
                gaussian.features_dc,
                gaussian.scaling_raw,
                gaussian.rotation_raw,
                gaussian.opacity_raw,
            )
        self._release("gaussian_decoder")
        return gaussians

    def __call__(
        self,
        image: str | Path | Any,
        *,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_steps: int = 25,
        slat_steps: int = 25,
        preprocess: bool = True,
        remove_background: bool = True,
        low_memory: bool = True,
    ) -> TrellisPipelineOutput:
        pixels = (
            self.preprocess_image(image, remove_background=remove_background)
            if preprocess
            else cast(mx.array, image)
        )
        conditioning = self.encode_image(pixels, low_memory=low_memory)
        key, structure_key, slat_key = mx.random.split(mx.random.key(seed), 3)
        del key
        coords = self.sample_sparse_structure(
            conditioning,
            key=structure_key,
            num_samples=num_samples,
            steps=sparse_structure_steps,
            low_memory=low_memory,
        )
        slat = self.sample_slat(
            conditioning,
            coords,
            key=slat_key,
            steps=slat_steps,
            low_memory=low_memory,
        )
        del conditioning
        gaussians = self.decode_gaussians(slat, low_memory=low_memory)
        return TrellisPipelineOutput(coords, slat, gaussians)


__all__ = [
    "SLAT_MEAN",
    "SLAT_STD",
    "TrellisImageTo3DPipeline",
    "TrellisPipelineOutput",
]
