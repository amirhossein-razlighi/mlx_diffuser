"""Tests for the core foundation: config round-trip, ModelMixin save/load."""

from __future__ import annotations

import dataclasses

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlx_diffuser
from mlx_diffuser.configuration import Config
from mlx_diffuser.modeling import ModelMixin
from mlx_diffuser.utils import as_dtype, seed_everything, to_array, to_pil


@dataclasses.dataclass
class _TinyConfig(Config):
    in_features: int = 4
    out_features: int = 8


class _TinyModel(ModelMixin):
    config_class = _TinyConfig

    def __init__(self, config: _TinyConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.in_features, config.out_features)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


def test_version_exposed():
    assert isinstance(mlx_diffuser.__version__, str)
    assert mlx_diffuser.__version__.count(".") >= 2


def test_config_roundtrip(tmp_path):
    cfg = _TinyConfig(in_features=3, out_features=5)
    cfg.save(tmp_path)
    loaded = _TinyConfig.load(tmp_path)
    assert loaded == cfg
    # Unknown keys are ignored for forward compatibility.
    data = cfg.to_dict()
    data["future_field"] = 123
    assert _TinyConfig.from_dict(data) == cfg
    assert data["_class_name"] == "_TinyConfig"


def test_config_replace():
    cfg = _TinyConfig()
    assert cfg.replace(out_features=16).out_features == 16
    assert cfg.out_features == 8  # original untouched


def test_model_save_load_roundtrip(tmp_path):
    model = _TinyModel(_TinyConfig(in_features=4, out_features=8))
    x = mx.random.normal((2, 4))
    y_before = model(x)

    model.save_pretrained(tmp_path)
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "model.safetensors").exists()

    reloaded = _TinyModel.from_pretrained(tmp_path)
    y_after = reloaded(x)
    assert mx.allclose(y_before, y_after, atol=1e-5)


def test_from_pretrained_dtype(tmp_path):
    model = _TinyModel(_TinyConfig())
    model.save_pretrained(tmp_path)
    reloaded = _TinyModel.from_pretrained(tmp_path, dtype="bf16")
    assert reloaded.proj.weight.dtype == mx.bfloat16


def test_from_pretrained_config_override(tmp_path):
    _TinyModel(_TinyConfig(out_features=8)).save_pretrained(tmp_path)
    reloaded = _TinyModel.from_pretrained(tmp_path, out_features=8)
    assert reloaded.config.out_features == 8


def test_num_parameters():
    model = _TinyModel(_TinyConfig(in_features=4, out_features=8))
    # Linear: weight (8x4) + bias (8) = 40
    assert model.num_parameters() == 40


def test_as_dtype():
    assert as_dtype("fp16") == mx.float16
    assert as_dtype(mx.float32) == mx.float32
    assert as_dtype(None) is None
    with pytest.raises(ValueError):
        as_dtype("not_a_dtype")


def test_seed_everything_reproducible():
    k1 = seed_everything(0)
    a = mx.random.normal((3,))
    seed_everything(0)
    b = mx.random.normal((3,))
    assert mx.allclose(a, b)
    assert k1.dtype == mx.uint32


def test_image_roundtrip():
    pytest.importorskip("PIL")
    img = mx.random.uniform(shape=(8, 8, 3)) * 2 - 1
    pil = to_pil(img)
    back = to_array(pil)
    # uint8 quantization tolerance.
    assert mx.max(mx.abs(back - img)).item() < 0.02
