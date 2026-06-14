"""CLI tests: drive `main(argv)` for generate / train / convert on tiny models."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_diffusion.cli import build_parser, main
from mlx_diffusion.models import DiT, DiTConfig
from mlx_diffusion.pipelines import ClassConditionalPipeline
from mlx_diffusion.schedulers import FlowMatchEulerScheduler

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _tiny_pipeline_dir(tmp_path):
    cfg = DiTConfig(
        in_channels=3, patch_size=2, hidden_size=16, depth=2, num_heads=2, num_classes=8
    )
    pipe = ClassConditionalPipeline(DiT(cfg), FlowMatchEulerScheduler())
    d = tmp_path / "pipe"
    pipe.save_pretrained(d)
    return d


def _make_image_folder(tmp_path, n=4, size=8):
    folder = tmp_path / "imgs"
    folder.mkdir()
    for i in range(n):
        arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(folder / f"{i}.png")
    return folder


def test_parser_builds():
    parser = build_parser()
    args = parser.parse_args(["generate", "model", "--labels", "1,2"])
    assert args.command == "generate"
    assert args.labels == "1,2"


def test_generate_writes_images(tmp_path):
    model_dir = _tiny_pipeline_dir(tmp_path)
    out = tmp_path / "out"
    main(
        [
            "generate",
            str(model_dir),
            "--labels",
            "0,3",
            "--steps",
            "3",
            "--size",
            "8",
            "--out",
            str(out),
        ]
    )
    assert (out / "sample_000.png").exists()
    assert (out / "sample_001.png").exists()


def test_train_from_scratch(tmp_path):
    folder = _make_image_folder(tmp_path)
    out = tmp_path / "model"
    main(
        [
            "train",
            "--data",
            str(folder),
            "--out",
            str(out),
            "--steps",
            "3",
            "--batch",
            "2",
            "--size",
            "8",
            "--hidden",
            "16",
            "--depth",
            "2",
        ]
    )
    assert (out / "config.json").exists()
    assert (out / "model.safetensors").exists()


def test_train_lora(tmp_path):
    folder = _make_image_folder(tmp_path)
    base = tmp_path / "base"
    DiT(DiTConfig(in_channels=3, hidden_size=16, depth=2, num_heads=2)).save_pretrained(base)
    out = tmp_path / "adapter"
    main(
        [
            "train",
            "--data",
            str(folder),
            "--out",
            str(out),
            "--base",
            str(base),
            "--steps",
            "3",
            "--batch",
            "2",
            "--size",
            "8",
            "--lora",
            "--lora-rank",
            "4",
        ]
    )
    assert (out / "adapter_config.json").exists()
    assert (out / "adapter_model.safetensors").exists()


def test_convert_dtype(tmp_path):
    src = tmp_path / "src"
    DiT(DiTConfig(in_channels=3, hidden_size=16, depth=2, num_heads=2)).save_pretrained(src)
    dst = tmp_path / "dst"
    main(["convert", str(src), str(dst), "--dtype", "bf16"])
    reloaded = DiT.from_pretrained(dst)
    # A representative weight is now bf16.
    assert reloaded.blocks[0].attn.q_proj.weight.dtype == mx.bfloat16
