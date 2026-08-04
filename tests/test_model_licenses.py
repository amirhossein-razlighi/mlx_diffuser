"""Tests for preserving upstream model licenses beside converted checkpoints."""

from __future__ import annotations

import pytest

from mlx_diffuser.cli import _download, _ModelSpec
from mlx_diffuser.licenses import install_model_license_files, model_license_summary


def test_install_ltx_license_files(tmp_path):
    install_model_license_files("Lightricks/LTX-2.3", tmp_path)

    assert (tmp_path / "LTX-2-COMMUNITY.txt").read_text().lstrip().startswith(
        "LTX-2 Community License Agreement"
    )
    notice = (tmp_path / "MODEL_LICENSES.md").read_text()
    assert "Lightricks/LTX-2.3" in notice
    assert "Conversion to MLX" in notice
    assert "CC0-1.0" in notice


def test_install_trellis_and_conditioner_licenses(tmp_path):
    install_model_license_files("microsoft/TRELLIS-image-large", tmp_path)

    assert (tmp_path / "TRELLIS-MIT.txt").exists()
    assert (tmp_path / "APACHE-2.0.txt").exists()


def test_unknown_model_license_errors(tmp_path):
    with pytest.raises(ValueError, match="no model-license metadata"):
        install_model_license_files("example/unknown", tmp_path)


def test_restricted_model_summaries_are_explicit():
    assert "Non-Commercial" in model_license_summary("black-forest-labs/FLUX.1-dev")
    assert "Community License" in model_license_summary("Lightricks/LTX-2.3")


def test_custom_download_installs_license_files(tmp_path):
    def download(folder):
        folder.mkdir(parents=True, exist_ok=True)

    spec = _ModelSpec(
        modality="video",
        repo="Lightricks/LTX-2.3",
        local=str(tmp_path),
        patterns=(),
        run=lambda *_: None,
        download=download,
    )

    _download(spec, tmp_path)

    assert (tmp_path / "LTX-2-COMMUNITY.txt").exists()
    assert (tmp_path / "MODEL_LICENSES.md").exists()
