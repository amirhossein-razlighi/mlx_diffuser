"""License metadata for supported third-party model checkpoints."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

_MODEL_LICENSES: dict[str, tuple[str, tuple[str, ...]]] = {
    "stabilityai/stable-diffusion-xl-base-1.0": (
        "CreativeML Open RAIL++-M",
        ("SDXL-OPENRAIL++.txt",),
    ),
    "black-forest-labs/FLUX.1-schnell": (
        "Apache License 2.0",
        ("APACHE-2.0.txt",),
    ),
    "black-forest-labs/FLUX.1-dev": (
        "FLUX.1 [dev] Non-Commercial License",
        ("FLUX.1-DEV.txt",),
    ),
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": (
        "Apache License 2.0 and the upstream model-card usage terms",
        ("APACHE-2.0.txt",),
    ),
    "Lightricks/LTX-2.3": (
        "LTX-2 Community License Agreement, including use restrictions and commercial terms",
        ("LTX-2-COMMUNITY.txt",),
    ),
    "microsoft/TRELLIS-image-large": (
        "MIT License for TRELLIS and Apache License 2.0 for the DINOv2 conditioner",
        ("TRELLIS-MIT.txt", "APACHE-2.0.txt"),
    ),
}


def model_license_summary(repo: str) -> str:
    """Return the human-readable upstream license for a supported model repo."""

    try:
        return _MODEL_LICENSES[repo][0]
    except KeyError as exc:
        raise ValueError(f"no model-license metadata registered for {repo!r}") from exc


def install_model_license_files(repo: str, output: str | Path) -> None:
    """Copy applicable licenses and a provenance notice into a checkpoint folder.

    Quantization, conversion, and tensor-layout changes do not replace the license
    attached to the upstream weights.  Keeping the notice beside the converted
    checkpoint makes that boundary explicit if the folder is later redistributed.
    """

    try:
        summary, license_files = _MODEL_LICENSES[repo]
    except KeyError as exc:
        raise ValueError(f"no model-license metadata registered for {repo!r}") from exc

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    resources = files(__package__)
    for filename in license_files:
        (output / filename).write_bytes(resources.joinpath(filename).read_bytes())

    filenames = ", ".join(f"`{name}`" for name in license_files)
    notice = f"""# Model license notice

This checkpoint was downloaded and/or converted from `{repo}`.

Applicable upstream terms: **{summary}**.

Conversion to MLX, quantization, or a change in tensor layout does not replace or
relax the upstream license. Review the copied license file(s) before using,
modifying, or redistributing this checkpoint: {filenames}.

The CC0-1.0 license for original `mlx-diffuser` code does not apply to these model
weights or to derivatives of them.
"""
    (output / "MODEL_LICENSES.md").write_text(notice)


__all__ = ["install_model_license_files", "model_license_summary"]
