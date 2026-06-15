# Installation

mlx-diffusion requires **Apple silicon** (M-series) and **Python 3.11+**.

```bash
pip install mlx-diffusion          # core
pip install "mlx-diffusion[hub]"   # + Hugging Face Hub loading
```

## Extras

| Extra  | Adds                                        |
|--------|---------------------------------------------|
| `hub`  | `huggingface_hub` for loading/pushing repos |
| `dev`  | `pytest`, `ruff`, `mypy`                     |
| `docs` | `mkdocs-material`, `mkdocstrings`            |

## From source

```bash
git clone https://github.com/AmirHossein-razlighi/mlx_diffusion
cd mlx_diffusion
uv sync --extra dev
uv run pytest -q
```

The whole test suite runs in seconds on CPU/GPU with no model downloads.
