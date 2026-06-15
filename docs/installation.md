# Installation

mlx-diffuser requires **Apple silicon** (M-series) and **Python 3.11+**.

```bash
pip install mlx-diffuser          # core
pip install "mlx-diffuser[hub]"   # + Hugging Face Hub loading
```

## Extras

| Extra  | Adds                                        |
|--------|---------------------------------------------|
| `hub`  | `huggingface_hub` for loading/pushing repos |
| `dev`  | `pytest`, `ruff`, `mypy`                     |
| `docs` | `mkdocs-material`, `mkdocstrings`            |

## From source

```bash
git clone https://github.com/AmirHossein-razlighi/mlx_diffuser
cd mlx_diffuser
uv sync --extra dev
uv run pytest -q
```

The whole test suite runs in seconds on CPU/GPU with no model downloads.
