# Installation

mlx-diffuser requires **Apple silicon** (M-series) and **Python 3.11+**.

```bash
pip install mlx-diffuser          # core
pip install "mlx-diffuser[hub]"   # + Hugging Face Hub loading
pip install "mlx-diffuser[trellis]" # + TRELLIS checkpoint conversion
```

## Extras

| Extra  | Adds                                        |
|--------|---------------------------------------------|
| `hub`  | `huggingface_hub` for loading/pushing repos |
| `trellis` | official TRELLIS checkpoint download/conversion |
| `dev`  | `pytest`, `ruff`, `mypy`                     |
| `docs` | `mkdocs-material`, `mkdocstrings`            |

## From source

```bash
git clone https://github.com/AmirHossein-razlighi/mlx_diffuser
cd mlx_diffuser
uv sync --extra dev
uv run pytest -q
```

For the native TRELLIS example, include both development and checkpoint extras:

```bash
uv sync --extra dev --extra trellis
```

The whole test suite runs in seconds on CPU/GPU with no model downloads.
