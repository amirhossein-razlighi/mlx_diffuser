# Contributing

Thanks for your interest in mlx-diffuser! This is a Mac-native library, so
development requires Apple silicon.

## Setup

```bash
git clone https://github.com/AmirHossein-razlighi/mlx_diffusion
cd mlx_diffusion
uv sync --extra dev
```

## Before opening a PR

Run the same checks CI runs:

```bash
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv run mypy
uv run pytest -q
```

The whole test suite runs in seconds — tests use tiny configs and download
nothing. New features should come with tests that follow that principle (scale
models down so they run on CPU/GPU in milliseconds).

## Conventions

- Channels-last tensors `(B, H, W, C)`.
- Configs are dataclasses; models are config-driven `nn.Module`s with
  `from_pretrained`/`save_pretrained`.
- Keep the process / network / pipeline split (see [DESIGN.md](DESIGN.md)).
- Prefer adding a config field over a new class; reach for abstraction only when it
  removes real duplication.

## Releasing (maintainers)

1. Bump `__version__` in `src/mlx_diffusion/version.py`.
2. Update `CHANGELOG.md`.
3. Tag and push: `git tag v0.1.0 && git push --tags`.
4. Publish a GitHub Release for the tag — the `Release` workflow builds and
   publishes to PyPI via trusted publishing (configure the PyPI publisher for this
   repo + `pypi` environment first).

## Docs

```bash
uv run --extra docs mkdocs serve
```
