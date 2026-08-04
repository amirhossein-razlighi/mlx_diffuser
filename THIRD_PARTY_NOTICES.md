# Third-party notices

`mlx-diffuser` is an independent MLX implementation that interoperates with model
architectures and checkpoints published by third parties. The repository does not
bundle upstream model weights. Its download and conversion commands fetch those
weights from their official repositories at the user's request.

The project's CC0-1.0 dedication applies only to original material authored for this
repository. It does not supersede licenses governing upstream code, model weights,
converted checkpoints, model derivatives, tokenizers, or other third-party assets.

## Model-specific implementations

The model-specific modules use published architectures and checkpoint layouts to
provide weight-compatible MLX implementations. To the extent a model-specific file
or implementation is a derivative of upstream material under applicable law, the
corresponding upstream terms continue to apply to that portion.

| Implementation | Upstream project | Terms and bundled copy |
| --- | --- | --- |
| SDXL | [Stability AI SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and Hugging Face Diffusers/Transformers | [CreativeML Open RAIL++-M](src/mlx_diffuser/licenses/SDXL-OPENRAIL++.txt); Apache-2.0 for applicable Hugging Face code |
| FLUX.1-schnell | [Black Forest Labs FLUX](https://github.com/black-forest-labs/flux) and Hugging Face Diffusers/Transformers | [Apache-2.0](src/mlx_diffuser/licenses/APACHE-2.0.txt) |
| FLUX.1-dev | [Black Forest Labs FLUX](https://github.com/black-forest-labs/flux) | [FLUX.1 Dev Non-Commercial License](src/mlx_diffuser/licenses/FLUX.1-DEV.txt) for the model and derivatives governed by that agreement |
| WAN 2.1 | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) and Hugging Face Diffusers/Transformers | [Apache-2.0](src/mlx_diffuser/licenses/APACHE-2.0.txt), plus the model-card usage terms |
| LTX-2.3 | [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) | [LTX-2 Community License Agreement](src/mlx_diffuser/licenses/LTX-2-COMMUNITY.txt) |
| TRELLIS image-large | [Microsoft TRELLIS](https://github.com/microsoft/TRELLIS) | [MIT](src/mlx_diffuser/licenses/TRELLIS-MIT.txt) |
| DINOv2 with registers | [Meta DINOv2](https://github.com/facebookresearch/dinov2) | [Apache-2.0](src/mlx_diffuser/licenses/APACHE-2.0.txt) |

The LTX-2 agreement expressly addresses derivative model architectures and requires
its agreement and applicable notices to accompany redistribution of governed
derivatives. The FLUX.1-dev agreement likewise requires its license and attribution
notice when distributing the model or governed derivatives. The bundled copies are
included so they remain available in source and wheel distributions and are copied
into checkpoints created through `mlx-diffuser generate --download`.

## Framework and reference implementations

- [MLX](https://github.com/ml-explore/mlx) is licensed under the MIT License.
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) and
  [Transformers](https://github.com/huggingface/transformers) are licensed under
  Apache-2.0. Model layouts, naming, and numerical reference checks in this project
  may correspond to those implementations.

Copyright and attribution notices from upstream projects remain the property of their
respective owners. No project name or trademark is used to suggest sponsorship or
endorsement.

## Generated and converted artifacts

Converted checkpoints are transformations of upstream weights, not newly CC0-licensed
weights. Anyone sharing a converted checkpoint must carry forward the applicable
upstream license, notices, use restrictions, and attribution requirements. Outputs may
also be subject to license-specific conditions and applicable law; consult the relevant
model license rather than relying on the `mlx-diffuser` code license.

This notice is informational and is not legal advice. If an intended use or
redistribution is commercially significant, obtain advice from qualified counsel.
