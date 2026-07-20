"""Native TRELLIS architecture, sparse Metal kernel, and pipeline tests."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_diffuser.converters.trellis import (
    DINOv2WithRegistersConverter,
    TrellisGaussianDecoderConverter,
    TrellisSLatFlowConverter,
    TrellisSparseStructureDecoderConverter,
    TrellisSparseStructureFlowConverter,
)
from mlx_diffuser.layers.sparse import (
    SparseConv3D,
    SparseTensor,
    sparse_downsample,
    sparse_subdivide,
    sparse_upsample,
)
from mlx_diffuser.models.dinov2 import DINOv2Config, DINOv2Model
from mlx_diffuser.models.trellis import (
    TrellisSparseStructureDecoder,
    TrellisSparseStructureDecoderConfig,
    TrellisSparseStructureFlowConfig,
    TrellisSparseStructureFlowModel,
)
from mlx_diffuser.models.trellis_gaussian import (
    TrellisGaussianDecoder,
    TrellisGaussianDecoderConfig,
)
from mlx_diffuser.models.trellis_slat import TrellisSLatFlowConfig, TrellisSLatFlowModel
from mlx_diffuser.pipelines.trellis import TrellisImageTo3DPipeline
from mlx_diffuser.schedulers.trellis_flow import TrellisFlowEulerSampler


def _coords(size: int = 4) -> mx.array:
    return mx.array(
        [[0, z, y, x] for z in range(size) for y in range(size) for x in range(size)],
        dtype=mx.int32,
    )


def _tiny_structure_flow() -> TrellisSparseStructureFlowModel:
    return TrellisSparseStructureFlowModel(
        TrellisSparseStructureFlowConfig(
            resolution=4,
            in_channels=2,
            out_channels=2,
            model_channels=24,
            cond_channels=12,
            num_blocks=2,
            num_heads=3,
            mlp_ratio=2,
            patch_size=2,
            use_fp16=False,
        )
    )


def test_sparse_structure_flow_shape_and_zero_init():
    model = _tiny_structure_flow()
    x = mx.random.normal((1, 4, 4, 4, 2))
    output = model(x, mx.array([500.0]), mx.random.normal((1, 5, 12)))
    assert output.shape == x.shape
    assert mx.max(mx.abs(output)).item() < 1e-6


def test_sparse_structure_flow_patch_order_roundtrip():
    model = _tiny_structure_flow()
    x = mx.arange(1 * 4 * 4 * 4 * 2).reshape(1, 4, 4, 4, 2)
    patches = model._patchify(x)
    restored = model._unpatchify(patches)
    assert mx.array_equal(restored, x)


def test_sparse_structure_decoder_and_coordinate_extraction():
    decoder = TrellisSparseStructureDecoder(
        TrellisSparseStructureDecoderConfig(
            latent_channels=2,
            channels=(8, 4),
            num_res_blocks=1,
            num_res_blocks_middle=1,
            use_fp16=False,
        )
    )
    decoder.out_layer[2].weight = mx.zeros_like(decoder.out_layer[2].weight)
    decoder.out_layer[2].bias = mx.ones_like(decoder.out_layer[2].bias)
    latent = mx.zeros((1, 4, 4, 4, 2))
    logits = decoder(latent)
    coords = decoder.occupied_coordinates(latent)
    assert logits.shape == (1, 8, 8, 8, 1)
    assert coords.shape == (8**3, 4)
    assert coords.dtype == mx.int32


def test_trellis_flow_sampler_reference_cfg_formula():
    sampler = TrellisFlowEulerSampler()

    def model(x, t, cond):
        del t
        return mx.ones_like(x) * cond[:, 0, 0:1]

    result = sampler.sample(
        model,
        mx.ones((1, 2)),
        mx.ones((1, 1, 1)),
        negative_cond=mx.zeros((1, 1, 1)),
        steps=1,
        rescale_t=1.0,
        cfg_strength=2.0,
        cfg_interval=(0.5, 1.0),
    )
    # Official TRELLIS CFG is (1+s)*positive - s*negative, so velocity is 3.
    assert mx.allclose(result.samples, mx.full((1, 2), -2.0))


def test_sparse_metal_conv_matches_pure_mlx():
    coords = mx.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
        dtype=mx.int32,
    )
    tensor = SparseTensor(mx.arange(12, dtype=mx.float32).reshape(4, 3), coords)
    conv = SparseConv3D(3, 2, 3, use_metal=False)
    reference = conv(tensor).features
    conv.use_metal = True
    conv.eval()
    actual = conv(tensor).features
    mx.eval(reference, actual)
    assert mx.allclose(actual, reference, atol=1e-5)


def test_sparse_metal_conv_uses_autodiff_fallback_while_training():
    coords = mx.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=mx.int32)
    conv = SparseConv3D(2, 2, 3, use_metal=True)

    def loss(features):
        tensor = SparseTensor(features, coords)
        return mx.sum(conv(tensor).features)

    gradient = mx.grad(loss)(mx.ones((2, 2)))
    assert gradient.shape == (2, 2)
    assert mx.all(mx.isfinite(gradient))


def test_sparse_downsample_upsample_and_subdivide():
    coords = _coords(2)
    tensor = SparseTensor(
        mx.arange(16, dtype=mx.float32).reshape(8, 2),
        coords,
        spatial_shape=(2, 2, 2),
    )
    down = sparse_downsample(tensor)
    up = sparse_upsample(down)
    subdivided = sparse_subdivide(tensor)
    assert down.coords.shape == (1, 4)
    assert up.coords.shape == tensor.coords.shape
    pooled = mx.sum(tensor.features, axis=0) / 9
    assert mx.allclose(up.features, mx.broadcast_to(pooled, (8, 2)))
    assert subdivided.coords.shape == (64, 4)


def test_subdivide_does_not_reuse_parent_neighbor_map():
    tensor = SparseTensor(mx.ones((8, 2)), _coords(2), spatial_shape=(2, 2, 2))
    conv = SparseConv3D(2, 2, 3, use_metal=False)
    parent = conv(tensor)
    children = sparse_subdivide(parent)
    output = conv(children)
    assert output.features.shape == (64, 2)


def test_slat_flow_preserves_sparse_topology():
    model = TrellisSLatFlowModel(
        TrellisSLatFlowConfig(
            resolution=8,
            in_channels=2,
            out_channels=2,
            model_channels=16,
            cond_channels=12,
            num_blocks=2,
            num_heads=2,
            mlp_ratio=2,
            io_block_channels=(4,),
            use_fp16=False,
        )
    )
    coords = _coords(4)
    sparse = SparseTensor(mx.random.normal((coords.shape[0], 2)), coords, spatial_shape=(8, 8, 8))
    output = model(sparse, mx.array([500.0]), mx.random.normal((1, 5, 12)))
    assert output.features.shape == sparse.features.shape
    assert mx.array_equal(output.coords, sparse.coords)
    assert mx.max(mx.abs(output.features)).item() < 1e-6


def test_dinov2_register_tokens_and_trellis_norm():
    model = DINOv2Model(
        DINOv2Config(
            image_size=28,
            patch_size=7,
            hidden_size=24,
            num_hidden_layers=2,
            num_attention_heads=3,
            mlp_ratio=2,
            num_register_tokens=2,
        )
    )
    output = model.trellis_conditioning(mx.random.normal((1, 28, 28, 3)))
    assert output.shape == (1, 1 + 2 + 16, 24)
    assert abs(mx.mean(output).item()) < 1e-5
    assert abs(mx.mean(mx.square(output)).item() - 1.0) < 1e-4


def test_gaussian_decoder_and_ply_export(tmp_path):
    decoder = TrellisGaussianDecoder(
        TrellisGaussianDecoderConfig(
            resolution=8,
            model_channels=16,
            latent_channels=2,
            num_blocks=2,
            num_heads=2,
            mlp_ratio=2,
            window_size=2,
            use_fp16=False,
            num_gaussians=4,
        )
    )
    coords = mx.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]], dtype=mx.int32)
    sparse = SparseTensor(mx.random.normal((3, 2)), coords, spatial_shape=(8, 8, 8))
    gaussian = decoder(sparse)[0]
    assert gaussian.xyz.shape == (12, 3)
    assert gaussian.scaling.shape == (12, 3)
    assert gaussian.rotation.shape == (12, 4)
    path = gaussian.save_ply(tmp_path / "asset.ply")
    assert path.read_text().startswith("ply\nformat ascii 1.0\nelement vertex 12\n")


def test_gaussian_ply_uses_official_coordinate_transform(tmp_path):
    from mlx_diffuser.models.trellis_gaussian import GaussianSplat3D

    gaussian = GaussianSplat3D(
        xyz_normalized=mx.array([[0.6, 0.7, 0.8]]),
        features_dc=mx.zeros((1, 1, 3)),
        scaling_raw=mx.zeros((1, 3)),
        rotation_raw=mx.zeros((1, 4)),
        opacity_raw=mx.zeros((1, 1)),
    )
    lines = gaussian.save_ply(tmp_path / "oriented.ply").read_text().splitlines()
    values = [float(value) for value in lines[lines.index("end_header") + 1].split()]
    assert np.allclose(values[:3], [0.1, -0.3, 0.2], atol=1e-6)
    assert np.allclose(values[-4:], [2**-0.5, 2**-0.5, 0.0, 0.0], atol=1e-6)


def test_official_config_converters_build_tiny_components():
    flow = TrellisSparseStructureFlowConverter().build_config(
        {"models": {"denoiser": {"args": {"model_channels": 24, "num_heads": 3}}}}
    )
    decoder = TrellisSparseStructureDecoderConverter().build_config(
        {"models": {"decoder": {"args": {"channels": [8, 4], "latent_channels": 2}}}}
    )
    slat = TrellisSLatFlowConverter().build_config(
        {"models": {"denoiser": {"args": {"model_channels": 16, "num_heads": 2}}}}
    )
    gaussian = TrellisGaussianDecoderConverter().build_config(
        {
            "models": {
                "decoder": {
                    "args": {
                        "model_channels": 16,
                        "num_heads": 2,
                        "representation_config": {"num_gaussians": 4},
                    }
                }
            }
        }
    )
    dino = DINOv2WithRegistersConverter().build_config(
        {"hidden_size": 24, "num_attention_heads": 3}
    )
    assert flow.model_channels == 24
    assert decoder.channels == (8, 4)
    assert slat.model_channels == 16
    assert gaussian.out_channels == 56
    assert dino.hidden_size == 24


def test_hub_style_flat_component_config_is_unwrapped():
    flow = TrellisSparseStructureFlowConverter().build_config(
        {
            "name": "SparseStructureFlowModel",
            "args": {
                "model_channels": 24,
                "num_heads": 3,
                "qk_rms_norm": True,
            },
        }
    )
    gaussian = TrellisGaussianDecoderConverter().build_config(
        {
            "name": "ElasticSLatGaussianDecoder",
            "args": {
                "model_channels": 24,
                "num_heads": 3,
                "representation_config": {
                    "num_gaussians": 7,
                    "voxel_size": 2.0,
                },
            },
        }
    )
    assert flow.qk_rms_norm is True
    assert gaussian.model_channels == 24
    assert gaussian.num_gaussians == 7
    assert gaussian.voxel_size == 2.0


def test_configless_official_flows_enable_checkpoint_rms_norm():
    assert TrellisSparseStructureFlowConverter().build_config({}).qk_rms_norm is True
    assert TrellisSLatFlowConverter().build_config({}).qk_rms_norm is True


def test_complete_tiny_image_to_gaussian_pipeline():
    dino = DINOv2Model(
        DINOv2Config(hidden_size=12, num_hidden_layers=1, num_attention_heads=3, mlp_ratio=2)
    )
    structure_flow = TrellisSparseStructureFlowModel(
        TrellisSparseStructureFlowConfig(
            resolution=2,
            model_channels=12,
            cond_channels=12,
            num_blocks=1,
            num_heads=3,
            mlp_ratio=2,
            use_fp16=False,
        )
    )
    structure_decoder = TrellisSparseStructureDecoder(
        TrellisSparseStructureDecoderConfig(
            channels=(4, 2), num_res_blocks=1, num_res_blocks_middle=1, use_fp16=False
        )
    )
    structure_decoder.out_layer[2].weight = mx.zeros_like(structure_decoder.out_layer[2].weight)
    structure_decoder.out_layer[2].bias = mx.ones_like(structure_decoder.out_layer[2].bias)
    slat_flow = TrellisSLatFlowModel(
        TrellisSLatFlowConfig(
            resolution=4,
            model_channels=12,
            cond_channels=12,
            num_blocks=1,
            num_heads=3,
            mlp_ratio=2,
            io_block_channels=(4,),
            use_fp16=False,
        )
    )
    gaussian_decoder = TrellisGaussianDecoder(
        TrellisGaussianDecoderConfig(
            resolution=4,
            model_channels=12,
            num_blocks=1,
            num_heads=3,
            mlp_ratio=2,
            window_size=2,
            use_fp16=False,
            num_gaussians=2,
        )
    )
    pipeline = TrellisImageTo3DPipeline(
        components={
            "image_conditioner": dino,
            "sparse_structure_flow": structure_flow,
            "sparse_structure_decoder": structure_decoder,
            "slat_flow": slat_flow,
            "gaussian_decoder": gaussian_decoder,
        }
    )
    conditioning = pipeline.encode_image(mx.zeros((1, 518, 518, 3)), low_memory=False)
    structure_key, slat_key = mx.random.split(mx.random.key(7))
    coords = pipeline.sample_sparse_structure(
        conditioning, key=structure_key, steps=1, low_memory=False
    )
    slat = pipeline.sample_slat(conditioning, coords, key=slat_key, steps=1, low_memory=False)
    gaussians = pipeline.decode_gaussians(slat, low_memory=False)
    assert conditioning.shape == (1, 1374, 12)
    assert coords.shape == (64, 4)
    assert slat.features.shape == (64, 8)
    assert gaussians[0].xyz.shape == (128, 3)
