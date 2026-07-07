"""Tests for the LTX-2.3 component ports on tiny configs (no downloads)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_diffuser.caching import FirstBlockCache
from mlx_diffuser.converters.ltx2 import (
    _ComponentWriter,
    map_connectors_key,
    map_gemma_key,
    map_transformer_key,
    map_vae_key,
)
from mlx_diffuser.models import (
    Gemma3Config,
    Gemma3TextEncoder,
    LTX2ConnectorsConfig,
    LTX2TextConnectors,
    LTX2Transformer3DModel,
    LTX2TransformerConfig,
    LTX2VAEDecoderConfig,
    LTX2VideoDecoder,
)


def tiny_transformer(**kw) -> LTX2TransformerConfig:
    base = dict(
        in_channels=8,
        out_channels=8,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        num_layers=2,
        caption_channels=16,
    )
    base.update(kw)
    return LTX2TransformerConfig(**base)


def tiny_connectors(**kw) -> LTX2ConnectorsConfig:
    base = dict(
        caption_channels=16,
        text_proj_in_factor=3,
        video_hidden_dim=12,
        audio_hidden_dim=8,
        video_num_attention_heads=2,
        video_attention_head_dim=6,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        num_layers=2,
        num_learnable_registers=4,
    )
    base.update(kw)
    return LTX2ConnectorsConfig(**base)


def tiny_vae(**kw) -> LTX2VAEDecoderConfig:
    base = dict(
        latent_channels=8,
        base_channels=4,
        patch_size=4,
        decoder_blocks=(
            ("res_x", {"num_layers": 1}),
            ("compress_space", {"multiplier": 2}),
            ("res_x", {"num_layers": 1}),
            ("compress_time", {"multiplier": 2}),
            ("res_x", {"num_layers": 1}),
            ("compress_all", {"multiplier": 1}),
            ("res_x", {"num_layers": 1}),
            ("compress_all", {"multiplier": 2}),
            ("res_x", {"num_layers": 1}),
        ),
    )
    base.update(kw)
    return LTX2VAEDecoderConfig(**base)


def tiny_gemma(**kw) -> Gemma3Config:
    base = dict(
        vocab_size=64,
        hidden_size=32,  # multiple of the minimum quantization group size (32)
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        query_pre_attn_scalar=8.0,
        sliding_window=4,
    )
    base.update(kw)
    return Gemma3Config(**base)


def _run_tiny_transformer(model, batch=1, cache=None):
    f, h, w, sa = 2, 3, 4, 5
    coords_v, coords_a = model.prepare_coords(batch, f, h, w, sa, 24.0)
    return model(
        mx.random.normal((batch, f * h * w, 8)),
        mx.random.normal((batch, sa, 4)),
        mx.random.normal((batch, 6, 12)),
        mx.random.normal((batch, 6, 8)),
        mx.full((batch,), 431.0),
        coords_v,
        coords_a,
        cache=cache,
    )


def test_transformer_output_shapes():
    model = LTX2Transformer3DModel(tiny_transformer())
    video, audio = _run_tiny_transformer(model)
    assert video.shape == (1, 24, 8)
    assert audio.shape == (1, 5, 4)


def test_transformer_first_block_cache():
    model = LTX2Transformer3DModel(tiny_transformer())
    cache = FirstBlockCache(threshold=1e9)  # force reuse on the second call
    v1, a1 = _run_tiny_transformer(model, cache=cache)
    v2, a2 = _run_tiny_transformer(model, cache=cache)
    assert cache.skipped >= 1
    assert v2.shape == v1.shape and a2.shape == a1.shape


def test_connectors_shapes_and_register_fill():
    conn = LTX2TextConnectors(tiny_connectors())
    hidden = mx.random.normal((2, 8, 16, 3))
    mask = mx.ones((2, 8), dtype=mx.int32)
    mask[0, :3] = 0  # left padding on the first row
    video, audio = conn(hidden, mask)
    assert video.shape == (2, 8, 12)
    assert audio.shape == (2, 8, 8)


def test_vae_decoder_shape():
    vae = LTX2VideoDecoder(tiny_vae())
    z = mx.random.normal((1, 3, 2, 2, 8))
    video = vae.decode(vae.denormalize_latents(z))
    # 8x temporal (8*(3-1)+1 = 17) and 32x spatial (2*32 = 64)
    assert video.shape == (1, 17, 64, 64, 3)


def test_gemma_hidden_states():
    model = Gemma3TextEncoder(tiny_gemma())
    ids = mx.array([[1, 2, 3, 4, 5, 6]])
    mask = mx.array([[0, 0, 1, 1, 1, 1]])
    states = model(ids, mask)
    assert states.shape == (1, 6, 32, 4)  # embeddings + 3 layers
    assert bool(mx.all(mx.isfinite(states)))


def test_converter_key_mapping():
    dm = "model.diffusion_model."
    assert map_transformer_key(f"{dm}patchify_proj.weight") == "proj_in.weight"
    assert map_transformer_key(f"{dm}audio_patchify_proj.bias") == "audio_proj_in.bias"
    assert (
        map_transformer_key(f"{dm}adaln_single.emb.timestep_embedder.linear_1.weight")
        == "time_embed.emb.timestep_embedder.linear_1.weight"
    )
    assert (
        map_transformer_key(f"{dm}audio_adaln_single.linear.bias") == "audio_time_embed.linear.bias"
    )
    assert (
        map_transformer_key(f"{dm}prompt_adaln_single.linear.weight")
        == "prompt_adaln.linear.weight"
    )
    assert (
        map_transformer_key(f"{dm}transformer_blocks.0.attn1.q_norm.weight")
        == "transformer_blocks.0.attn1.norm_q.weight"
    )
    assert (
        map_transformer_key(f"{dm}transformer_blocks.3.scale_shift_table_a2v_ca_video")
        == "transformer_blocks.3.video_a2v_cross_attn_scale_shift_table"
    )
    assert map_transformer_key(f"{dm}video_embeddings_connector.learnable_registers") is None
    assert map_transformer_key("vae.decoder.conv_in.conv.weight") is None

    assert (
        map_connectors_key("text_embedding_projection.video_aggregate_embed.weight")
        == "video_text_proj_in.weight"
    )
    assert (
        map_connectors_key(
            f"{dm}audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"
        )
        == "audio_connector.transformer_blocks.0.attn1.to_q.weight"
    )

    assert map_vae_key("vae.decoder.up_blocks.1.conv.conv.weight") == "up_blocks.1.conv.conv.weight"
    assert map_vae_key("vae.per_channel_statistics.std-of-means") == "latents_std"
    assert map_vae_key("vae.per_channel_statistics.channel") is None
    assert map_vae_key("audio_vae.decoder.conv_in.weight") is None

    assert map_gemma_key("language_model.model.layers.0.self_attn.q_proj.weight") == (
        "layers.0.self_attn.q_proj.weight"
    )
    assert map_gemma_key("vision_tower.vision_model.embeddings.patch_embedding.weight") is None


def test_component_writer_quantized_roundtrip(tmp_path):
    """_ComponentWriter output loads back through from_pretrained (pre-quantized)."""
    source = Gemma3TextEncoder(tiny_gemma())
    from mlx.utils import tree_flatten

    source_params = dict(tree_flatten(source.parameters()))

    writer = _ComponentWriter(
        Gemma3TextEncoder(tiny_gemma()), tmp_path / "text_encoder", quantize=4, group_size=32
    )
    for name, tensor in source_params.items():
        writer.add(name, tensor)
    writer.finish()

    model = Gemma3TextEncoder.from_pretrained(tmp_path / "text_encoder")
    ids = mx.array([[1, 2, 3, 4]])
    out = model(ids, mx.ones((1, 4), dtype=mx.int32))
    assert out.shape == (1, 4, 32, 4)
    assert bool(mx.all(mx.isfinite(out)))


def test_pipeline_end_to_end_tiny(tmp_path, monkeypatch):
    """The staged pipeline runs on a tiny converted checkpoint with a stub tokenizer."""
    from mlx.utils import tree_flatten

    from mlx_diffuser.pipelines.ltx2 import LTX2Pipeline

    layers = 4  # tiny gemma: 3 layers -> 4 hidden states
    specs = [
        ("text_encoder", Gemma3TextEncoder(tiny_gemma()), 4),
        (
            "connectors",
            LTX2TextConnectors(tiny_connectors(caption_channels=32, text_proj_in_factor=layers)),
            None,
        ),
        ("transformer", LTX2Transformer3DModel(tiny_transformer()), None),
        ("vae_decoder", LTX2VideoDecoder(tiny_vae(latent_channels=8)), None),
    ]
    for name, model, bits in specs:
        writer = _ComponentWriter(model, tmp_path / name, quantize=bits, group_size=32)
        for key, tensor in tree_flatten(
            type(model)(model.config).parameters()  # fresh params, unquantized tree
        ):
            if tensor.ndim == 5:  # the writer expects PyTorch conv layout
                tensor = tensor.transpose(0, 4, 1, 2, 3)
            writer.add(key, tensor)
        writer.finish()
    (tmp_path / "tokenizer").mkdir()

    class _StubTokenizer:
        pad_token = "<pad>"

        def __call__(self, text, **kw):
            return {
                "input_ids": np.array([[1, 2, 3, 4, 5, 6, 7, 8]]),
                "attention_mask": np.array([[0, 0, 1, 1, 1, 1, 1, 1]]),
            }

    pipe = LTX2Pipeline(tmp_path)
    monkeypatch.setattr(pipe, "_load_tokenizer", lambda: _StubTokenizer())
    # tiny transformer: in_channels=8, caption dims 12/8; tiny vae: 8 latent channels
    video = pipe("a tiny test", height=64, width=64, num_frames=9, seed=0)
    assert video.shape == (1, 9, 64, 64, 3)
    assert bool(mx.all(mx.isfinite(video)))
