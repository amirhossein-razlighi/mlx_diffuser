"""Text-to-video generation on Apple silicon with mlx-diffuser.

Builds a VideoDiT + causal 3D VAE + flow-matching pipeline, generates a short
clip, and saves it as an animated GIF. Pass ``--quantize 4`` to load the
transformer in 4-bit and watch peak memory drop — the path that lets large video
models fit on a 16 GB Mac.

    uv run python examples/text_to_video.py --frames 16 --size 128 --steps 30
    uv run python examples/text_to_video.py --quantize 4            # low memory

Note: this demo uses randomly-initialized weights and random text embeddings, so
the output is noise. It exists to exercise the *pipeline and memory profile* on
your machine, not to produce a real video — that needs trained or converted
weights. The ``VideoDiTConfig.wan_t2v_1_3b()`` / ``.ltx_video()`` presets give the
real architecture shapes.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from mlx_diffuser.models import (
    AutoencoderKLVideo,
    AutoencoderKLVideoConfig,
    VideoDiT,
    VideoDiTConfig,
)
from mlx_diffuser.perf import memory_report, reset_peak_memory
from mlx_diffuser.pipelines import TextToVideoPipeline
from mlx_diffuser.schedulers import FlowMatchEulerScheduler
from mlx_diffuser.utils import to_pil


def build_pipeline(hidden: int, depth: int, quantize: int | None) -> TextToVideoPipeline:
    text_dim = 64
    vae = AutoencoderKLVideo(
        AutoencoderKLVideoConfig(
            in_channels=3,
            latent_channels=16,
            block_out_channels=(64, 128, 256),  # spatial compression 4
            layers_per_block=1,
            temporal_compression=4,
            norm_groups=32,
        )
    )
    cfg = VideoDiTConfig(
        in_channels=16,
        patch_size=(1, 2, 2),
        hidden_size=hidden,
        depth=depth,
        num_heads=max(1, hidden // 64),
        cross_attn_dim=text_dim,
    )
    transformer = VideoDiT(cfg)
    if quantize:
        # In practice you'd quantize a pretrained checkpoint via
        # VideoDiT.from_pretrained(path, quantize=4); here we quantize in place.
        from mlx_diffuser.quantization import quantize_module

        quantize_module(transformer, bits=quantize, group_size=64)
        mx.eval(transformer.parameters())
    return TextToVideoPipeline(transformer, vae, FlowMatchEulerScheduler())


def save_gif(video: mx.array, path: str, fps: int = 8) -> None:
    """Save ``(T, H, W, C)`` in [-1, 1] as an animated GIF."""
    frames = [to_pil(video[i]) for i in range(video.shape[0])]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--quantize", type=int, default=None, choices=[2, 3, 4, 6, 8])
    p.add_argument("--out", type=str, default="video.gif")
    args = p.parse_args()

    print("Architecture presets (real model shapes):")
    for name in ("wan_t2v_1_3b", "wan_t2v_14b", "ltx_video"):
        c = getattr(VideoDiTConfig, name)()
        print(f"  {name:14s} hidden={c.hidden_size} depth={c.depth} heads={c.num_heads}")

    reset_peak_memory()
    pipe = build_pipeline(args.hidden, args.depth, args.quantize)
    print(f"\ntransformer: {pipe.transformer.num_parameters() / 1e6:.1f}M params", end="")
    print(f"  (quantized {args.quantize}-bit)" if args.quantize else "")

    prompt_embeds = mx.random.normal((1, 16, pipe.transformer.config.cross_attn_dim))

    start = time.perf_counter()
    video = pipe(
        prompt_embeds,
        num_frames=args.frames,
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=0,
    )
    mx.eval(video)
    secs = time.perf_counter() - start
    mem = memory_report()

    print(f"generated {tuple(video.shape)} in {secs:.1f}s  |  peak {mem['peak_gb']:.2f} GB")
    save_gif(video[0], args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
