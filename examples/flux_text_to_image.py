"""Text-to-image with FLUX.1 on Apple silicon, end-to-end in MLX.

Downloads (once) and converts the official FLUX.1-schnell checkpoint, then generates a
1024×1024 image from a prompt. FLUX.1 is a 12B-parameter model, so the transformer and
the T5 encoder are loaded **4-bit** by default — the whole pipeline then fits in ~10 GB
of unified memory and runs on a 16 GB Mac.

    # one-time download (~34 GB) into checkpoints/ (needs `huggingface-cli login` and
    # accepting the license at https://huggingface.co/black-forest-labs/FLUX.1-schnell)
    uv run python examples/flux_text_to_image.py --download

    uv run python examples/flux_text_to_image.py \
        --prompt "a majestic lion standing on a cliff at sunset, photorealistic" \
        --steps 4 --out flux.png

    # lower peak memory (free the text encoders before denoising) + First-Block cache
    uv run python examples/flux_text_to_image.py --release-text-encoders --cache 0.1
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from mlx_diffuser.perf import memory_report, reset_peak_memory
from mlx_diffuser.pipelines import FluxPipeline
from mlx_diffuser.utils import to_pil

REPO = "black-forest-labs/FLUX.1-schnell"
LOCAL = "checkpoints/flux1-schnell"


def download() -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading {REPO} (bf16) -> {LOCAL}…")
    snapshot_download(
        REPO,
        local_dir=LOCAL,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "tokenizer_2/*",
            "text_encoder/config.json",
            "text_encoder/*.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/*.safetensors",
            "transformer/config.json",
            "transformer/*.safetensors",
            "vae/config.json",
            "vae/*.safetensors",
        ],
    )
    print("done.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="download the checkpoint and exit")
    p.add_argument("--prompt", type=str, default="a majestic lion on a cliff at sunset, cinematic")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=4, help="schnell: ~4; dev: ~50")
    p.add_argument("--guidance", type=float, default=0.0, help="dev only (schnell ignores it)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-seq", type=int, default=256, help="T5 token budget (schnell 256, dev 512)"
    )
    p.add_argument("--quant", type=int, default=4, choices=[4, 8], help="transformer/T5 bits")
    p.add_argument("--cache", type=float, default=0.0, help="First-Block-Cache threshold (0=off)")
    p.add_argument(
        "--release-text-encoders", action="store_true", help="free CLIP+T5 before denoising"
    )
    p.add_argument("--tile-vae", action="store_true", help="tiled VAE decode (keeps <16 GB)")
    p.add_argument("--out", type=str, default="flux.png")
    args = p.parse_args()

    if args.download:
        download()
        return

    reset_peak_memory()
    print(f"loading + converting FLUX.1-schnell ({args.quant}-bit transformer/T5, fp32 VAE)…")
    pipe = FluxPipeline.from_diffusers(
        LOCAL, quantize_transformer=args.quant, quantize_t5=args.quant
    )

    start = time.perf_counter()
    image = pipe(
        args.prompt,
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        max_sequence_length=args.max_seq,
        seed=args.seed,
        cache_threshold=args.cache,
        release_text_encoders=args.release_text_encoders,
        tile_vae=args.tile_vae,
    )
    mx.eval(image)
    secs = time.perf_counter() - start
    mem = memory_report()
    print(f"\ngenerated {tuple(image.shape)} in {secs:.1f}s  |  peak {mem['peak_gb']:.2f} GB")

    to_pil(image[0]).save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
