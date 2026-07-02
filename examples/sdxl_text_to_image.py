"""Text-to-image with Stable Diffusion XL on Apple silicon, end-to-end in MLX.

Downloads (once) and converts the official SDXL base checkpoint, then generates a
1024×1024 image from a prompt.

    # one-time download (~7 GB) into checkpoints/
    uv run python examples/sdxl_text_to_image.py --download

    uv run python examples/sdxl_text_to_image.py \
        --prompt "a majestic lion standing on a cliff at sunset, photorealistic" \
        --steps 30 --out lion.png

    # faster / smaller: DeepCache + 8-bit UNet + VAE tiling
    uv run python examples/sdxl_text_to_image.py --cache 2 --quant-unet 8 --tile-vae
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from mlx_diffuser.perf import memory_report, reset_peak_memory
from mlx_diffuser.pipelines import StableDiffusionXLPipeline
from mlx_diffuser.utils import to_pil

REPO = "stabilityai/stable-diffusion-xl-base-1.0"
LOCAL = "checkpoints/sdxl-base-1.0"


def download() -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading {REPO} (fp16) -> {LOCAL}…")
    snapshot_download(
        REPO,
        local_dir=LOCAL,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "tokenizer_2/*",
            "text_encoder/config.json",
            "text_encoder/*.fp16.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/*.fp16.safetensors",
            "unet/config.json",
            "unet/*.fp16.safetensors",
            "vae/config.json",
            "vae/*.fp16.safetensors",
        ],
    )
    print("done.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="download the checkpoint and exit")
    p.add_argument("--prompt", type=str, default="a majestic lion on a cliff at sunset, cinematic")
    p.add_argument("--negative", type=str, default="blurry, low quality")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", type=int, default=1, help="DeepCache interval (1=off; 2 ≈ 1.5-1.8×)")
    p.add_argument("--quant-unet", type=int, default=None, choices=[4, 8])
    p.add_argument("--tile-vae", action="store_true", help="tiled VAE decode (less memory)")
    p.add_argument("--out", type=str, default="sdxl.png")
    args = p.parse_args()

    if args.download:
        download()
        return

    reset_peak_memory()
    print("loading + converting SDXL (fp16 UNet/CLIP, fp32 VAE)…")
    pipe = StableDiffusionXLPipeline.from_diffusers(LOCAL, quantize_unet=args.quant_unet)

    start = time.perf_counter()
    image = pipe(
        args.prompt,
        negative_prompt=args.negative,
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        cache_interval=args.cache,
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
