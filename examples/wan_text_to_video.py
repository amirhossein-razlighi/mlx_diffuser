"""Text-to-video with WAN 2.1 on Apple silicon, end-to-end in MLX.

Downloads (once) and converts the official diffusers checkpoint, then generates a
short clip from a prompt and saves it as an MP4/GIF. The 5.6B umT5 text encoder is
loaded 4-bit so the whole stack (umT5 + DiT + VAE) fits in ~6 GB.

    # one-time download (~29 GB) into checkpoints/
    uv run python examples/wan_text_to_video.py --download

    uv run python examples/wan_text_to_video.py \
        --prompt "a red fox trotting through snow, cinematic" \
        --frames 17 --size 256 --steps 30 --out fox.gif

Tip: smaller --size / --frames / --steps run much faster on a 16 GB Mac.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from mlx_diffuser.perf import memory_report, reset_peak_memory
from mlx_diffuser.pipelines import WanPipeline
from mlx_diffuser.utils import to_pil

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
LOCAL = "checkpoints/wan2.1-t2v-1.3b"


def download() -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading {REPO} -> {LOCAL} (~29 GB, one time)…")
    snapshot_download(REPO, local_dir=LOCAL)
    print("done.")


def save_gif(video: mx.array, path: str, fps: int = 8) -> None:
    frames = [to_pil(video[i]) for i in range(video.shape[0])]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="download the checkpoint and exit")
    p.add_argument("--prompt", type=str, default="a red fox trotting through snow, cinematic")
    p.add_argument("--negative", type=str, default="")
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="wan.gif")
    args = p.parse_args()

    if args.download:
        download()
        return

    reset_peak_memory()
    print("loading + converting WAN 2.1 (umT5 4-bit, DiT bf16, VAE)…")
    pipe = WanPipeline.from_diffusers(LOCAL)

    start = time.perf_counter()
    video = pipe(
        args.prompt,
        negative_prompt=args.negative,
        num_frames=args.frames,
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    )
    mx.eval(video)
    secs = time.perf_counter() - start
    mem = memory_report()
    print(f"\ngenerated {tuple(video.shape)} in {secs:.1f}s  |  peak {mem['peak_gb']:.2f} GB")

    save_gif(video[0], args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
