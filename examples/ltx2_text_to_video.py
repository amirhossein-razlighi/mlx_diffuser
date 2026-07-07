"""Text-to-video with LTX-2.3 on Apple silicon, end-to-end in MLX.

LTX-2.3 is Lightricks' 22B-parameter joint audio-video model. The official
release is a single ~46 GB checkpoint plus a ~48 GB fp32 Gemma-3-12B text
encoder — far bigger than a 16 GB Mac. `--download` therefore *streams* the
originals over HTTP and quantizes each tensor as it arrives, writing ~20 GB of
4-bit MLX components and never holding the full-precision model on disk or in
RAM. Generation is staged (text encode -> free -> denoise -> free -> decode)
so the peak stays inside 16 GB.

    # one-time streaming download + conversion (transfers ~90 GB, writes ~20 GB)
    uv run python examples/ltx2_text_to_video.py --download

    uv run python examples/ltx2_text_to_video.py \
        --prompt "a golden retriever puppy chasing autumn leaves, sunny park" \
        --out puppy.mp4

The distilled checkpoint runs a fixed 8-step schedule with no CFG. Width and
height must be multiples of 32 and frames 1 + 8*k (default 768x512 x 121 at
24 fps ~= 5 s). Saving .mp4 needs ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import subprocess
import time

import mlx.core as mx
import numpy as np

from mlx_diffuser.perf import memory_report, reset_peak_memory
from mlx_diffuser.pipelines import LTX2Pipeline

LOCAL = "checkpoints/ltx-2.3-distilled-mlx"


def save_mp4(frames: mx.array, path: str, fps: float) -> None:
    u8 = np.array(mx.clip((frames + 1.0) * 127.5, 0, 255).astype(mx.uint8))
    t, h, w, _ = u8.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path,
    ]  # fmt: skip
    subprocess.run(cmd, input=u8.tobytes(), check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="stream-convert the checkpoint and exit")
    p.add_argument(
        "--prompt",
        type=str,
        default="a golden retriever puppy chasing autumn leaves in a sunny park, cinematic",
    )
    p.add_argument("--height", type=int, default=512, help="multiple of 32")
    p.add_argument("--width", type=int, default=768, help="multiple of 32")
    p.add_argument("--frames", type=int, default=121, help="1 + multiple of 8")
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", type=float, default=0.0, help="First-Block-Cache threshold (0=off)")
    p.add_argument("--out", type=str, default="ltx2.mp4")
    args = p.parse_args()

    if args.download:
        from mlx_diffuser.converters.ltx2 import convert_ltx2_checkpoint

        convert_ltx2_checkpoint(LOCAL)
        print("done.")
        return

    reset_peak_memory()
    pipe = LTX2Pipeline.from_converted(LOCAL)

    start = time.perf_counter()
    video = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        frame_rate=args.fps,
        seed=args.seed,
        cache_threshold=args.cache,
    )
    secs = time.perf_counter() - start
    mem = memory_report()
    print(f"\ngenerated {tuple(video.shape)} in {secs:.0f}s  |  peak {mem['peak_gb']:.2f} GB")

    save_mp4(video[0], args.out, args.fps)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
