"""Command-line interface: ``mlx-diffuser generate | train | convert``."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..utils import get_logger, to_array, to_pil

logger = get_logger()


# --------------------------------------------------------------------------- #
# generate — text-to-image / text-to-video over the real-checkpoint pipelines
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class _ModelSpec:
    """A named text-to-X model: where its checkpoint lives and how to run it."""

    modality: str  # "image" | "video" | "3d"
    repo: str  # Hugging Face repo id (for --download)
    local: str  # default local checkpoint dir
    patterns: tuple[str, ...]  # snapshot_download allow_patterns
    # (folder, args) -> (B, ...) array, or (video, audio) for audio-video models
    run: Callable[[str, argparse.Namespace], Any]
    aliases: tuple[str, ...] = ()
    default_hw: tuple[int, int] | None = None  # (height, width) when --size unset
    default_fps: int = 8
    video_ext: str = ".gif"
    download: Callable[[str], None] | None = None  # custom fetch (else snapshot_download)
    supports_image_input: bool = False


def _diffusers_patterns(*components: str) -> tuple[str, ...]:
    base = ["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*"]
    for c in components:
        base += [f"{c}/config.json", f"{c}/*.safetensors"]
    return tuple(base)


def _run_sdxl(folder: str, a: argparse.Namespace) -> mx.array:
    from ..pipelines import StableDiffusionXLPipeline

    quantize = a.quantize if a.quantize is not None else (8 if a.low_memory else None)
    pipe = StableDiffusionXLPipeline.from_diffusers(folder, quantize_unet=quantize)
    return pipe(
        a.prompt,
        negative_prompt=a.negative or "",
        image=a.image,
        strength=a.strength,
        height=a.height,
        width=a.width,
        num_inference_steps=a.steps if a.steps is not None else 30,
        guidance_scale=a.guidance if a.guidance is not None else 5.0,
        seed=a.seed,
        cache_interval=int(a.cache) if a.cache else 1,
        tile_vae=a.tile_vae or a.low_memory,
        release_text_encoders=a.release_text_encoders or a.low_memory,
    )


def _run_flux(folder: str, a: argparse.Namespace, *, dev: bool) -> mx.array:
    from ..pipelines import FluxPipeline

    q = a.quantize if a.quantize is not None else 4  # FLUX needs 4-bit to fit 16 GB
    pipe = FluxPipeline.from_diffusers(folder, quantize_transformer=q, quantize_t5=q)
    return pipe(
        a.prompt,
        height=a.height,
        width=a.width,
        num_inference_steps=a.steps if a.steps is not None else (50 if dev else 4),
        guidance_scale=a.guidance if a.guidance is not None else (3.5 if dev else 0.0),
        max_sequence_length=a.max_seq if a.max_seq is not None else (512 if dev else 256),
        seed=a.seed,
        cache_threshold=a.cache or 0.0,
        tile_vae=a.tile_vae or a.low_memory,
        release_text_encoders=a.release_text_encoders or a.low_memory,
    )


def _run_wan(folder: str, a: argparse.Namespace) -> mx.array:
    from ..pipelines import WanPipeline

    quantize = a.quantize if a.quantize is not None else (4 if a.low_memory else None)
    pipe = WanPipeline.from_diffusers(folder, quantize_transformer=quantize)
    return pipe(
        a.prompt,
        negative_prompt=a.negative or "",
        num_frames=a.frames if a.frames is not None else 17,
        height=a.height,
        width=a.width,
        num_inference_steps=a.steps if a.steps is not None else 30,
        guidance_scale=a.guidance if a.guidance is not None else 5.0,
        seed=a.seed,
        cache_threshold=a.cache or 0.0,
    )


def _run_ltx2(folder: str, a: argparse.Namespace) -> tuple[mx.array, mx.array]:
    from ..pipelines import LTX2Pipeline

    pipe = LTX2Pipeline.from_converted(folder)
    return pipe(  # (video, 48 kHz stereo audio)
        a.prompt,
        negative_prompt=a.negative or "",
        height=a.height,
        width=a.width,
        num_frames=a.frames if a.frames is not None else 121,
        frame_rate=a.fps if a.fps is not None else 24,
        guidance_scale=a.guidance if a.guidance is not None else 1.0,  # distilled: CFG off
        seed=a.seed,
        cache_threshold=a.cache or 0.0,
    )


def _download_ltx2(folder: str) -> None:
    from ..converters.ltx2 import convert_ltx2_checkpoint

    # LTX-2.3 ships as one 46 GB file + a 48 GB fp32 Gemma: stream-convert to
    # 4-bit MLX components instead of downloading the originals to disk.
    convert_ltx2_checkpoint(folder)


def _run_trellis(folder: str, args: argparse.Namespace):
    from ..pipelines import TrellisImageTo3DPipeline

    if not args.image:
        raise SystemExit("TRELLIS image-to-3D requires --image PATH")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(folder)
    return pipeline(
        args.image,
        seed=args.seed,
        sparse_structure_steps=args.steps if args.steps is not None else 25,
        slat_steps=args.steps if args.steps is not None else 25,
        remove_background=args.remove_background,
        low_memory=True,
    )


def _download_trellis(folder: str) -> None:
    from ..converters.trellis import download_and_convert_trellis

    download_and_convert_trellis(folder)


MODELS: dict[str, _ModelSpec] = {
    "sdxl": _ModelSpec(
        "image",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "checkpoints/sdxl-base-1.0",
        _diffusers_patterns("text_encoder", "text_encoder_2", "unet", "vae"),
        _run_sdxl,
        supports_image_input=True,
    ),
    "flux-schnell": _ModelSpec(
        "image",
        "black-forest-labs/FLUX.1-schnell",
        "checkpoints/flux1-schnell",
        _diffusers_patterns("text_encoder", "text_encoder_2", "transformer", "vae"),
        lambda f, a: _run_flux(f, a, dev=False),
        aliases=("flux",),
    ),
    "flux-dev": _ModelSpec(
        "image",
        "black-forest-labs/FLUX.1-dev",
        "checkpoints/flux1-dev",
        _diffusers_patterns("text_encoder", "text_encoder_2", "transformer", "vae"),
        lambda f, a: _run_flux(f, a, dev=True),
    ),
    "wan-1.3b": _ModelSpec(
        "video",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "checkpoints/wan2.1-t2v-1.3b",
        _diffusers_patterns("text_encoder", "transformer", "vae"),
        _run_wan,
        aliases=("wan",),
    ),
    "ltx-2.3": _ModelSpec(
        "video",
        "Lightricks/LTX-2.3",
        "checkpoints/ltx-2.3-distilled-mlx",
        (),  # custom streaming download+conversion (see _download_ltx2)
        _run_ltx2,
        aliases=("ltx", "ltx2"),
        default_hw=(512, 768),
        default_fps=24,
        video_ext=".mp4",
        download=_download_ltx2,
    ),
    "trellis": _ModelSpec(
        "3d",
        "microsoft/TRELLIS-image-large",
        "checkpoints/trellis-image-large-mlx",
        (),
        _run_trellis,
        aliases=("trellis-image",),
        video_ext=".ply",
        download=_download_trellis,
        supports_image_input=True,
    ),
}


def _resolve_model(name: str) -> tuple[str, _ModelSpec]:
    """Map a model name (or alias) to its canonical key + spec."""
    for key, spec in MODELS.items():
        if name == key or name in spec.aliases:
            return key, spec
    known = sorted(set(MODELS) | {a for s in MODELS.values() for a in s.aliases})
    raise SystemExit(f"Unknown model {name!r}. Known text-to-X models: {known}")


def _save_output(
    out: Path, array: Any, modality: str, fps: int, audio: mx.array | None = None
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if modality == "3d":
        array.save_ply(out)
    elif modality == "video" and out.suffix.lower() == ".mp4":  # (B, T, H, W, 3) -> H.264
        _save_mp4(out, array[0], fps, audio=audio)
    elif modality == "video":  # (B, T, H, W, 3) -> animated GIF
        frames = [to_pil(array[0, i]) for i in range(array.shape[1])]
        frames[0].save(
            out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0
        )
        if audio is not None:  # GIF can't carry sound: drop a wav next to it
            _save_wav(out.with_suffix(".wav"), audio)
            logger.info("GIF has no audio track; waveform saved -> %s", out.with_suffix(".wav"))
    else:  # (B, H, W, 3) -> one image per batch element
        if array.shape[0] == 1:
            to_pil(array[0]).save(out)
        else:
            out.mkdir(parents=True, exist_ok=True)
            for i in range(array.shape[0]):
                to_pil(array[i]).save(out / f"sample_{i:03d}.png")
    logger.info("saved -> %s", out)


def _save_wav(path: Path, audio: mx.array, sample_rate: int = 48000) -> None:
    """Write a ``(channels, samples)`` waveform in [-1, 1] as 16-bit PCM."""
    import wave

    import numpy as np

    pcm = np.array(mx.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(pcm.shape[0])
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.T.tobytes())  # interleaved


def _save_mp4(out: Path, frames: mx.array, fps: int, audio: mx.array | None = None) -> None:
    """Encode ``(T, H, W, 3)`` frames in [-1, 1] to H.264 via the ffmpeg binary.

    ``audio`` (``(channels, samples)`` at 48 kHz, [-1, 1]) is muxed in as AAC.
    """
    import shutil
    import subprocess
    import tempfile

    import numpy as np

    if shutil.which("ffmpeg") is None:
        raise SystemExit("saving .mp4 needs ffmpeg on PATH (brew install ffmpeg), or use .gif")
    u8 = np.array(mx.clip((frames + 1.0) * 127.5, 0, 255).astype(mx.uint8))
    t, h, w, _ = u8.shape
    video_in = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    video_out = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
    with tempfile.TemporaryDirectory() as tmp:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", *video_in]
        if audio is not None:
            wav = Path(tmp) / "audio.wav"
            _save_wav(wav, audio)
            cmd += ["-i", str(wav), *video_out, "-c:a", "aac", "-b:a", "192k", "-shortest"]
        else:
            cmd += video_out
        subprocess.run([*cmd, str(out)], input=u8.tobytes(), check=True)


def _cmd_generate(args: argparse.Namespace) -> None:
    model = args.model or args.model_pos
    if model is None:
        raise SystemExit(
            "pass a model: --model sdxl|flux|flux-dev|wan|ltx-2.3|trellis (or a path with --labels)"
        )

    try:
        key, spec = _resolve_model(model)
    except SystemExit:
        # Legacy class-conditional path: a saved DiffusionPipeline folder driven by labels.
        if args.prompt is None:
            _generate_class_conditional(model, args)
            return
        raise
    if args.prompt is None and spec.modality != "3d":
        raise SystemExit(f"model {key!r} requires --prompt")
    if args.modality and args.modality != spec.modality:
        raise SystemExit(f"model {key!r} is a {spec.modality} model, not {args.modality!r}")
    if args.image and not spec.supports_image_input:
        raise SystemExit(f"model {key!r} does not support --image conditioning yet.")
    if spec.modality == "3d" and not args.image:
        raise SystemExit(f"model {key!r} requires --image PATH")

    folder = args.checkpoint or spec.local
    if args.download:
        _download(spec, folder)
    if not Path(folder).exists():
        raise SystemExit(
            f"checkpoint not found at {folder!r}. Run again with --download to fetch "
            f"{spec.repo!r}, or pass --checkpoint PATH."
        )

    # Defaults for height/width come from --size (square), the model spec, or modality.
    if args.size is not None:
        default_h = default_w = args.size
    elif spec.default_hw is not None:
        default_h, default_w = spec.default_hw
    else:
        default_h = default_w = 256 if spec.modality == "video" else 1024
    args.height = args.height if args.height is not None else default_h
    args.width = args.width if args.width is not None else default_w

    logger.info("generating %s with %s …", spec.modality, key)
    mx.reset_peak_memory()
    out_array = spec.run(folder, args)
    peak_memory_gb = mx.get_peak_memory() / 1024**3
    audio = None
    if isinstance(out_array, tuple):  # audio-video models return (video, audio)
        out_array, audio = out_array
    if spec.modality != "3d":
        mx.eval(out_array)
    default_ext = spec.video_ext if spec.modality in {"video", "3d"} else ".png"
    out = Path(args.out) if args.out else Path(f"{key}{default_ext}")
    _save_output(
        out,
        out_array,
        spec.modality,
        args.fps if args.fps is not None else spec.default_fps,
        audio=audio,
    )
    logger.info("MLX peak memory: %.2f GB", peak_memory_gb)


def _download(spec: _ModelSpec, folder: str) -> None:
    logger.info("downloading %s -> %s …", spec.repo, folder)
    if spec.download is not None:
        spec.download(folder)
        return
    from huggingface_hub import snapshot_download

    snapshot_download(spec.repo, local_dir=folder, allow_patterns=list(spec.patterns))


def _generate_class_conditional(model: str, args: argparse.Namespace) -> None:
    from ..pipelines import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(model, dtype=args.dtype, quantize=args.quantize)
    labels = [int(x) for x in args.labels.split(",")]
    images = pipe(  # type: ignore[operator]  # concrete pipelines are callable
        labels,
        sample_size=args.size if args.size is not None else 32,
        num_inference_steps=args.steps if args.steps is not None else 50,
        guidance_scale=args.guidance if args.guidance is not None else 4.0,
        seed=args.seed,
        progress=True,
    )
    out = Path(args.out) if args.out else Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    for i in range(images.shape[0]):
        path = out / f"sample_{i:03d}.png"
        to_pil(images[i]).save(path)
        logger.info("wrote %s", path)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def _load_image_folder(folder: Path, size: int) -> mx.array:
    from PIL import Image

    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not paths:
        raise SystemExit(f"No images found in {folder}")
    imgs = [to_array(Image.open(p).convert("RGB").resize((size, size))) for p in paths]
    return mx.stack(imgs)


def _cmd_train(args: argparse.Namespace) -> None:
    from ..lora import inject_lora, save_lora
    from ..models import DiT, DiTConfig
    from ..schedulers import DDPMScheduler, FlowMatchEulerScheduler
    from ..training import DiffusionTrainer, batch_iterator

    data = _load_image_folder(Path(args.data), args.size)
    logger.info("loaded %d images at %dx%d", data.shape[0], args.size, args.size)

    if args.base:
        model = DiT.from_pretrained(args.base)
    else:
        model = DiT(
            DiTConfig(
                in_channels=3,
                hidden_size=args.hidden,
                depth=args.depth,
                num_heads=max(1, args.hidden // 64),
            )
        )

    if args.lora:
        n = inject_lora(model, rank=args.lora_rank)
        logger.info("injected LoRA into %d layers", n)

    scheduler = FlowMatchEulerScheduler() if args.scheduler == "flow" else DDPMScheduler()
    trainer = DiffusionTrainer(model, scheduler, lr=args.lr, ema_decay=args.ema)

    step = 0
    while step < args.steps:
        for batch in batch_iterator(data, args.batch, seed=step):
            loss = trainer.step(batch)
            step += 1
            if step % args.log_every == 0:
                logger.info("step %d/%d  loss %.4f", step, args.steps, loss.item())
            if step >= args.steps:
                break

    out = Path(args.out)
    if args.lora:
        save_lora(model, out, rank=args.lora_rank, alpha=2 * args.lora_rank)
    else:
        model.save_pretrained(out)
    logger.info("saved to %s", out)


# --------------------------------------------------------------------------- #
# convert
# --------------------------------------------------------------------------- #
def _cmd_convert(args: argparse.Namespace) -> None:
    from ..pipelines import DiffusionPipeline  # noqa: F401  (ensures registration)
    from ..pipelines.base import MODEL_REGISTRY  # registers models on import

    config = json.loads((Path(args.input) / "config.json").read_text())
    # config.json tags the config class name (e.g. "DiTConfig"); map it to a model.
    by_config = {m.config_class.__name__: m for m in MODEL_REGISTRY.values()}
    cls = by_config.get(config.get("_class_name"))
    if cls is None:
        raise SystemExit(
            f"Unknown model config {config.get('_class_name')!r}. Known: {sorted(by_config)}"
        )
    model = cls.from_pretrained(args.input, dtype=args.dtype, quantize=args.quantize)
    model.save_pretrained(args.output)
    logger.info(
        "converted %s -> %s (dtype=%s quantize=%s)",
        args.input,
        args.output,
        args.dtype,
        args.quantize,
    )


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlx-diffuser", description="Diffusion on Apple silicon with MLX."
    )
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser(
        "generate",
        help="image, video, and image-to-3D generation from a named model",
        description="Examples:\n"
        '  mlx-diffuser generate --model flux  --prompt "a red fox in snow" --out fox.png\n'
        '  mlx-diffuser generate --model sdxl  --prompt "a lion at sunset"  --out lion.png\n'
        '  mlx-diffuser generate --model sdxl --image photo.jpg --strength .65 --prompt "oil painting"\n'
        "  mlx-diffuser generate --model trellis --image object.png --out object.ply\n"
        '  mlx-diffuser generate --model wan --modality video --prompt "a panda surfing" '
        "--out panda.gif\n"
        "  mlx-diffuser generate --model flux --prompt ... --download   # fetch the checkpoint first",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g.add_argument(
        "model_pos",
        nargs="?",
        default=None,
        metavar="MODEL",
        help="(legacy) saved class-conditional pipeline path; prefer --model",
    )
    g.add_argument("--model", default=None, help="sdxl | flux | flux-dev | wan | ltx-2.3 | trellis")
    g.add_argument("--modality", choices=["image", "video", "3d"], default=None, help="cross-check")
    g.add_argument("--prompt", default=None, help="text prompt (selects the text-to-X path)")
    g.add_argument("--negative", default=None, help="negative prompt (SDXL / WAN)")
    g.add_argument("--image", default=None, help="input image path (SDXL img2img / TRELLIS 3D)")
    g.add_argument(
        "--remove-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="remove_background",
        help="TRELLIS: remove opaque backgrounds with optional rembg (default: on)",
    )
    g.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="input transformation strength in (0, 1] (default: 0.8)",
    )
    g.add_argument("--out", default=None, help="output file (image .png / video .gif / 3D .ply)")
    g.add_argument("--checkpoint", default=None, help="checkpoint dir override")
    g.add_argument("--download", action="store_true", help="download the checkpoint first")
    # generation knobs (per-model defaults applied when unset)
    g.add_argument("--steps", type=int, default=None)
    g.add_argument("--guidance", type=float, default=None)
    g.add_argument("--size", type=int, default=None, help="square size; or use --height/--width")
    g.add_argument("--height", type=int, default=None)
    g.add_argument("--width", type=int, default=None)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--quantize", type=int, default=None, help="weight bits (4/8)")
    g.add_argument(
        "--low-memory",
        action="store_true",
        dest="low_memory",
        help="16 GB preset: quantize large weights, release encoders, and tile VAE decode",
    )
    g.add_argument(
        "--cache", type=float, default=None, help="DeepCache interval / FBCache threshold"
    )
    g.add_argument("--tile-vae", action="store_true", dest="tile_vae", help="tiled VAE decode")
    g.add_argument(
        "--release-text-encoders",
        action="store_true",
        dest="release_text_encoders",
        help="free text encoders before denoising (SDXL / FLUX)",
    )
    g.add_argument(
        "--frames",
        type=int,
        default=None,
        help="video frames (WAN: 1+4k, default 17; LTX: 1+8k, default 121)",
    )
    g.add_argument("--fps", type=int, default=None, help="video fps (WAN default 8, LTX 24)")
    g.add_argument("--max-seq", type=int, default=None, dest="max_seq", help="FLUX T5 token budget")
    g.add_argument("--labels", default="0", help="(legacy) class labels, e.g. 1,2,3")
    g.add_argument("--dtype", default=None, help="(legacy) class-conditional dtype")
    g.set_defaults(func=_cmd_generate)

    t = sub.add_parser("train", help="train or fine-tune a DiT on an image folder")
    t.add_argument("--data", required=True, help="folder of images")
    t.add_argument("--out", required=True)
    t.add_argument("--base", default=None, help="base model to fine-tune (else train from scratch)")
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--size", type=int, default=32)
    t.add_argument("--hidden", type=int, default=384)
    t.add_argument("--depth", type=int, default=12)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--ema", type=float, default=None)
    t.add_argument("--scheduler", choices=["flow", "ddpm"], default="flow")
    t.add_argument("--lora", action="store_true", help="fine-tune with LoRA adapters")
    t.add_argument("--lora-rank", type=int, default=8, dest="lora_rank")
    t.add_argument("--log-every", type=int, default=50, dest="log_every")
    t.set_defaults(func=_cmd_train)

    c = sub.add_parser("convert", help="re-save a model with a new dtype / quantization")
    c.add_argument("input", help="model directory (config.json + weights)")
    c.add_argument("output")
    c.add_argument("--dtype", default=None)
    c.add_argument("--quantize", type=int, default=None)
    c.set_defaults(func=_cmd_convert)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
