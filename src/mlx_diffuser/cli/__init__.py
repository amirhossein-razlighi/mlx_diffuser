"""Command-line interface: ``mlx-diffuser generate | train | convert``."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from ..utils import get_logger, to_array, to_pil

logger = get_logger()


# --------------------------------------------------------------------------- #
# generate — text-to-image / text-to-video over the real-checkpoint pipelines
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class _ModelSpec:
    """A named text-to-X model: where its checkpoint lives and how to run it."""

    modality: str  # "image" | "video"
    repo: str  # Hugging Face repo id (for --download)
    local: str  # default local checkpoint dir
    patterns: tuple[str, ...]  # snapshot_download allow_patterns
    run: Callable[[str, argparse.Namespace], mx.array]  # (folder, args) -> (B, ...) array
    aliases: tuple[str, ...] = ()


def _diffusers_patterns(*components: str) -> tuple[str, ...]:
    base = ["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*"]
    for c in components:
        base += [f"{c}/config.json", f"{c}/*.safetensors"]
    return tuple(base)


def _run_sdxl(folder: str, a: argparse.Namespace) -> mx.array:
    from ..pipelines import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_diffusers(folder, quantize_unet=a.quantize)
    return pipe(
        a.prompt,
        negative_prompt=a.negative or "",
        height=a.height,
        width=a.width,
        num_inference_steps=a.steps if a.steps is not None else 30,
        guidance_scale=a.guidance if a.guidance is not None else 5.0,
        seed=a.seed,
        cache_interval=int(a.cache) if a.cache else 1,
        tile_vae=a.tile_vae,
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
        tile_vae=a.tile_vae,
        release_text_encoders=a.release_text_encoders,
    )


def _run_wan(folder: str, a: argparse.Namespace) -> mx.array:
    from ..pipelines import WanPipeline

    pipe = WanPipeline.from_diffusers(folder, quantize_transformer=a.quantize)
    return pipe(
        a.prompt,
        negative_prompt=a.negative or "",
        num_frames=a.frames,
        height=a.height,
        width=a.width,
        num_inference_steps=a.steps if a.steps is not None else 30,
        guidance_scale=a.guidance if a.guidance is not None else 5.0,
        seed=a.seed,
        cache_threshold=a.cache or 0.0,
    )


MODELS: dict[str, _ModelSpec] = {
    "sdxl": _ModelSpec(
        "image",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "checkpoints/sdxl-base-1.0",
        _diffusers_patterns("text_encoder", "text_encoder_2", "unet", "vae"),
        _run_sdxl,
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
}


def _resolve_model(name: str) -> tuple[str, _ModelSpec]:
    """Map a model name (or alias) to its canonical key + spec."""
    for key, spec in MODELS.items():
        if name == key or name in spec.aliases:
            return key, spec
    known = sorted(set(MODELS) | {a for s in MODELS.values() for a in s.aliases})
    raise SystemExit(f"Unknown model {name!r}. Known text-to-X models: {known}")


def _save_output(out: Path, array: mx.array, modality: str, fps: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if modality == "video":  # (B, T, H, W, 3) -> animated GIF
        frames = [to_pil(array[0, i]) for i in range(array.shape[1])]
        frames[0].save(
            out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0
        )
    else:  # (B, H, W, 3) -> one image per batch element
        if array.shape[0] == 1:
            to_pil(array[0]).save(out)
        else:
            out.mkdir(parents=True, exist_ok=True)
            for i in range(array.shape[0]):
                to_pil(array[i]).save(out / f"sample_{i:03d}.png")
    logger.info("saved -> %s", out)


def _cmd_generate(args: argparse.Namespace) -> None:
    model = args.model or args.model_pos
    if model is None:
        raise SystemExit("pass a model: --model sdxl|flux|flux-dev|wan (or a path with --labels)")

    # Legacy class-conditional path: a saved DiffusionPipeline folder driven by --labels.
    if args.prompt is None:
        _generate_class_conditional(model, args)
        return

    key, spec = _resolve_model(model)
    if args.modality and args.modality != spec.modality:
        raise SystemExit(f"model {key!r} is a {spec.modality} model, not {args.modality!r}")

    folder = args.checkpoint or spec.local
    if args.download:
        _download(spec, folder)
    if not Path(folder).exists():
        raise SystemExit(
            f"checkpoint not found at {folder!r}. Run again with --download to fetch "
            f"{spec.repo!r}, or pass --checkpoint PATH."
        )

    # Defaults for height/width come from --size (square); per-modality if unset.
    size = args.size if args.size is not None else (256 if spec.modality == "video" else 1024)
    args.height = args.height if args.height is not None else size
    args.width = args.width if args.width is not None else size

    logger.info("generating %s with %s …", spec.modality, key)
    out_array = spec.run(folder, args)
    mx.eval(out_array)
    default_ext = ".gif" if spec.modality == "video" else ".png"
    out = Path(args.out) if args.out else Path(f"{key}{default_ext}")
    _save_output(out, out_array, spec.modality, args.fps)


def _download(spec: _ModelSpec, folder: str) -> None:
    from huggingface_hub import snapshot_download

    logger.info("downloading %s -> %s …", spec.repo, folder)
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
        help="text-to-image / text-to-video from a named model, or a saved pipeline",
        description="Examples:\n"
        '  mlx-diffuser generate --model flux  --prompt "a red fox in snow" --out fox.png\n'
        '  mlx-diffuser generate --model sdxl  --prompt "a lion at sunset"  --out lion.png\n'
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
    g.add_argument("--model", default=None, help="sdxl | flux | flux-dev | wan (or a path)")
    g.add_argument("--modality", choices=["image", "video"], default=None, help="cross-check")
    g.add_argument("--prompt", default=None, help="text prompt (selects the text-to-X path)")
    g.add_argument("--negative", default=None, help="negative prompt (SDXL / WAN)")
    g.add_argument("--out", default=None, help="output file (image .png / video .gif)")
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
        "--cache", type=float, default=None, help="DeepCache interval / FBCache threshold"
    )
    g.add_argument("--tile-vae", action="store_true", dest="tile_vae", help="tiled VAE decode")
    g.add_argument(
        "--release-text-encoders",
        action="store_true",
        dest="release_text_encoders",
        help="free text encoders before denoising (FLUX)",
    )
    g.add_argument("--frames", type=int, default=17, help="video frames (1 + multiple of 4)")
    g.add_argument("--fps", type=int, default=8, help="video output fps")
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
