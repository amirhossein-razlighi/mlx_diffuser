"""Command-line interface: ``mlx-diffusion generate | train | convert``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from ..utils import get_logger, to_array, to_pil

logger = get_logger()


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def _cmd_generate(args: argparse.Namespace) -> None:
    from ..pipelines import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(args.model, dtype=args.dtype, quantize=args.quantize)
    labels = [int(x) for x in args.labels.split(",")]
    images = pipe(
        labels,
        sample_size=args.size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        progress=True,
    )
    out = Path(args.out)
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

    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not paths:
        raise SystemExit(f"No images found in {folder}")
    imgs = [to_array(Image.open(p).convert("RGB").resize((size, size))) for p in paths]
    return mx.stack(imgs)


def _cmd_train(args: argparse.Namespace) -> None:
    from ..models import DiT, DiTConfig
    from ..lora import inject_lora, save_lora
    from ..schedulers import DDPMScheduler, FlowMatchEulerScheduler
    from ..training import DiffusionTrainer, batch_iterator

    data = _load_image_folder(Path(args.data), args.size)
    logger.info("loaded %d images at %dx%d", data.shape[0], args.size, args.size)

    if args.base:
        model = DiT.from_pretrained(args.base)
    else:
        model = DiT(DiTConfig(in_channels=3, hidden_size=args.hidden, depth=args.depth, num_heads=max(1, args.hidden // 64)))

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
    from ..pipelines.base import MODEL_REGISTRY  # registers models on import
    from ..pipelines import DiffusionPipeline  # noqa: F401  (ensures registration)

    config = json.loads((Path(args.input) / "config.json").read_text())
    # config.json tags the config class name (e.g. "DiTConfig"); map it to a model.
    by_config = {m.config_class.__name__: m for m in MODEL_REGISTRY.values()}
    cls = by_config.get(config.get("_class_name"))
    if cls is None:
        raise SystemExit(
            f"Unknown model config {config.get('_class_name')!r}. "
            f"Known: {sorted(by_config)}"
        )
    model = cls.from_pretrained(args.input, dtype=args.dtype, quantize=args.quantize)
    model.save_pretrained(args.output)
    logger.info("converted %s -> %s (dtype=%s quantize=%s)", args.input, args.output, args.dtype, args.quantize)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlx-diffusion", description="Diffusion on Apple silicon with MLX.")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="sample images from a saved pipeline")
    g.add_argument("model", help="local path or Hub repo id")
    g.add_argument("--labels", default="0", help="comma-separated class labels, e.g. 1,2,3")
    g.add_argument("--steps", type=int, default=50)
    g.add_argument("--guidance", type=float, default=4.0)
    g.add_argument("--size", type=int, default=32)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--dtype", default=None)
    g.add_argument("--quantize", type=int, default=None)
    g.add_argument("--out", default="outputs")
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
