"""LoRA tests: identity at init, freezing, merge equivalence, save/load, training."""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from mlx_diffusion.lora import inject_lora, load_lora, merge_lora, save_lora
from mlx_diffusion.lora.lora import LoRALinear
from mlx_diffusion.models import DiT, DiTConfig
from mlx_diffusion.schedulers import FlowMatchEulerScheduler
from mlx_diffusion.training import DiffusionTrainer


def build() -> DiT:
    return DiT(DiTConfig(in_channels=3, patch_size=2, hidden_size=16, depth=2, num_heads=2))


def _perturb_adapters(model: DiT) -> None:
    # Make lora_b non-zero so the adapter actually changes the output.
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_b = module.lora_b + 0.1 * mx.random.normal(module.lora_b.shape)
    mx.eval(model.parameters())


def test_inject_is_identity_at_init():
    model = build()
    x = mx.random.normal((1, 8, 8, 3))
    t = mx.array([0.5])
    out_before = model(x, t)
    n = inject_lora(model, rank=4, alpha=8)
    assert n > 0
    out_after = model(x, t)
    assert mx.allclose(out_before, out_after, atol=1e-5)


def test_freezing_only_trains_adapters():
    model = build()
    total = sum(v.size for _, v in tree_flatten(model.parameters()))
    inject_lora(model, rank=4)
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    assert 0 < trainable < total
    # Every trainable leaf is a lora param.
    assert all(
        "lora_a" in name or "lora_b" in name
        for name, _ in tree_flatten(model.trainable_parameters())
    )


def test_merge_equivalence():
    model = build()
    inject_lora(model, rank=4, alpha=8)
    _perturb_adapters(model)
    x = mx.random.normal((1, 8, 8, 3))
    t = mx.array([0.3])
    out_lora = model(x, t)
    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    out_merged = model(x, t)
    assert mx.allclose(out_lora, out_merged, atol=1e-4)


def test_save_load_roundtrip(tmp_path):
    base = build()
    base.save_pretrained(tmp_path / "base")

    m1 = DiT.from_pretrained(tmp_path / "base")
    inject_lora(m1, rank=4, alpha=8)
    _perturb_adapters(m1)
    save_lora(m1, tmp_path / "adapter", rank=4, alpha=8)

    m2 = DiT.from_pretrained(tmp_path / "base")
    load_lora(m2, tmp_path / "adapter")

    x = mx.random.normal((1, 8, 8, 3))
    t = mx.array([0.6])
    assert mx.allclose(m1(x, t), m2(x, t), atol=1e-4)


def test_lora_finetune_updates_adapters_only():
    model = build()
    # Simulate a pretrained base: the stock DiT's final layer is zero-init (and
    # frozen by LoRA), so gradients must reach the adapters via a non-zero base.
    model.update(tree_map(lambda p: p + 0.1 * mx.random.normal(p.shape), model.parameters()))
    mx.eval(model.parameters())

    inject_lora(model, rank=8, alpha=16)
    # Snapshot a frozen base weight and the (zero) adapters.
    base_w0 = mx.array(model.blocks[0].attn.q_proj.linear.weight)
    lora_b0 = mx.array(model.blocks[0].attn.q_proj.lora_b)

    trainer = DiffusionTrainer(model, FlowMatchEulerScheduler(), lr=5e-3, seed=0)
    x0 = mx.random.normal((4, 8, 8, 3))
    for _ in range(30):
        trainer.step(x0)

    # Adapters learned (moved off zero); base stayed frozen.
    assert mx.max(mx.abs(model.blocks[0].attn.q_proj.lora_b - lora_b0)).item() > 1e-4
    assert mx.allclose(model.blocks[0].attn.q_proj.linear.weight, base_w0)
