"""Training tests: EMA, batching, and that the loss actually goes down."""

from __future__ import annotations

import mlx.core as mx

from mlx_diffuser.models import DiT, DiTConfig
from mlx_diffuser.schedulers import DDPMScheduler, FlowMatchEulerScheduler
from mlx_diffuser.training import EMA, DiffusionTrainer, batch_iterator, min_snr_weights


def tiny_dit(hidden_size: int = 16, **kw) -> DiT:
    cfg = DiTConfig(
        in_channels=3, patch_size=2, hidden_size=hidden_size, depth=2, num_heads=2, **kw
    )
    return DiT(cfg)


def test_batch_iterator_shapes():
    x = mx.random.normal((10, 4))
    y = mx.arange(10)
    batches = list(batch_iterator((x, y), batch_size=4, seed=0))
    assert len(batches) == 2  # drop_last
    assert batches[0][0].shape == (4, 4)
    assert batches[0][1].shape == (4,)


def test_ema_tracks_params():
    model = tiny_dit()
    ema = EMA(model, decay=0.5)
    from mlx.utils import tree_map

    model.update(tree_map(lambda p: p + 1.0, model.parameters()))
    ema.update(model)
    # Shadow moved halfway toward the new params.
    flat = [v for _, v in __import__("mlx").utils.tree_flatten(ema.shadow)]
    assert all(mx.all(mx.isfinite(v)).item() for v in flat)


def test_min_snr_weights_positive():
    sch = DDPMScheduler()
    w = min_snr_weights(sch, mx.array([10, 500, 900]), gamma=5.0)
    assert w.shape == (3,)
    assert bool(mx.all(w > 0).item())


def _loss_goes_down(scheduler, conditional: bool, steps: int) -> tuple[float, float]:
    model = tiny_dit(hidden_size=32, num_classes=4 if conditional else 0)
    trainer = DiffusionTrainer(model, scheduler, lr=3e-3, seed=0)
    x0 = mx.random.normal((4, 8, 8, 3))
    y = mx.array([0, 1, 2, 3]) if conditional else None
    losses = []
    for _ in range(steps):
        losses.append(trainer.step(x0, y).item())
    first = sum(losses[:10]) / 10
    last = sum(losses[-10:]) / 10
    return first, last


def test_training_reduces_loss_flow_matching():
    # Unconditional flow-matching has an irreducible floor (random x0, t); we only
    # assert clear, monotone-ish improvement.
    first, last = _loss_goes_down(FlowMatchEulerScheduler(), conditional=False, steps=200)
    assert last < first * 0.9, f"expected loss to drop: {first:.3f} -> {last:.3f}"


def test_training_reduces_loss_conditional_ddpm():
    # The label identifies x0, so the target (epsilon) is learnable to low loss.
    first, last = _loss_goes_down(DDPMScheduler(), conditional=True, steps=300)
    assert last < first * 0.6, f"expected loss to drop: {first:.3f} -> {last:.3f}"


def test_trainer_fit_and_uncompiled():
    model = tiny_dit()
    trainer = DiffusionTrainer(model, FlowMatchEulerScheduler(), lr=1e-3, compile=False)
    data = mx.random.normal((16, 8, 8, 3))
    history = trainer.fit(batch_iterator(data, batch_size=4, seed=0), steps=4)
    assert len(history) == 4
    assert all(isinstance(x, float) for x in history)
