"""DiffusionTrainer: a compact, compiled training loop for any model + scheduler.

Works for unconditional and class-conditional models alike. All randomness
(noise, timesteps, label dropout) is drawn eagerly and passed into a compiled,
pure step function, so ``mx.compile`` can fuse the forward/backward/update.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ..schedulers import Scheduler
from ..utils import get_logger
from .ema import EMA
from .losses import mse_loss

logger = get_logger()


class DiffusionTrainer:
    def __init__(
        self,
        model: nn.Module,
        scheduler: Scheduler,
        *,
        lr: float = 1e-4,
        optimizer: optim.Optimizer | None = None,
        weight_decay: float = 0.0,
        grad_clip: float | None = None,
        ema_decay: float | None = None,
        loss_weighting: Callable[[Scheduler, mx.array], mx.array] | None = None,
        class_dropout_prob: float = 0.0,
        compile: bool = True,
        seed: int = 0,
    ):
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer or optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.loss_weighting = loss_weighting
        self.class_dropout_prob = class_dropout_prob
        self.ema = EMA(model, ema_decay) if ema_decay else None
        self._key = mx.random.key(seed)

        self.conditional = getattr(model, "y_embed", None) is not None
        self._null_label = model.config.num_classes if self.conditional else 0

        self._loss_and_grad = nn.value_and_grad(self.model, self._loss)
        if compile:
            state = [self.model.state, self.optimizer.state]
            self._step_fn = mx.compile(self._raw_step, inputs=state, outputs=state)
        else:
            self._step_fn = self._raw_step

    # --- loss / step ------------------------------------------------------
    def _loss(self, x0: mx.array, y: mx.array, noise: mx.array, t: mx.array) -> mx.array:
        xt = self.scheduler.add_noise(x0, noise, t)
        target = self.scheduler.get_target(x0, noise, t)
        pred = self.model(xt, t, y) if self.conditional else self.model(xt, t)
        weights = self.loss_weighting(self.scheduler, t) if self.loss_weighting else None
        return mse_loss(pred, target, weights)

    def _raw_step(self, x0: mx.array, y: mx.array, noise: mx.array, t: mx.array) -> mx.array:
        loss, grads = self._loss_and_grad(x0, y, noise, t)
        if self.grad_clip is not None:
            grads, _ = optim.clip_grad_norm(grads, self.grad_clip)
        self.optimizer.update(self.model, grads)
        return loss

    def _prep_labels(self, y: mx.array, key: mx.array) -> mx.array:
        if self.class_dropout_prob > 0:
            drop = mx.random.uniform(shape=y.shape, key=key) < self.class_dropout_prob
            y = mx.where(drop, mx.array(self._null_label), y)
        return y

    def step(self, x0: mx.array, y: mx.array | None = None) -> mx.array:
        """One optimization step on a batch; returns the (scalar) loss."""
        self._key, k_noise, k_t, k_drop = mx.random.split(self._key, 4)
        noise = mx.random.normal(x0.shape, key=k_noise)
        t = self.scheduler.sample_timesteps(x0.shape[0], k_t)
        if self.conditional:
            if y is None:
                raise ValueError("Model is class-conditional; provide labels `y`.")
            y = self._prep_labels(y, k_drop)
        else:
            y = mx.zeros((x0.shape[0],), dtype=mx.int32)

        loss = self._step_fn(x0, y, noise, t)
        mx.eval(self.model.parameters(), self.optimizer.state, loss)
        if self.ema is not None:
            self.ema.update(self.model)
        return loss

    # --- driver -----------------------------------------------------------
    def fit(
        self,
        dataset: Iterable,
        *,
        steps: int | None = None,
        epochs: int = 1,
        log_every: int = 50,
    ) -> list[float]:
        """Train over ``dataset`` (an iterable of ``x0`` or ``(x0, y)`` batches)."""
        history: list[float] = []
        done = 0
        for _ in range(epochs):
            for batch in dataset:
                x0, y = batch if isinstance(batch, tuple) else (batch, None)
                loss = self.step(x0, y)
                history.append(loss.item())
                done += 1
                if done % log_every == 0:
                    logger.info("step %d  loss %.4f", done, history[-1])
                if steps is not None and done >= steps:
                    return history
        return history
