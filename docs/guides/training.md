# Training

`DiffusionTrainer` ties a model + scheduler + optimizer into a compiled training
loop. It works for unconditional and class-conditional models.

```python
from mlx_diffuser import DiT, DiTConfig, DiffusionTrainer
from mlx_diffuser.schedulers import FlowMatchEulerScheduler
from mlx_diffuser.training import batch_iterator, min_snr_weights

model = DiT(DiTConfig(in_channels=3, hidden_size=384, depth=12, num_heads=6, num_classes=10))
trainer = DiffusionTrainer(
    model,
    FlowMatchEulerScheduler(),
    lr=1e-4,
    weight_decay=0.0,
    grad_clip=1.0,
    ema_decay=0.999,
    class_dropout_prob=0.1,         # for classifier-free guidance
)

history = trainer.fit(
    batch_iterator((images, labels), batch_size=32),
    steps=10_000,
    log_every=100,
)
```

## How it works

Each step draws noise and timesteps eagerly, then runs a single `mx.compile`-fused
function that fuses forward + backward + optimizer update. Temporaries are released
before evaluation to keep peak memory low.

```python
loss = trainer.step(x0, y)   # one batch; returns a scalar loss
```

## Loss weighting

Pass `loss_weighting=` a callable `(scheduler, t) -> weights`. The built-in
`min_snr_weights` implements Min-SNR-γ for VP diffusion:

```python
from functools import partial
trainer = DiffusionTrainer(model, scheduler,
                           loss_weighting=partial(min_snr_weights, gamma=5.0))
```

## EMA

Set `ema_decay` to track an exponential moving average of the weights; copy them in
for sampling:

```python
trainer.ema.copy_to(model)
model.save_pretrained("my-model-ema")
```

## Tips

- Keep batches a constant shape so the compiled step is not retraced.
- Use `dtype="bf16"` weights for large models; norms accumulate in higher precision
  automatically.
- See [Performance](performance.md) for memory and compilation knobs.
