# Quickstart

## Generate

```python
from mlx_diffusion import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("path/or/hub-id", dtype="bf16", quantize=4)
images = pipe([1, 2, 3], num_inference_steps=50, guidance_scale=4.0, seed=0)
# images: (B, H, W, C) in [-1, 1]
```

Convert to PIL:

```python
from mlx_diffusion import to_pil
to_pil(images[0]).save("sample.png")
```

## Train from scratch

```python
import mlx.core as mx
from mlx_diffusion import DiT, DiTConfig, DiffusionTrainer
from mlx_diffusion.schedulers import FlowMatchEulerScheduler
from mlx_diffusion.training import batch_iterator

data = mx.random.normal((512, 32, 32, 3))  # your images in [-1, 1], channels-last

model = DiT(DiTConfig(in_channels=3, hidden_size=384, depth=12, num_heads=6))
trainer = DiffusionTrainer(model, FlowMatchEulerScheduler(), lr=1e-4, ema_decay=0.999)
trainer.fit(batch_iterator(data, batch_size=32), steps=10_000)
model.save_pretrained("my-model")
```

Class-conditional training? Set `num_classes` and pass `(images, labels)` batches.

## Fine-tune with LoRA

```python
from mlx_diffusion import DiT, DiffusionTrainer, inject_lora, save_lora
from mlx_diffusion.schedulers import FlowMatchEulerScheduler
from mlx_diffusion.training import batch_iterator

model = DiT.from_pretrained("my-model")
inject_lora(model, rank=8)        # base frozen; only adapters train
trainer = DiffusionTrainer(model, FlowMatchEulerScheduler(), lr=5e-3)
trainer.fit(batch_iterator(data, batch_size=8), steps=1000)
save_lora(model, "my-lora", rank=8, alpha=16)
```

Load it back later with `load_lora(model, "my-lora")`, or bake it in with
`merge_lora(model)` for zero-overhead inference.

## From the command line

```bash
mlx-diffusion generate path/or/hub-id --labels 1,2,3 --steps 50 --out samples/
mlx-diffusion train --data ./photos --out my-model --steps 2000
mlx-diffusion train --data ./photos --base my-model --lora --out my-lora
mlx-diffusion convert my-model my-model-4bit --quantize 4
```
