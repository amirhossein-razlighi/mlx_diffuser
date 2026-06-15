# LoRA fine-tuning

Low-rank adapters let you fine-tune a large model on a Mac by training a tiny set
of extra weights while the base stays frozen.

```python
from mlx_diffuser import DiT, DiffusionTrainer, inject_lora, save_lora, load_lora, merge_lora
from mlx_diffuser.schedulers import FlowMatchEulerScheduler
from mlx_diffuser.training import batch_iterator

model = DiT.from_pretrained("my-model")
n = inject_lora(model, rank=8, alpha=16)     # base frozen; only adapters trainable
print(f"adapted {n} layers")

trainer = DiffusionTrainer(model, FlowMatchEulerScheduler(), lr=5e-3)
trainer.fit(batch_iterator(data, batch_size=8), steps=1000)

save_lora(model, "my-lora", rank=8, alpha=16)  # adapter_config.json + safetensors
```

## Using an adapter

```python
model = DiT.from_pretrained("my-model")
load_lora(model, "my-lora")          # re-injects + loads adapter weights
```

## Merge for inference

Fuse adapters into dense weights for zero runtime overhead:

```python
merge_lora(model)                    # in place; LoRALinear -> nn.Linear
model.save_pretrained("my-merged-model")
```

## Notes

- Adapters are **identity at init** (`B` is zero), so injecting never changes
  outputs until you train.
- Default targets are the attention projections (`q_proj`, `k_proj`, `v_proj`,
  `out_proj`); pass `targets=` to change them.
- Choose `alpha` ≈ 2×`rank` as a reasonable starting point.
