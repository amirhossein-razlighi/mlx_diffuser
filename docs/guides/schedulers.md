# Schedulers

A scheduler is the **process**: it owns both the training-time corruption and the
sampling-time reverse step.

| Scheduler                  | Family                     | Typical prediction        |
|----------------------------|----------------------------|---------------------------|
| `DDPMScheduler`            | VP diffusion (ancestral)   | `epsilon` / `v` / `sample`|
| `DDIMScheduler`            | VP diffusion (deterministic)| `epsilon` / `v` / `sample`|
| `EulerDiscreteScheduler`   | sigma-space (SD/SDXL)      | `epsilon` / `v`           |
| `FlowMatchEulerScheduler`  | rectified flow (SD3/FLUX)  | `velocity`                |

## Common interface

```python
from mlx_diffuser.schedulers import FlowMatchEulerScheduler

sch = FlowMatchEulerScheduler()

# training
t = sch.sample_timesteps(batch_size, key)
x_t = sch.add_noise(x0, noise, t)
target = sch.get_target(x0, noise, t)

# sampling
sch.set_timesteps(50)
for step_t in sch.timesteps:
    x = sch.step(model_output, step_t, x)
```

## Choosing one

- **Flow-matching** (`FlowMatchEulerScheduler`) for modern transformer models — the
  default pairing for `DiT`. Use `shift > 1` for higher resolutions.
- **DDIM** for fast deterministic sampling of VP-trained models.
- **Euler** for Stable-Diffusion-style models.
- **DDPM** for the classic ancestral sampler and a reference implementation.

Prediction type is set on the config (e.g.
`DDPMConfig(prediction_type="v_prediction")`).

## Persistence

```python
sch.save_pretrained("scheduler/")
from mlx_diffuser.schedulers import load_scheduler
sch = load_scheduler("scheduler/")   # concrete class restored from config
```
