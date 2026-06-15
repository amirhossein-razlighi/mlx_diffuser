# Command-line interface

Installing the package adds an `mlx-diffuser` command.

## generate

```bash
mlx-diffuser generate MODEL --labels 1,2,3 --steps 50 --guidance 4.0 \
    --size 32 --seed 0 --dtype bf16 --quantize 4 --out samples/
```

`MODEL` is a local pipeline directory or a Hub repo id. Writes `sample_000.png`,
`sample_001.png`, ….

## train

Train from scratch or fine-tune on a folder of images:

```bash
# from scratch
mlx-diffuser train --data ./images --out my-model --steps 5000 \
    --batch 16 --size 32 --hidden 384 --depth 12 --scheduler flow --ema 0.999

# LoRA fine-tune of an existing model
mlx-diffuser train --data ./photos --base my-model --lora --lora-rank 8 \
    --out my-lora --steps 1000
```

## convert

Re-save a model with a new dtype or weight quantization:

```bash
mlx-diffuser convert my-model my-model-4bit --quantize 4
mlx-diffuser convert my-model my-model-bf16 --dtype bf16
```

Run `mlx-diffuser <command> --help` for the full option list.
