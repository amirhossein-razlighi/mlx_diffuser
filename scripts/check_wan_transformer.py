"""Numerically verify the MLX WanTransformer3DModel against the diffusers reference.

Requires torch + diffusers and the downloaded transformer/ folder:

    PYTHONPATH=src python scripts/check_wan_transformer.py

Prints the max/mean absolute difference of the predicted flow output; should be
~1e-3 or smaller (bf16-free fp32 run; small drift from FP32LayerNorm ordering).
"""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/wan2.1-t2v-1.3b/transformer"


def main() -> None:
    import torch
    from diffusers import WanTransformer3DModel as RefT

    from mlx_diffuser.converters import get_converter

    torch.manual_seed(0)
    x = torch.randn(1, 16, 3, 16, 16, dtype=torch.float32)  # B, C, T, H, W (latent)
    t = torch.tensor([500.0], dtype=torch.float32)
    ctx = torch.randn(1, 12, 4096, dtype=torch.float32)

    ref = RefT.from_pretrained(FOLDER, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        out_t = ref(hidden_states=x, timestep=t, encoder_hidden_states=ctx, return_dict=True).sample

    model = get_converter("WanTransformer3DModel").convert(FOLDER)
    x_m = mx.array(x.numpy().transpose(0, 2, 3, 4, 1))  # -> B, T, H, W, C
    out_m = model(x_m, mx.array(t.numpy()), mx.array(ctx.numpy()))
    mx.eval(out_m)

    out = np.array(out_m).transpose(0, 4, 1, 2, 3)
    d = np.abs(out - out_t.numpy())
    print(f"output {out.shape}: max|d|={d.max():.3e} mean|d|={d.mean():.3e}")
    ok = d.max() < 5e-3
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
