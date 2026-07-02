"""Numerically verify the MLX WAN VAE against the official diffusers reference.

Requires the optional check deps (torch + diffusers) and a downloaded VAE folder:

    uv pip install torch diffusers
    python -c "from huggingface_hub import snapshot_download as d; \
        d('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', allow_patterns=['vae/*'], \
          local_dir='checkpoints/wan2.1-t2v-1.3b')"
    PYTHONPATH=src python scripts/check_wan_vae.py

Prints the max/mean absolute difference of the encoded latents and the
reconstruction; both should be ~1e-4 or smaller (float32 numerical noise).
"""

from __future__ import annotations

import sys

import mlx.core as mx
import numpy as np

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/wan2.1-t2v-1.3b/vae"


def main() -> None:
    import torch
    from diffusers import AutoencoderKLWan as RefVAE

    from mlx_diffuser.converters import get_converter

    torch.manual_seed(0)
    x_t = torch.randn(1, 3, 9, 32, 32, dtype=torch.float32)  # B, C, T, H, W

    ref = RefVAE.from_pretrained(FOLDER, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        lat_t = ref.encode(x_t).latent_dist.mode()
        rec_t = ref.decode(lat_t).sample

    vae = get_converter("AutoencoderKLWan").convert(FOLDER)
    x_m = mx.array(x_t.numpy().transpose(0, 2, 3, 4, 1))  # -> B, T, H, W, C
    lat_m = vae.encode(x_m).mode()
    rec_m = vae.decode(lat_m)
    mx.eval(lat_m, rec_m)

    lat = np.array(lat_m).transpose(0, 4, 1, 2, 3)
    rec = np.array(rec_m).transpose(0, 4, 1, 2, 3)
    le = np.abs(lat - lat_t.numpy())
    re = np.abs(rec - rec_t.numpy())
    print(f"latent {lat.shape}: max|d|={le.max():.3e} mean|d|={le.mean():.3e}")
    print(f"recon  {rec.shape}: max|d|={re.max():.3e} mean|d|={re.mean():.3e}")
    ok = le.max() < 1e-3 and re.max() < 1e-3
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
