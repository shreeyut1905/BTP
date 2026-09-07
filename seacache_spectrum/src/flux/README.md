# `src/flux/` — FLUX.1-dev patch

`flux_forward.py` replaces `FluxTransformer2DModel.forward` with
`cached_flux_forward`: same signature, same outputs, but steps whose
SEA-gated accumulator is below δ are answered from cache instead of
computed. Call `reset_cache_state` once per generation.

Minimal inference (run from `seacache_spectrum/`):

```python
import sys, torch; sys.path.insert(0, "src")
from argparse import Namespace
from diffusers import DiffusionPipeline
from flux.flux_forward import cached_flux_forward, reset_cache_state

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                         torch_dtype=torch.bfloat16).to("cuda")
pipe.transformer.__class__.forward = cached_flux_forward

args = Namespace(seacache_thresh=0.3,        # δ: 0.3 fast-good, 0.6 faster
                 spectrum_m=4, spectrum_k=100, spectrum_lam=0.1,
                 spectrum_w=0.5, spectrum_taylor_order=1,
                 spectrum_min_cheb_obs=4)
reset_cache_state(pipe.transformer, num_steps=50, mode="hybrid", args=args)
image = pipe("A red colored car.", num_inference_steps=50).images[0]
# mode="base" for the uncached reference, "seacache" for gate + copy-reuse.
```

Full eval — base vs hybrid PSNR/SSIM/LPIPS on DrawBench-200:

```bash
# 4-prompt smoke first (~1 min), then the full 200
.venv/bin/python scripts/eval_flux.py --num_prompts 4 --seacache_thresh 0.3 --output_dir ./outputs
.venv/bin/python scripts/eval_flux.py --prompt_file prompts/drawbench200.txt \
  --num_prompts 200 --seacache_thresh 0.3 --output_dir ./outputs_hybrid_d03
# metrics-only pass vs a base run (no GPU model load):
.venv/bin/python scripts/compute_metrics.py --base_dir ./outputs_base --hybrid_dir ./outputs_hybrid_d03
```

Reference numbers (B200): δ0.3 → 27.97 dB / 0.918 SSIM / 0.070 LPIPS @1241
TFLOPs; δ0.6 → 21.63 dB @774 TFLOPs. Compare against TFLOPs (ref 2976),
never wall seconds across hardware.
