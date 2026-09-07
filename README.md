# ReSPect: Residual Spectral Prediction for Efficient Caching of Transformers

Training-free diffusion cache: **SeaCache's** spectral-evolution (SEA) skip
schedule + **Spectrum-style** residual forecasting instead of residual reuse.
On skipped steps we forecast the transformer residual (Chebyshev ridge `M=4`,
`λ=0.1`, blended `w=0.5` with a discrete Taylor step, warm-up `min_cheb_obs=4`)
rather than copying the last cached residual.

Upstreams (not vendored here): [SeaCache](https://github.com/jiwoogit/SeaCache)
(CVPR 2026 Oral), [Spectrum](https://github.com/hanjq17/Spectrum) (CVPR 2026).

## Supported models

| Model | Patch | Eval |
|---|---|---|
| FLUX.1-dev 1024×1024 | `seacache_spectrum/flux_forward.py` | `compare_psnr.py` + `prompts.txt` (DrawBench-200) |
| Wan2.1-1.3B 480p×65f | `seacache_spectrum/wan_hybrid_forward.py` | `compare_video.py` + `vbench_prompts.txt` (VBench-946) |
| HunyuanVideo 480p×65f | `seacache_spectrum/hunyuan_hybrid_forward.py` (diffusers path) | `compare_video.py` |

Core (shared): `spectrum_forecaster.py`, `util_seacache.py`.
Method note: [`seacache_spectrum/research.md`](seacache_spectrum/research.md).
Paper draft: `paper/main.tex`.

## Setup

```bash
cd seacache_spectrum
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
# FLUX.1-dev is gated — log in first:
.venv/bin/huggingface-cli login
```

## Inference

```python
from diffusers import DiffusionPipeline
from flux_forward import cached_flux_forward, reset_cache_state
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                         torch_dtype=torch.bfloat16).to("cuda")
pipe.transformer.__class__.forward = cached_flux_forward
reset_cache_state(pipe.transformer, num_steps=50, mode="hybrid", args)  # or "base"/"seacache"
```

## Eval (quality vs uncached base, same seed)

```bash
cd seacache_spectrum
# FLUX, 4-prompt smoke
.venv/bin/python compare_psnr.py --num_prompts 4 --seacache_thresh 0.3 --output_dir ./outputs
# FLUX, DrawBench-200
.venv/bin/python compare_psnr.py --num_prompts 200 --seacache_thresh 0.3 --output_dir ./outputs_hybrid_d03
.venv/bin/python compute_metrics.py --base_dir ./outputs_base --hybrid_dir ./outputs_hybrid_d03
# Wan2.1 (see seacache_spectrum/README.md for resume/reuse flags)
.venv/bin/python compare_video.py --model wan --modes base hybrid \
  --prompt_file vbench_prompts.txt --num_prompts 20 --seacache_thresh 0.2 \
  --save_base_frames --compile --attn cudnn --output_dir ./outputs_video_wan_d02
```

## Results (ours, B200; baselines SeaCache-published at matched TFLOPs)

FLUX.1-dev DrawBench-200: δ0.3 → **27.97 dB** (+1.68) @1241 TFLOPs;
δ0.6 → **21.63 dB** (+0.30) @774 TFLOPs. Examples:
`seacache_spectrum/examples_flux/`. Video Tab.2 finalizing (Wan base running).
