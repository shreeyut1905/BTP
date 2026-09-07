# ReSPect (SeaCache schedule + Spectrum forecast)

Training-free diffusion cache. **SeaCache** decides *when* to skip
(SEA-filtered first-block modulation, accumulated relative-L1 vs δ);
**Spectrum** decides *what* to put on a skip — a forecast of the transformer
**residual** (Chebyshev ridge `M=4`, `λ=0.1`, `K=100`, blended `w=0.5` with a
first-order discrete Taylor step, Taylor-only warm-up until `min_cheb_obs=4`
observations). Modes: `base` (full compute) · `seacache` (gate + copy-reuse) ·
`hybrid` (gate + forecast).

See [research.md](research.md) for the method and ablations.

## Layout

```
spectrum_forecaster.py   shared Chebyshev+Taylor residual forecaster
util_seacache.py         shared SEA Wiener filter
flux_forward.py          FLUX.1-dev patch (cached_flux_forward)
wan_hybrid_forward.py    Wan2.1-1.3B patch (cached_wan_forward)
hunyuan_hybrid_forward.py HunyuanVideo patch, diffusers path (cached_hunyuan_forward)
compare_psnr.py          FLUX eval: base vs hybrid PSNR/SSIM/LPIPS
compare_video.py         video eval: base vs hybrid, MP4s + frame metrics
compute_metrics.py       image metric-only pass (no GPU model load)
prompts.txt              4-prompt FLUX smoke list
vbench_prompts.txt       946-prompt VBench list (video)
examples_flux/           base-vs-ours strips (δ=0.3)
```

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/huggingface-cli login   # FLUX.1-dev is gated
```

## Inference

```python
from diffusers import DiffusionPipeline
from flux_forward import cached_flux_forward, reset_cache_state
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                         torch_dtype=torch.bfloat16).to("cuda")
pipe.transformer.__class__.forward = cached_flux_forward
reset_cache_state(pipe.transformer, num_steps=50, mode="hybrid", args)
# args: seacache_thresh=0.3, spectrum_m=4, spectrum_k=100,
#       spectrum_lam=0.1, spectrum_w=0.5, spectrum_taylor_order=1,
#       spectrum_min_cheb_obs=4
```

Wan2.1 is the same shape (`WanPipeline`, `cached_wan_forward`,
`reset_wan_cache_state`); keep `pipe.to("cuda")` — cpu-offload bypasses the
patched forward. CFG streams are demuxed by timestep equality.

## Eval

```bash
# FLUX smoke (4 prompts) then full DrawBench-200
.venv/bin/python compare_psnr.py --num_prompts 4 --seacache_thresh 0.3 --output_dir ./outputs
.venv/bin/python compare_psnr.py --num_prompts 200 --seacache_thresh 0.3 --output_dir ./outputs_hybrid_d03
.venv/bin/python compute_metrics.py --base_dir ./outputs_base --hybrid_dir ./outputs_hybrid_d03

# Video: base (+save frames for reuse), then hybrid delta(s) off the same base
.venv/bin/python compare_video.py --model wan --modes base hybrid \
  --prompt_file vbench_prompts.txt --num_prompts 20 --num_frames 65 \
  --num_inference_steps 50 --seacache_thresh 0.2 --save_base_frames \
  --compile --attn cudnn --output_dir ./outputs_video_wan_d02
.venv/bin/python compare_video.py --model wan --modes hybrid \
  --prompt_file vbench_prompts.txt --num_prompts 20 --num_frames 65 \
  --num_inference_steps 50 --seacache_thresh 0.35 \
  --reuse_base ./outputs_video_wan_d02 --compile --attn cudnn \
  --output_dir ./outputs_video_wan_d035
# Resume an interrupted run without redoing finished prompts:
.venv/bin/python compare_video.py --model wan --modes base \
  --prompt_file vbench_prompts.txt --offset 824 --num_prompts 122 \
  --save_base_frames --compile --attn cudnn --output_dir ./outputs_video_wan_d02_resume
```

Notes: fresh `sdpa_kernel(...)` context per generation; `torch.compile`
per-block with `dynamic=False`; never compare seconds across hardware —
match TFLOPs (FLUX ref 2976, Wan ref 8214, Hunyuan ref 14038).

## Results (ours, B200; baselines SeaCache-published at matched TFLOPs)

| Config | PSNR↑ | SSIM↑ | LPIPS↓ | TFLOPs | s/img |
|---|---|---|---|---|---|
| FLUX δ=0.3 | 27.969 | 0.9182 | 0.0699 | 1241 | 3.33 |
| FLUX δ=0.6 | 21.630 | 0.8225 | 0.1781 | 774 | 2.14 |

δ0.3: 27.97 vs 26.29 (**+1.68 dB**); δ0.6: 21.63 vs 21.33 (+0.30 dB @774).
Video Tab.2 finalizing.

## Qualitative examples (FLUX base vs ours, δ=0.3)

![flux 143 base vs ours](examples_flux/flux_143_base_vs_ours.png)
*143 — baby fennec macro, PSNR 33.79, SSIM 0.971.*

![flux 156 base vs ours](examples_flux/flux_156_base_vs_ours.png)
*156 — Greek statue, PSNR 32.15, SSIM 0.981.*

![flux 093 base vs ours](examples_flux/flux_093_base_vs_ours.png)
*93 — robot, PSNR 32.43, SSIM 0.985.*
