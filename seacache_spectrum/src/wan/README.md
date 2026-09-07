# `src/wan/` — Wan2.1-1.3B patch

`wan_forward.py` replaces the Wan transformer forward with
`cached_wan_forward`. Two Wan-specific details: with classifier-free
guidance the cond/uncond streams are demuxed by timestep equality, and
`wan_cache_totals(transformer)` reports `{computed, skipped}` for TFLOPs
accounting. Call `reset_wan_cache_state` once per generation.

Minimal inference (run from `seacache_spectrum/`):

```python
import sys, torch; sys.path.insert(0, "src")
from argparse import Namespace
from diffusers import WanPipeline
from wan.wan_forward import cached_wan_forward, reset_wan_cache_state

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                                   torch_dtype=torch.bfloat16).to("cuda")
# keep it on CUDA: cpu-offload bypasses the patched forward
pipe.transformer.__class__.forward = cached_wan_forward

args = Namespace(seacache_thresh=0.2,        # δ: 0.2 quality, 0.35 fast
                 spectrum_m=4, spectrum_k=100, spectrum_lam=0.1,
                 spectrum_w=0.5, spectrum_taylor_order=1,
                 spectrum_min_cheb_obs=4)
reset_wan_cache_state(pipe.transformer, num_steps=50, mode="hybrid", args=args)
out = pipe("A cat playing piano.", num_frames=65, num_inference_steps=50)
```

Full eval on VBench-946 — run base once (saving frames), then each delta
config reuses it; resume interrupted runs with `--offset`:

```bash
# base reference (long!): saves per-frame PNGs for reuse
.venv/bin/python scripts/eval_video.py --model wan --modes base hybrid \
  --prompt_file prompts/vbench946.txt --num_prompts 946 \
  --num_inference_steps 50 --seacache_thresh 0.2 --save_base_frames \
  --compile --attn cudnn --output_dir ./outputs_video_wan_d02
# second delta reuses the same base (no regeneration):
.venv/bin/python scripts/eval_video.py --model wan --modes hybrid \
  --prompt_file prompts/vbench946.txt --num_prompts 946 \
  --num_inference_steps 50 --seacache_thresh 0.35 \
  --reuse_base ./outputs_video_wan_d02 --compile --attn cudnn \
  --output_dir ./outputs_video_wan_d035
# resume a killed run at prompt 822 (never redo finished prompts):
.venv/bin/python scripts/eval_video.py --model wan --modes hybrid \
  --prompt_file prompts/vbench946.txt --offset 822 --num_prompts 124 \
  --num_inference_steps 50 --seacache_thresh 0.35 \
  --reuse_base ./outputs_video_wan_d02 --compile --attn cudnn \
  --output_dir ./outputs_video_wan_d035_r124
```

Reference numbers (B200, full 946): δ0.2 → 28.38 dB / 0.905 / 0.066
@4303 TFLOPs; δ0.35 → 26.70 dB / 0.884 / 0.084 @3336 TFLOPs (ref 8214).
Rows scored off MP4-decoded hybrids carry a ~0.1 dB codec caveat.
