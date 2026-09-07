# `src/hunyuan/` — HunyuanVideo patch (diffusers path)

`hunyuan_forward.py` replaces the HunyuanVideo transformer forward with
`cached_hunyuan_forward`. Same structure as the Wan patch (CFG demux by
timestep, `hunyuan_cache_totals` for accounting); reset with
`reset_hunyuan_cache_state` once per generation.

Minimal inference (run from `seacache_spectrum/`):

```python
import sys, torch; sys.path.insert(0, "src")
from argparse import Namespace
from diffusers import HunyuanVideoPipeline
from hunyuan.hunyuan_forward import cached_hunyuan_forward, reset_hunyuan_cache_state

pipe = HunyuanVideoPipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo",
                                            torch_dtype=torch.bfloat16).to("cuda")
# keep it on CUDA: cpu-offload bypasses the patched forward
pipe.transformer.__class__.forward = cached_hunyuan_forward

args = Namespace(seacache_thresh=0.19,
                 spectrum_m=4, spectrum_k=100, spectrum_lam=0.1,
                 spectrum_w=0.5, spectrum_taylor_order=1,
                 spectrum_min_cheb_obs=4)
reset_hunyuan_cache_state(pipe.transformer, num_steps=50, mode="hybrid", args=args)
out = pipe("A panda surfing.", num_frames=65, num_inference_steps=50)
```

Eval uses the same video script with `--model hunyuan`
(TFLOPs ref 14038):

```bash
.venv/bin/python scripts/eval_video.py --model hunyuan --modes base hybrid \
  --prompt_file prompts/vbench946.txt --num_prompts 946 \
  --num_inference_steps 50 --seacache_thresh 0.19 --save_base_frames \
  --compile --attn cudnn --output_dir ./outputs_hunyuan_480p_d019
```

Status: Hunyuan runs have not started yet — the SeaCache-reported targets
are 32.39 dB @6747 TFLOPs (δ0.19) and 26.46 dB @4598 TFLOPs (δ0.35).
