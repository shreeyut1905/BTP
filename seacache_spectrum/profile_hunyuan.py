#!/usr/bin/env python3
"""Profile HunyuanVideo 65f/50-step: stage breakdown + mask cost. Background use only."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from diffusers import HunyuanVideoPipeline
from hunyuan_hybrid_forward import cached_hunyuan_forward, reset_hunyuan_cache_state
from types import SimpleNamespace

BACKENDS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

pipe = HunyuanVideoPipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo",
                                            torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.transformer.__class__.forward = cached_hunyuan_forward
pipe.transformer.scheduler = pipe.scheduler
tr = pipe.transformer
args = SimpleNamespace(seacache_thresh=100.0, spectrum_m=4, spectrum_k=100,
                       spectrum_lam=0.1, spectrum_w=0.5, spectrum_taylor_order=1,
                       spectrum_min_cheb_obs=4)

# 1. text encode cost
t0 = time.perf_counter()
gen = torch.Generator(device="cuda").manual_seed(42)
with sdpa_kernel(BACKENDS):
    out = pipe(prompt="In a still frame, a stop sign", height=544, width=960,
               num_frames=65, num_inference_steps=5, guidance_scale=6.0,
               generator=gen, output_type="latent").frames
print(f"5-step-latent wall={time.perf_counter()-t0:.1f}s", flush=True)

# 2. single transformer step cost, mask vs no-mask
reset_hunyuan_cache_state(tr, 50, "base", args)
tr.scheduler.set_timesteps(50)
device = next(tr.parameters()).device
B, C, T, H, W = 1, 16, 17, 68, 120
hs = torch.randn(B, C, T, H, W, dtype=torch.bfloat16, device=device)
ts = tr.scheduler.timesteps[:1].expand(B)
enc = torch.randn(B, 256, 4096, dtype=torch.bfloat16, device=device)
emask = torch.ones(B, 256, dtype=torch.long, device=device)
pooled = torch.randn(B, 2048, dtype=torch.bfloat16, device=device)
guid = torch.tensor([6000.0] * B, dtype=torch.bfloat16, device=device)
torch.cuda.synchronize()
with sdpa_kernel(BACKENDS):
    for _ in range(3):
        tr(hs, ts, enc, emask, pooled, guid, return_dict=False)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(5):
        tr(hs, ts, enc, emask, pooled, guid, return_dict=False)
    torch.cuda.synchronize()
    print(f"one fwd (masked) avg={(time.perf_counter()-t0)/5:.2f}s", flush=True)

# 3. VAE decode cost
lat = torch.randn(1, 16, 17, 68, 120, dtype=torch.bfloat16, device=device)
torch.cuda.synchronize(); t0 = time.perf_counter()
with torch.no_grad():
    pipe.vae.decode(lat).sample
torch.cuda.synchronize()
print(f"vae-decode-65f wall={time.perf_counter()-t0:.1f}s", flush=True)
print("PROFILE_DONE")
