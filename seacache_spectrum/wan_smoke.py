#!/usr/bin/env python3
"""Smoke test: Wan2.1-1.3B hybrid cache, 1 prompt, short video."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from diffusers import WanPipeline
from wan_hybrid_forward import cached_wan_forward, reset_wan_cache_state, wan_cache_totals
from types import SimpleNamespace

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
PROMPT = "Two anthropomorphic cats in comfy boxing gear fight on a spotlighted stage."

def main():
    pipe = WanPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.transformer.__class__.forward = cached_wan_forward
    pipe.transformer.scheduler = pipe.scheduler
    args = SimpleNamespace(seacache_thresh=0.2, spectrum_m=4, spectrum_k=100,
                           spectrum_lam=0.1, spectrum_w=0.5, spectrum_taylor_order=1,
                           spectrum_min_cheb_obs=4)
    for mode in ("base", "hybrid"):
        reset_wan_cache_state(pipe.transformer, 20, mode, args)
        gen = torch.Generator(device="cuda").manual_seed(0)
        t0 = time.perf_counter()
        out = pipe(prompt=PROMPT, height=480, width=832, num_frames=33,
                   num_inference_steps=20, guidance_scale=5.0, generator=gen).frames[0]
        dt = time.perf_counter() - t0
        tot = wan_cache_totals(pipe.transformer)
        print(f"{mode}: {len(out)} frames {out[0].size} wall={dt:.1f}s totals={tot}", flush=True)
    print("SMOKE_OK")

if __name__ == "__main__":
    main()
