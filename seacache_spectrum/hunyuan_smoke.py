#!/usr/bin/env python3
"""Smoke test: HunyuanVideo hybrid cache, 1 prompt, tiny video."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from diffusers import HunyuanVideoPipeline
from hunyuan_hybrid_forward import (
    cached_hunyuan_forward, reset_hunyuan_cache_state, hunyuan_cache_totals,
)
from types import SimpleNamespace

MODEL = "hunyuanvideo-community/HunyuanVideo"
PROMPT = "A cat wearing sunglasses at a pool."

def main():
    pipe = HunyuanVideoPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.transformer.__class__.forward = cached_hunyuan_forward
    pipe.transformer.scheduler = pipe.scheduler
    args = SimpleNamespace(seacache_thresh=0.19, spectrum_m=4, spectrum_k=100,
                           spectrum_lam=0.1, spectrum_w=0.5, spectrum_taylor_order=1,
                           spectrum_min_cheb_obs=4)
    for mode in ("base", "hybrid"):
        reset_hunyuan_cache_state(pipe.transformer, 10, mode, args)
        gen = torch.Generator(device="cuda").manual_seed(0)
        t0 = time.perf_counter()
        out = pipe(prompt=PROMPT, height=320, width=544, num_frames=13,
                   num_inference_steps=10, guidance_scale=6.0, generator=gen).frames[0]
        dt = time.perf_counter() - t0
        tot = hunyuan_cache_totals(pipe.transformer)
        print(f"{mode}: {len(out)} frames wall={dt:.1f}s totals={tot}", flush=True)
    print("SMOKE_OK")

if __name__ == "__main__":
    main()
