#!/usr/bin/env python3
"""Latency / measured-FLOPs / exact-NFE benchmark on a single B200.

Protocol per config: 1 warm-up generation (discarded), 1 timed generation
(no instrumentation) for latency + exact NFE, 1 instrumented generation
under FlopCounterMode for measured FLOPs. torch.compile is NOT used.

Resolutions match the quality runs: FLUX 1024x1024 / 50 steps,
Wan + Hunyuan 480x832 / 65 frames / 50 steps.

Usage: bench_latency.py --model {flux,wan,hunyuan} [--out FILE]
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.flop_counter import FlopCounterMode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

PROMPT = "A cat walks on the grass, realistic style."
BACKENDS = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def sp(thresh):
    return dict(seacache_thresh=thresh, spectrum_m=4, spectrum_k=100,
                spectrum_lam=0.1, spectrum_w=0.5, spectrum_taylor_order=1,
                spectrum_min_cheb_obs=4)


def build(model):
    """Return (pipe, reset_fn, totals_fn, gen_fn, configs)."""
    if model == "flux":
        from diffusers import DiffusionPipeline, FluxTransformer2DModel
        from flux.flux_forward import cached_flux_forward, reset_cache_state
        FluxTransformer2DModel.forward = cached_flux_forward
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.transformer.__class__.forward = cached_flux_forward
        pipe.transformer.scheduler = pipe.scheduler

        def totals(tr):
            return {"computed": int(tr.computed_steps),
                    "skipped": int(tr.skipped_steps)}

        def gen(steps, g):
            return pipe(prompt=PROMPT, height=1024, width=1024,
                        num_inference_steps=steps, guidance_scale=3.5,
                        generator=g)

        cfgs = [("Original (50 steps)", "base", 50, None),
                ("Vanilla 25 steps", "base", 25, None),
                ("Vanilla 15 steps", "base", 15, None),
                ("SeaCache d=0.3", "seacache", 50, 0.3),
                ("SeaCache d=0.6", "seacache", 50, 0.6),
                ("ReSPect d=0.3", "hybrid", 50, 0.3),
                ("ReSPect d=0.6", "hybrid", 50, 0.6)]
        return pipe, reset_cache_state, totals, gen, cfgs

    if model == "wan":
        from diffusers import WanPipeline
        from wan.wan_forward import (cached_wan_forward, reset_wan_cache_state,
                                     wan_cache_totals)
        pipe = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.transformer.__class__.forward = cached_wan_forward
        pipe.transformer.scheduler = pipe.scheduler

        def gen(steps, g):
            return pipe(prompt=PROMPT, height=480, width=832, num_frames=65,
                        num_inference_steps=steps, guidance_scale=5.0,
                        generator=g)

        cfgs = [("Original (50 steps)", "base", 50, None),
                ("Vanilla 25 steps", "base", 25, None),
                ("Vanilla 15 steps", "base", 15, None),
                ("SeaCache d=0.2", "seacache", 50, 0.2),
                ("SeaCache d=0.35", "seacache", 50, 0.35),
                ("ReSPect d=0.17", "hybrid", 50, 0.17),
                ("ReSPect d=0.31", "hybrid", 50, 0.31),
                ("SeaCache d=0.17", "seacache", 50, 0.17),
                ("SeaCache d=0.31", "seacache", 50, 0.31)]
        return pipe, reset_wan_cache_state, wan_cache_totals, gen, cfgs

    from diffusers import HunyuanVideoPipeline
    from hunyuan.hunyuan_forward import (cached_hunyuan_forward,
                                         reset_hunyuan_cache_state,
                                         hunyuan_cache_totals)
    pipe = HunyuanVideoPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.transformer.__class__.forward = cached_hunyuan_forward
    pipe.transformer.scheduler = pipe.scheduler

    def gen(steps, g):
        return pipe(prompt=PROMPT, height=480, width=832, num_frames=65,
                    num_inference_steps=steps, guidance_scale=6.0, generator=g)

    cfgs = [("Original (50 steps)", "base", 50, None),
            ("Vanilla 25 steps", "base", 25, None),
            ("Vanilla 15 steps", "base", 15, None),
            ("SeaCache d=0.19", "seacache", 50, 0.19),
            ("SeaCache d=0.35", "seacache", 50, 0.35),
            ("ReSPect d=0.27", "hybrid", 50, 0.27),
            ("ReSPect d=0.48", "hybrid", 50, 0.48),
            ("SeaCache d=0.27", "seacache", 50, 0.27),
            ("SeaCache d=0.48", "seacache", 50, 0.48)]
    return pipe, reset_hunyuan_cache_state, hunyuan_cache_totals, gen, cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["flux", "wan", "hunyuan"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated substrings; run matching configs only")
    args = ap.parse_args()

    pipe, reset_fn, totals_fn, gen, cfgs = build(args.model)
    if args.only:
        pats = [t.strip() for t in args.only.split(",") if t.strip()]
        cfgs = [c for c in cfgs if any(t in c[0] for t in pats)]
    tr = pipe.transformer
    out = []

    for name, mode, steps, thresh in cfgs:
        ns = NS(**sp(thresh if thresh is not None else 0.2))
        # FLUX's reset_cache_state does not set the gate threshold (eval_flux.py
        # assigns it on the transformer); set it here for all models.
        tr.seacache_thresh = float(thresh if thresh is not None else 0.2)

        # ---- warm-up (discarded) ----
        reset_fn(tr, steps, mode, ns)
        with sdpa_kernel(BACKENDS):
            gen(steps, torch.Generator("cuda").manual_seed(0))
        torch.cuda.synchronize()

        # ---- timed, uninstrumented ----
        reset_fn(tr, steps, mode, ns)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with sdpa_kernel(BACKENDS):
            gen(steps, torch.Generator("cuda").manual_seed(0))
        torch.cuda.synchronize()
        lat = time.perf_counter() - t0
        nfe = totals_fn(tr)["computed"]

        # ---- measured FLOPs (transformer-only + whole pipeline) ----
        reset_fn(tr, steps, mode, ns)
        fc = FlopCounterMode(mods=[tr], display=False, depth=None)
        with fc:
            with sdpa_kernel(BACKENDS):
                gen(steps, torch.Generator("cuda").manual_seed(0))
        total = fc.get_total_flops()
        tr_name = type(tr).__name__
        tr_flops = fc.get_flop_counts().get(tr_name, {})
        tr_total = sum(tr_flops.values()) if tr_flops else None

        rec = {"config": name, "mode": mode, "steps": steps, "delta": thresh,
               "nfe": nfe, "latency_s": lat,
               "tflops_pipeline": total / 1e12,
               "tflops_transformer": (tr_total / 1e12) if tr_total else None}
        out.append(rec)
        print(f"{name:24s} NFE={nfe:4d} lat={lat:7.2f}s "
              f"TFLOPs(pipe)={rec['tflops_pipeline']:8.1f} "
              f"TFLOPs(xf)={rec['tflops_transformer'] or float('nan'):8.1f}",
              flush=True)
        torch.cuda.empty_cache()

    path = args.out or f"/tmp/bench_{args.model}.json"
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
