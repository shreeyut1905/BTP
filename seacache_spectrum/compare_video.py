#!/usr/bin/env python3
"""Video hybrid eval: base vs SeaCache-hybrid on Wan2.1-1.3B / HunyuanVideo.

One video per prompt (single-sample batches; video memory is heavy).
Metrics per prompt vs base video (same seed): mean over frames of
PSNR / SSIM / LPIPS + latency + TFLOPs (scaled from paper reference).
Videos saved as MP4; no per-frame PNGs kept.

SeaCache paper protocol: VBench prompts, 480p, 65 frames, 50 steps.
Thresholds: Wan hybrid d={0.2,0.35}; Hunyuan hybrid d={0.19,0.35}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_psnr import image_to_float, lpips_score, psnr, read_prompts, safe_filename, ssim_score

TFLOPS_REF = {"wan": 8214.0, "hunyuan": 14038.0}


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def frame_metrics(ref: Image.Image, pred: Image.Image, device: str, do_lpips: bool):
    out = {"psnr": psnr(ref, pred), "ssim": ssim_score(ref, pred)}
    out["lpips"] = None
    if do_lpips:
        try:
            out["lpips"] = lpips_score(ref, pred, device)
        except Exception as e:
            print(f"LPIPS failed: {e}")
    return out


def as_pil(f):
    if isinstance(f, Image.Image):
        return f.convert("RGB")
    a = np.asarray(f)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(a).convert("RGB")


def save_mp4(frames, path: str, fps: int = 16):
    arr = [np.asarray(as_pil(f)) for f in frames]
    imageio.mimsave(path, arr, fps=fps, codec="libx264", quality=8)


def load_pipe(model: str, compile: bool = False):
    if model == "wan":
        from diffusers import WanPipeline
        from wan_hybrid_forward import (
            cached_wan_forward, reset_wan_cache_state, wan_cache_totals,
        )
        pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                                           torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.transformer.__class__.forward = cached_wan_forward
        pipe.transformer.scheduler = pipe.scheduler
        if compile:
            for i, b in enumerate(pipe.transformer.blocks):
                pipe.transformer.blocks[i] = torch.compile(b, fullgraph=False)
        fwd = (cached_wan_forward, reset_wan_cache_state, wan_cache_totals)
    else:
        from diffusers import HunyuanVideoPipeline
        from hunyuan_hybrid_forward import (
            cached_hunyuan_forward, reset_hunyuan_cache_state, hunyuan_cache_totals,
        )
        pipe = HunyuanVideoPipeline.from_pretrained("hunyuanvideo-community/HunyuanVideo",
                                                    torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        pipe.transformer.__class__.forward = cached_hunyuan_forward
        pipe.transformer.scheduler = pipe.scheduler
        if compile:
            for i, b in enumerate(pipe.transformer.transformer_blocks):
                pipe.transformer.transformer_blocks[i] = torch.compile(b, fullgraph=False)
            for i, b in enumerate(pipe.transformer.single_transformer_blocks):
                pipe.transformer.single_transformer_blocks[i] = torch.compile(b, fullgraph=False)
        fwd = (cached_hunyuan_forward, reset_hunyuan_cache_state, hunyuan_cache_totals)
    return pipe, fwd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["wan", "hunyuan"], required=True)
    p.add_argument("--modes", nargs="+", default=["base", "hybrid"])
    p.add_argument("--prompt_file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "vbench_prompts.txt"))
    p.add_argument("--num_prompts", type=int, default=20)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num_frames", type=int, default=65)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seacache_thresh", type=float, default=0.2)
    p.add_argument("--spectrum_m", type=int, default=4)
    p.add_argument("--spectrum_k", type=int, default=100)
    p.add_argument("--spectrum_lam", type=float, default=0.1)
    p.add_argument("--spectrum_w", type=float, default=0.5)
    p.add_argument("--spectrum_taylor_order", type=int, default=1)
    p.add_argument("--spectrum_min_cheb_obs", type=int, default=4)
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile transformer blocks for speed")
    p.add_argument("--attn", type=str, default="cudnn", choices=["cudnn", "flash", "auto"],
                   help="SDPA backend priority (cudnn handles masked Hunyuan attn fastest on B200)")
    p.add_argument("--reuse_base", type=str, default=None,
                   help="dir of a previous run: load base frames from its videos/frames_base_* PNGs instead of generating")
    p.add_argument("--save_base_frames", action="store_true",
                   help="save base frames as lossless PNGs for reuse by other threshold runs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    vid_dir = os.path.join(args.output_dir, "videos")
    os.makedirs(vid_dir, exist_ok=True)
    height = args.height or (480 if args.model == "wan" else 544)
    width = args.width or (832 if args.model == "wan" else 960)
    guidance = args.guidance or (5.0 if args.model == "wan" else 6.0)
    prompts = read_prompts(args.prompt_file)[args.offset:args.offset + args.num_prompts]
    n = len(prompts)
    seeds = [args.seed + args.offset + i for i in range(n)]
    slugs = [f"{args.offset + i:04d}-{safe_filename(p)[:50]}" for i, p in enumerate(prompts)]

    device = "cuda"
    print(f"[{now_str()}] Loading {args.model} (this takes minutes)...", flush=True)
    pipe, (_, reset_fn, totals_fn) = load_pipe(args.model, compile=args.compile)
    tr = pipe.transformer
    ns = SimpleNamespace(seacache_thresh=args.seacache_thresh, spectrum_m=args.spectrum_m,
                         spectrum_k=args.spectrum_k, spectrum_lam=args.spectrum_lam,
                         spectrum_w=args.spectrum_w,
                         spectrum_taylor_order=args.spectrum_taylor_order,
                         spectrum_min_cheb_obs=args.spectrum_min_cheb_obs)

    from torch.nn.attention import sdpa_kernel, SDPBackend
    _BACKENDS = {
        "cudnn": [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                  SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
        "flash": [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                  SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
        "auto": [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                 SDPBackend.MATH],
    }
    attn_backends = _BACKENDS[args.attn]
    all_frames, walls, totals = {}, {}, {}
    if args.reuse_base and "base" in args.modes:
        args.modes = [m for m in args.modes if m != "base"]
        print(f"[{now_str()}] Will load base frames from {args.reuse_base}", flush=True)
    base_frame_dir = os.path.join(args.reuse_base, "videos") if args.reuse_base else vid_dir
    for mode in args.modes:
        frames_list, wall, comp = [], 0.0, 0
        skip = 0
        for i, prompt in enumerate(prompts):
            reset_fn(tr, args.num_inference_steps, mode, ns)
            gen = torch.Generator(device="cuda").manual_seed(seeds[i])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with sdpa_kernel(attn_backends):
                frames = pipe(prompt=prompt, height=height, width=width,
                              num_frames=args.num_frames,
                              num_inference_steps=args.num_inference_steps,
                              guidance_scale=guidance, generator=gen).frames[0]
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            wall += dt
            t = totals_fn(tr)
            comp += t["computed"]
            skip += t["skipped"]
            frames_list.append(frames)
            save_mp4(frames, os.path.join(vid_dir, f"{slugs[i]}_{mode}.mp4"), args.fps)
            if mode == "base" and args.save_base_frames:
                for f, fr in enumerate(frames):
                    as_pil(fr).save(os.path.join(vid_dir, f"{slugs[i]}_base_f{f:03d}.png"))
            print(f"[{now_str()}] {mode} [{i + 1}/{n}] {dt:.1f}s fwd={t['computed']} skip={t['skipped']}",
                  flush=True)
            torch.cuda.empty_cache()
        all_frames[mode] = frames_list
        walls[mode] = wall
        totals[mode] = {"computed": comp, "skipped": skip}

    if args.reuse_base:
        with open(os.path.join(args.reuse_base, "comparison.json")) as f:
            prior = json.load(f)["summary"]
        import glob as _glob
        base_frames = []
        for i in range(n):
            fpaths = sorted(_glob.glob(os.path.join(base_frame_dir, f"{slugs[i]}_base_f*.png")))
            if not fpaths:
                raise FileNotFoundError(f"no reused base frames for {slugs[i]}")
            base_frames.append([Image.open(p).convert("RGB") for p in fpaths])
        all_frames["base"] = base_frames
        walls["base"] = float(prior["base"]["total_wall"])
        totals["base"] = {"computed": int(prior["base"]["computed"]),
                          "skipped": int(prior["base"]["skipped"])}
        if "base" not in args.modes:
            args.modes = args.modes + ["base"]

    rows = []
    if "base" in all_frames:
        for m in args.modes:
            if m == "base":
                continue
            for i in range(n):
                pm = [frame_metrics(as_pil(b), as_pil(h), device, not args.no_lpips)
                      for b, h in zip(all_frames["base"][i], all_frames[m][i])]
                rows.append({
                    "idx": args.offset + i, "prompt": prompts[i], "seed": seeds[i], "mode": m,
                    "psnr": float(np.mean([x["psnr"] for x in pm])),
                    "ssim": float(np.mean([x["ssim"] for x in pm])),
                    "lpips": float(np.mean([x["lpips"] for x in pm if x["lpips"] is not None]))
                    if any(x["lpips"] is not None for x in pm) else None,
                })
                print(f"  [{i}] {m} PSNR={rows[-1]['psnr']:.2f} SSIM={rows[-1]['ssim']:.4f} "
                      f"LPIPS={rows[-1]['lpips']}", flush=True)

    summary = {"model": args.model, "height": height, "width": width,
               "num_frames": args.num_frames, "steps": args.num_inference_steps,
               "guidance": guidance, "num_prompts": n, "offset": args.offset,
               "seacache_thresh": args.seacache_thresh,
               "tflops_ref": TFLOPS_REF[args.model]}
    for m in args.modes:
        per_vid_calls = totals[m]["computed"] / n
        full_calls = (args.num_inference_steps
                      * (2 if args.model == "wan" and guidance > 1 else 1))
        summary[m] = {"latency": walls[m] / n, "total_wall": walls[m],
                      "computed": totals[m]["computed"], "skipped": totals[m]["skipped"],
                      "tflops": TFLOPS_REF[args.model] * per_vid_calls / full_calls,
                      "theoretical_speedup": full_calls / max(1, per_vid_calls)}
        if m != "base":
            mr = [r for r in rows if r["mode"] == m]
            summary[m].update({
                "mean_psnr": float(np.mean([r["psnr"] for r in mr])),
                "mean_ssim": float(np.mean([r["ssim"] for r in mr])),
                "mean_lpips": float(np.mean([r["lpips"] for r in mr if r["lpips"] is not None]))})
    with open(os.path.join(args.output_dir, "comparison.json"), "w") as f:
        json.dump({"summary": summary,
                   "rows": [{k: v for k, v in r.items() if k != "prompt"} | {"prompt": r["prompt"]}
                             for r in rows]}, f, indent=2)
    print("SUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
