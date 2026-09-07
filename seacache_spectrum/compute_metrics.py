#!/usr/bin/env python3
"""Compute PSNR/SSIM/LPIPS of hybrid images vs base reference (no GPU model load).

Expects:
  base_dir/images/{slug}_base.png
  hybrid_dir/images/{slug}_hybrid.png
with identical prompt_file / num_prompts / ordering (slug = idx + sanitized prompt).

Merges with hybrid-only comparison.json (latency/TFLOPs) if present and writes
comparison_vs_base.json + .md into hybrid_dir.
Run with the project venv: ./.venv/bin/python compute_metrics.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_psnr import image_to_float, lpips_score, psnr, read_prompts, safe_filename, ssim_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)
    p.add_argument("--hybrid_dir", required=True)
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--num_prompts", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_lpips", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    prompts = read_prompts(args.prompt_file)[: args.num_prompts]
    n = len(prompts)
    slugs = [f"{i:02d}-{safe_filename(p)[:60]}" for i, p in enumerate(prompts)]
    base_img_dir = os.path.join(args.base_dir, "images")
    hyb_img_dir = os.path.join(args.hybrid_dir, "images")

    # latency/TFLOPs from hybrid-only run
    prior = {}
    prior_path = os.path.join(args.hybrid_dir, "comparison.json")
    if os.path.exists(prior_path):
        with open(prior_path) as f:
            prior = json.load(f).get("summary", {})

    rows = []
    for i, prompt in enumerate(prompts):
        b_path = os.path.join(base_img_dir, f"{slugs[i]}_base.png")
        h_path = os.path.join(hyb_img_dir, f"{slugs[i]}_hybrid.png")
        if not os.path.exists(b_path):
            raise FileNotFoundError(f"missing base image: {b_path}")
        if not os.path.exists(h_path):
            raise FileNotFoundError(f"missing hybrid image: {h_path}")
        base = Image.open(b_path).convert("RGB")
        hyb = Image.open(h_path).convert("RGB")
        row = {"idx": i, "prompt": prompt, "seed": args.seed + i,
               "psnr_hybrid": psnr(base, hyb),
               "ssim_hybrid": ssim_score(base, hyb),
               "l1_hybrid": float(np.mean(np.abs(image_to_float(base) - image_to_float(hyb))))}
        if not args.no_lpips:
            try:
                row["lpips_hybrid"] = lpips_score(base, hyb, "cpu")
            except Exception as e:
                print(f"[{i}] LPIPS failed: {e}")
                row["lpips_hybrid"] = None
        rows.append(row)
        if i % 20 == 0:
            print(f"[{i}/{n}] PSNR={row['psnr_hybrid']:.2f} SSIM={row['ssim_hybrid']:.4f} "
                  f"LPIPS={row.get('lpips_hybrid')}", flush=True)

    summary = {
        "num_prompts": n,
        "base_dir": args.base_dir,
        "hybrid_dir": args.hybrid_dir,
        "mean_psnr_hybrid": float(np.mean([r["psnr_hybrid"] for r in rows])),
        "mean_ssim_hybrid": float(np.mean([r["ssim_hybrid"] for r in rows])),
        "mean_l1_hybrid": float(np.mean([r["l1_hybrid"] for r in rows])),
        "mean_time_hybrid": prior.get("mean_time_hybrid"),
        "tflops_hybrid": prior.get("tflops_hybrid"),
        "theoretical_speedup_hybrid": prior.get("theoretical_speedup_hybrid"),
        "computed_hybrid": prior.get("computed_hybrid"),
        "skipped_hybrid": prior.get("skipped_hybrid"),
        "refresh_ratio_hybrid": prior.get("refresh_ratio_hybrid"),
        "seacache_thresh": prior.get("seacache_thresh"),
    }
    lp = [r["lpips_hybrid"] for r in rows if r.get("lpips_hybrid") is not None]
    summary["mean_lpips_hybrid"] = float(np.mean(lp)) if lp else None

    out_json = os.path.join(args.hybrid_dir, "comparison_vs_base.json")
    out_md = os.path.join(args.hybrid_dir, "comparison_vs_base.md")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    with open(out_md, "w") as f:
        f.write("# Hybrid vs base (FLUX.1-dev 1024x1024, 50 steps)\n\n")
        f.write(f"- prompts: {n}\n- base: `{args.base_dir}`\n- hybrid: `{args.hybrid_dir}`\n")
        f.write(f"- thresh δ={summary.get('seacache_thresh')}\n\n")
        f.write(f"- Mean PSNR: **{summary['mean_psnr_hybrid']:.3f} dB**\n")
        f.write(f"- Mean SSIM: **{summary['mean_ssim_hybrid']:.4f}**\n")
        f.write(f"- Mean LPIPS: **{summary['mean_lpips_hybrid']}**\n")
        f.write(f"- Latency: **{summary['mean_time_hybrid']} s/img** (B200)\n")
        f.write(f"- TFLOPs: **{summary['tflops_hybrid']}** (scaled from 2976 ref)\n")
        f.write(f"- Theoretical speedup: **{summary['theoretical_speedup_hybrid']}x**\n")
    print("Wrote", out_json)
    print("Wrote", out_md)
    print(f"SUMMARY PSNR={summary['mean_psnr_hybrid']:.3f} SSIM={summary['mean_ssim_hybrid']:.4f} "
          f"LPIPS={summary['mean_lpips_hybrid']} time={summary['mean_time_hybrid']} "
          f"TFLOPs={summary['tflops_hybrid']}")


if __name__ == "__main__":
    main()
