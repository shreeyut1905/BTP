#!/usr/bin/env python3
"""Compare SeaCache vs SeaCache+Spectrum PSNR vs FLUX.1-dev base at 1024x1024.

For each prompt we generate three images with the same seed:
  1. base           - full FLUX.1-dev (no cache)
  2. seacache       - original SeaCache residual reuse
  3. hybrid         - SeaCache SEA skip schedule + Spectrum residual forecast

PSNR is measured against the base image. Results are written to
``output_dir/comparison.json`` and ``output_dir/comparison.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.models import FluxTransformer2DModel
from PIL import Image

# Local modules live next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flux_forward import cached_flux_forward, reset_cache_state  # noqa: E402


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def safe_filename(name: str) -> str:
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^0-9A-Za-z._-]", "", name)
    return name or "img"


def image_to_float(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    return arr / 255.0


def psnr(ref: Image.Image, pred: Image.Image, eps: float = 1e-12) -> float:
    a = image_to_float(ref)
    b = image_to_float(pred)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    mse = np.mean((a - b) ** 2)
    if mse <= eps:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def l1_error(ref: Image.Image, pred: Image.Image) -> float:
    a = image_to_float(ref)
    b = image_to_float(pred)
    return float(np.mean(np.abs(a - b)))
def make_grid(images, labels, out_path: str, pad: int = 12) -> None:
    """Horizontal comparison strip with captions (base | seacache | hybrid)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6.4))
    if n == 1:
        axes = [axes]
    for ax, im, lab in zip(axes, images, labels):
        ax.imshow(im)
        ax.set_title(lab, fontsize=11)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_markdown(rows, summary, out_path: str) -> None:
    lines = [
        "# FLUX.1-dev PSNR comparison (1024x1024)",
        "",
        "PSNR is computed against the **base** (uncached) FLUX.1-dev image.",
        "",
        f"- Model: `{summary['model_id']}`",
        f"- Resolution: {summary['width']}x{summary['height']}",
        f"- Steps: {summary['num_inference_steps']}",
        f"- Guidance: {summary['guidance']}",
        f"- Seed: {summary['seed']}",
        f"- SeaCache threshold: {summary['seacache_thresh']}",
        f"- Spectrum: m={summary['spectrum_m']}, w={summary['spectrum_w']}, "
        f"lam={summary['spectrum_lam']}, taylor_order={summary['spectrum_taylor_order']}",
        "",
        "| # | Prompt | SeaCache PSNR | Hybrid PSNR | Delta (hybrid-seacache) | "
        "SeaCache sample-forwards | Hybrid sample-forwards | Base s/img | SeaCache s/img | Hybrid s/img |",
        "|---|--------|---------------|-------------|-------------------------|--------------------------|------------------------|------------|----------------|---------------|",
    ]
    for r in rows:
        prompt = r["prompt"].replace("|", "/")
        if len(prompt) > 70:
            prompt = prompt[:67] + "..."
        lines.append(
            f"| {r['idx']} | {prompt} | {r['psnr_seacache']:.3f} | {r['psnr_hybrid']:.3f} | "
            f"{r['psnr_hybrid'] - r['psnr_seacache']:+.3f} | "
            f"{r['seacache_computed']} | {r['hybrid_computed']} | "
            f"{r['time_base']:.2f} | {r['time_seacache']:.2f} | {r['time_hybrid']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Averages",
            "",
            f"- Mean SeaCache PSNR: **{summary['mean_psnr_seacache']:.3f} dB**",
            f"- Mean Hybrid (SeaCache+Spectrum) PSNR: **{summary['mean_psnr_hybrid']:.3f} dB**",
            f"- Mean PSNR delta (hybrid - seacache): **{summary['mean_psnr_delta']:+.3f} dB**",
            f"- Mean SeaCache speedup vs base: **{summary['mean_speedup_seacache']:.2f}x**",
            f"- Mean Hybrid speedup vs base: **{summary['mean_speedup_hybrid']:.2f}x**",
            "",
        ]
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_batch(pipe, prompts, seeds, height, width, steps, guidance, device):
    """Generate one image per prompt in a single pipeline call."""
    generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = pipe(
        prompt=list(prompts),
        num_inference_steps=int(steps),
        height=int(height),
        width=int(width),
        guidance_scale=float(guidance),
        max_sequence_length=512,
        num_images_per_prompt=1,
        generator=generators,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tr = pipe.transformer
    n = len(prompts)
    stats = {
        "computed": int(getattr(tr, "computed_steps", steps * n)),
        "skipped": int(getattr(tr, "skipped_steps", 0)),
        "time": float(elapsed),
        "batch_size": n,
    }
    return list(out.images), stats
def parse_args():
    p = argparse.ArgumentParser(description="SeaCache vs SeaCache+Spectrum PSNR on FLUX.1-dev")
    p.add_argument("--prompt_file", type=str, default=None)
    p.add_argument("--prompts", nargs="*", default=None)
    p.add_argument("--num_prompts", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="/workspace/BTP/seacache_spectrum/outputs")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--seacache_thresh", type=float, default=0.3)
    p.add_argument("--spectrum_m", type=int, default=4)
    p.add_argument("--spectrum_k", type=int, default=100)
    p.add_argument("--spectrum_lam", type=float, default=0.1)
    p.add_argument("--spectrum_w", type=float, default=0.5)
    p.add_argument("--spectrum_taylor_order", type=int, default=1)
    p.add_argument("--spectrum_min_cheb_obs", type=int, default=4)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--offload", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    if args.prompt_file:
        prompts = read_prompts(args.prompt_file)
    elif args.prompts:
        prompts = list(args.prompts)
    else:
        default_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.txt")
        prompts = read_prompts(default_file)
    prompts = prompts[: int(args.num_prompts)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    height = (args.height // 16) * 16
    width = (args.width // 16) * 16
    steps = int(args.num_inference_steps)

    print(f"[{now_str()}] Loading {args.model_id} on {device} ({torch_dtype})")
    FluxTransformer2DModel.forward = cached_flux_forward
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe.transformer.__class__.forward = cached_flux_forward

    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    tr = pipe.transformer
    tr.scheduler = pipe.scheduler
    tr.seacache_thresh = float(args.seacache_thresh)

    hybrid_ns = SimpleNamespace(
        spectrum_m=args.spectrum_m,
        spectrum_k=args.spectrum_k,
        spectrum_lam=args.spectrum_lam,
        spectrum_w=args.spectrum_w,
        spectrum_taylor_order=args.spectrum_taylor_order,
        spectrum_min_cheb_obs=args.spectrum_min_cheb_obs,
    )

    n = len(prompts)
    seeds = [int(args.seed) + i for i in range(n)]
    slugs = [f"{i:02d}-{safe_filename(p)[:60]}" for i, p in enumerate(prompts)]
    print(
        f"[{now_str()}] Start | steps={steps} | size={width}x{height} | "
        f"thresh={args.seacache_thresh} | prompts={n} (batched) | seed={args.seed} | "
        f"spectrum w={args.spectrum_w} k={args.spectrum_k} min_cheb={args.spectrum_min_cheb_obs}"
    )
    for i, prompt in enumerate(prompts):
        print(f"  [{i}] seed={seeds[i]}  {prompt}")

    mode_images = {}
    mode_stats = {}
    for mode in ("base", "seacache", "hybrid"):
        reset_cache_state(tr, steps, mode, hybrid_ns if mode == "hybrid" else args, batch_size=n)
        print(f"\n[{now_str()}] Generating batch mode={mode}  B={n}")
        imgs, st = generate_batch(
            pipe, prompts, seeds, height, width, steps, args.guidance, device
        )
        mode_images[mode] = imgs
        mode_stats[mode] = st
        for i, img in enumerate(imgs):
            out_fn = os.path.join(img_dir, f"{slugs[i]}_{mode}.png")
            img.save(out_fn)
        print(
            f"  {mode:9s}  wall={st['time']:.2f}s  "
            f"sample-forwards={st['computed']}/{steps * n}  skipped={st['skipped']}"
        )

    rows = []
    for i, prompt in enumerate(prompts):
        images = {m: mode_images[m][i] for m in ("base", "seacache", "hybrid")}
        psnr_sea = psnr(images["base"], images["seacache"])
        psnr_hyb = psnr(images["base"], images["hybrid"])
        l1_sea = l1_error(images["base"], images["seacache"])
        l1_hyb = l1_error(images["base"], images["hybrid"])
        row = {
            "idx": i,
            "prompt": prompt,
            "seed": seeds[i],
            "steps": steps,
            "psnr_seacache": psnr_sea,
            "psnr_hybrid": psnr_hyb,
            "l1_seacache": l1_sea,
            "l1_hybrid": l1_hyb,
            "seacache_computed": mode_stats["seacache"]["computed"],
            "hybrid_computed": mode_stats["hybrid"]["computed"],
            "seacache_skipped": mode_stats["seacache"]["skipped"],
            "hybrid_skipped": mode_stats["hybrid"]["skipped"],
            "time_base": mode_stats["base"]["time"] / n,
            "time_seacache": mode_stats["seacache"]["time"] / n,
            "time_hybrid": mode_stats["hybrid"]["time"] / n,
            "batch_wall_base": mode_stats["base"]["time"],
            "batch_wall_seacache": mode_stats["seacache"]["time"],
            "batch_wall_hybrid": mode_stats["hybrid"]["time"],
        }
        rows.append(row)
        print(
            f"  [{i}] PSNR seacache={psnr_sea:.3f} dB | hybrid={psnr_hyb:.3f} dB | "
            f"delta={psnr_hyb - psnr_sea:+.3f} dB"
        )
        grid_path = os.path.join(args.output_dir, f"{slugs[i]}_grid.png")
        labels = [
            "base (FLUX.1-dev)",
            f"SeaCache  PSNR={psnr_sea:.2f} dB",
            f"SeaCache+Spectrum  PSNR={psnr_hyb:.2f} dB",
        ]
        make_grid([images[m] for m in ("base", "seacache", "hybrid")], labels, grid_path)

    mean_sea = float(np.mean([r["psnr_seacache"] for r in rows]))
    mean_hyb = float(np.mean([r["psnr_hybrid"] for r in rows]))
    mean_t_base = float(np.mean([r["time_base"] for r in rows]))
    mean_t_sea = float(np.mean([r["time_seacache"] for r in rows]))
    mean_t_hyb = float(np.mean([r["time_hybrid"] for r in rows]))
    summary = {
        "model_id": args.model_id,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "seacache_thresh": args.seacache_thresh,
        "spectrum_m": args.spectrum_m,
        "spectrum_w": args.spectrum_w,
        "spectrum_lam": args.spectrum_lam,
        "spectrum_taylor_order": args.spectrum_taylor_order,
        "num_prompts": len(rows),
        "mean_psnr_seacache": mean_sea,
        "mean_psnr_hybrid": mean_hyb,
        "mean_psnr_delta": mean_hyb - mean_sea,
        "mean_time_base": mean_t_base,
        "mean_time_seacache": mean_t_sea,
        "mean_time_hybrid": mean_t_hyb,
        "mean_speedup_seacache": (mean_t_base / mean_t_sea) if mean_t_sea else None,
        "mean_speedup_hybrid": (mean_t_base / mean_t_hyb) if mean_t_hyb else None,
    }
    payload = {"summary": summary, "rows": rows}
    json_path = os.path.join(args.output_dir, "comparison.json")
    md_path = os.path.join(args.output_dir, "comparison.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_markdown(rows, summary, md_path)

    print("\n========== SUMMARY ==========")
    print(f"  Mean SeaCache PSNR : {mean_sea:.3f} dB")
    print(f"  Mean Hybrid PSNR   : {mean_hyb:.3f} dB")
    print(f"  Delta (hyb-sea)    : {mean_hyb - mean_sea:+.3f} dB")
    print(f"  Mean times (s)     : base={mean_t_base:.2f}  seacache={mean_t_sea:.2f}  hybrid={mean_t_hyb:.2f}")
    print(f"  Wrote {json_path}")
    print(f"  Wrote {md_path}")


if __name__ == "__main__":
    main()


