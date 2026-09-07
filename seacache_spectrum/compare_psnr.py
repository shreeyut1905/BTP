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


def ssim_score(ref: Image.Image, pred: Image.Image) -> float:
    from skimage.metrics import structural_similarity

    a = image_to_float(ref)
    b = image_to_float(pred)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))


_LPIPS_METRIC = None


def lpips_score(ref: Image.Image, pred: Image.Image, device: str = "cuda") -> float:
    """LPIPS (AlexNet) via pyiqa. Lower is better. Expects RGB [0,1] NCHW."""
    global _LPIPS_METRIC
    import torch

    if _LPIPS_METRIC is None:
        import pyiqa

        dev = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        _LPIPS_METRIC = pyiqa.create_metric("lpips", device=dev)
    def pil_to_nchw(img: Image.Image):
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t.to(_LPIPS_METRIC.device if hasattr(_LPIPS_METRIC, "device") else device)
    with torch.no_grad():
        v = _LPIPS_METRIC(pil_to_nchw(pred), pil_to_nchw(ref))
        if isinstance(v, torch.Tensor):
            return float(v.detach().reshape(-1)[0].item())
        return float(v)
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
    p.add_argument("--modes", nargs="+", default=["base", "seacache", "hybrid"],
                   help="subset of [base, seacache, hybrid] to generate (base needed as reference)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--reuse_images", action="store_true",
                   help="reuse existing mode PNGs in output_dir/images instead of regenerating")
    p.add_argument("--no_lpips", action="store_true", help="skip LPIPS (faster smoke test)")
    p.add_argument("--no_grid", action="store_true", help="skip side-by-side grid PNGs")
    p.add_argument("--base_tflops", type=float, default=2976.0,
                   help="reference TFLOPs for full 50-step 1024x1024 run (SeaCache Table 1)")
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
    p.add_argument("--compile", action="store_true",
                   help="torch.compile FLUX transformer blocks for speed")
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
    if args.compile:
        print(f"[{now_str()}] torch.compiling FLUX transformer blocks...")
        for i, b in enumerate(tr.transformer_blocks):
            tr.transformer_blocks[i] = torch.compile(b, fullgraph=False)
        for i, b in enumerate(tr.single_transformer_blocks):
            tr.single_transformer_blocks[i] = torch.compile(b, fullgraph=False)

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

    modes = [m for m in args.modes if m in ("base", "seacache", "hybrid")]
    if not modes:
        modes = ["hybrid"]
    has_base = "base" in modes
    if not has_base:
        print("NOTE: base not in --modes; skipping PSNR/SSIM/LPIPS (need base reference).")
        print("Will report Latency, computed/skipped ratio, TFLOPs only.")
    bs = max(1, int(args.batch_size))

    mode_images: dict = {m: [None] * n for m in modes}
    mode_wall = {m: 0.0 for m in modes}
    mode_computed = {m: 0 for m in modes}
    mode_skipped = {m: 0 for m in modes}
    for mode in modes:
        # reuse check: all PNGs exist?
        if args.reuse_images and all(
            os.path.exists(os.path.join(img_dir, f"{slugs[i]}_{mode}.png")) for i in range(n)
        ):
            print(f"\n[{now_str()}] Reusing existing images mode={mode}")
            for i in range(n):
                mode_images[mode][i] = Image.open(
                    os.path.join(img_dir, f"{slugs[i]}_{mode}.png")
                ).convert("RGB")
            continue
        wall = 0.0
        comp = 0
        skip = 0
        for s in range(0, n, bs):
            e = min(n, s + bs)
            chunk_prompts = prompts[s:e]
            chunk_seeds = seeds[s:e]
            chunk_n = e - s
            reset_cache_state(
                tr, steps, mode,
                hybrid_ns if mode == "hybrid" else args,
                batch_size=chunk_n,
            )
            print(f"\n[{now_str()}] Generating mode={mode} batch {s}:{e} B={chunk_n}")
            imgs, st = generate_batch(
                pipe, chunk_prompts, chunk_seeds, height, width, steps, args.guidance, device
            )
            wall += st["time"]
            comp += st["computed"]
            skip += st["skipped"]
            for j, img in enumerate(imgs):
                i = s + j
                mode_images[mode][i] = img
                img.save(os.path.join(img_dir, f"{slugs[i]}_{mode}.png"))
            print(
                f"  {mode:9s} batch wall={st['time']:.2f}s "
                f"sample-forwards={st['computed']}/{steps * chunk_n} skipped={st['skipped']}"
            )
            if device == "cuda":
                torch.cuda.empty_cache()
        mode_wall[mode] = wall
        mode_computed[mode] = comp
        mode_skipped[mode] = skip
        print(
            f"  {mode:9s} TOTAL wall={wall:.2f}s s/img={wall / n:.2f}s "
            f"sample-forwards={comp}/{steps * n} skipped={skip}"
        )

    # TFLOPs: scale SeaCache reference (2976 TFLOPs for full 50-step 1024x1024)
    # base = full compute; cached modes scale by computed ratio.
    def tflops_for(mode: str) -> float:
        if mode == "base":
            return float(args.base_tflops)
        ratio = mode_computed[mode] / max(1, steps * n)
        return float(args.base_tflops * ratio)

    mode_tflops = {m: tflops_for(m) for m in modes}
    mode_s_per_img = {m: mode_wall[m] / n for m in modes}

    rows = []
    for i, prompt in enumerate(prompts):
        if has_base:
            base_img = mode_images["base"][i]
            row = {
                "idx": i,
                "prompt": prompt,
                "seed": seeds[i],
                "steps": steps,
                "time_base": mode_s_per_img["base"],
                "batch_wall_base": mode_wall["base"],
                "tflops_base": mode_tflops["base"],
            }
        else:
            row = {"idx": i, "prompt": prompt, "seed": seeds[i], "steps": steps}
        for m in ("seacache", "hybrid"):
            if m not in modes:
                continue
            img = mode_images[m][i]
            if has_base:
                row[f"psnr_{m}"] = psnr(base_img, img)
                row[f"ssim_{m}"] = ssim_score(base_img, img)
                row[f"l1_{m}"] = l1_error(base_img, img)
                if not args.no_lpips:
                    try:
                        row[f"lpips_{m}"] = lpips_score(base_img, img, device)
                    except Exception as e:
                        print(f"  [{i}] LPIPS failed for {m}: {e}")
                        row[f"lpips_{m}"] = None
            row[f"{m}_computed"] = mode_computed[m]
            row[f"{m}_skipped"] = mode_skipped[m]
            row[f"time_{m}"] = mode_s_per_img[m]
            row[f"batch_wall_{m}"] = mode_wall[m]
            row[f"tflops_{m}"] = mode_tflops[m]
        rows.append(row)
        if has_base:
            msg = f"  [{i}]"
            for m in ("seacache", "hybrid"):
                if m in modes:
                    msg += f" {m} PSNR={row[f'psnr_{m}']:.2f} SSIM={row[f'ssim_{m}']:.4f}"
                    if not args.no_lpips and row.get(f"lpips_{m}") is not None:
                        msg += f" LPIPS={row[f'lpips_{m}']:.4f}"
                    msg += " |"
            print(msg)
        elif i % 10 == 0:
            print(f"  [{i}] generated (no-ref mode)")
        if not args.no_grid and has_base:
            grid_modes = [m for m in ("base", "seacache", "hybrid") if m in modes]
            grid_path = os.path.join(args.output_dir, f"{slugs[i]}_grid.png")
            labels = []
            for m in grid_modes:
                if m == "base":
                    labels.append("base (FLUX.1-dev)")
                else:
                    labels.append(f"{m} PSNR={row[f'psnr_{m}']:.2f} dB")
            make_grid([mode_images[m][i] for m in grid_modes], labels, grid_path)

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
        "batch_size": bs,
        "modes": modes,
        "base_tflops_ref": float(args.base_tflops),
        "has_base_reference": has_base,
    }
    if has_base:
        summary["mean_time_base"] = mode_s_per_img["base"]
        summary["tflops_base"] = mode_tflops["base"]
        summary["total_wall_base"] = mode_wall["base"]
    for m in ("seacache", "hybrid"):
        if m not in modes:
            continue
        if has_base:
            psnrs = [r[f"psnr_{m}"] for r in rows]
            ssims = [r[f"ssim_{m}"] for r in rows]
            l1s = [r[f"l1_{m}"] for r in rows]
            lpips_vals = [r[f"lpips_{m}"] for r in rows if r.get(f"lpips_{m}") is not None]
            summary[f"mean_psnr_{m}"] = float(np.mean(psnrs))
            summary[f"mean_ssim_{m}"] = float(np.mean(ssims))
            summary[f"mean_l1_{m}"] = float(np.mean(l1s))
            summary[f"mean_lpips_{m}"] = float(np.mean(lpips_vals)) if lpips_vals else None
        summary[f"mean_time_{m}"] = mode_s_per_img[m]
        summary[f"tflops_{m}"] = mode_tflops[m]
        summary[f"total_wall_{m}"] = mode_wall[m]
        summary[f"computed_{m}"] = mode_computed[m]
        summary[f"skipped_{m}"] = mode_skipped[m]
        summary[f"refresh_ratio_{m}"] = mode_computed[m] / max(1, steps * len(rows))
        if has_base:
            summary[f"mean_speedup_{m}"] = (
                mode_s_per_img["base"] / mode_s_per_img[m] if mode_s_per_img[m] else None
            )
        else:
            # theoretical speedup vs full compute (no base wall available)
            summary[f"theoretical_speedup_{m}"] = (steps * len(rows)) / max(1, mode_computed[m])
    if "seacache" in modes and "hybrid" in modes:
        summary["mean_psnr_delta"] = summary["mean_psnr_hybrid"] - summary["mean_psnr_seacache"]
    payload = {"summary": summary, "rows": rows}
    json_path = os.path.join(args.output_dir, "comparison.json")
    md_path = os.path.join(args.output_dir, "comparison.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    try:
        if has_base:
            write_markdown(rows, summary, md_path)
        else:
            raise RuntimeError("no base reference, use paper-metrics markdown")
    except Exception as e:
        print(f"write_markdown fallback: {e}")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# FLUX.1-dev paper metrics\n\n")
            f.write(f"- prompts: {len(rows)} modes={modes}\n")
            for m in ("seacache", "hybrid"):
                if m in modes:
                    if has_base:
                        f.write(
                            f"- {m}: PSNR={summary[f'mean_psnr_{m}']:.3f} "
                            f"SSIM={summary[f'mean_ssim_{m}']:.4f} "
                            f"LPIPS={summary[f'mean_lpips_{m}']} "
                            f"lat={summary[f'mean_time_{m}']:.2f}s "
                            f"TFLOPs={summary[f'tflops_{m}']:.1f} "
                            f"speedup={summary.get(f'mean_speedup_{m}')}x\n"
                        )
                    else:
                        f.write(
                            f"- {m}: lat={summary[f'mean_time_{m}']:.2f}s "
                            f"TFLOPs={summary[f'tflops_{m}']:.1f} "
                            f"theoretical_speedup={summary[f'theoretical_speedup_{m}']:.2f}x "
                            f"computed={summary[f'computed_{m}']} "
                            f"skipped={summary[f'skipped_{m}']} "
                            f"refresh={summary[f'refresh_ratio_{m}']:.3f}\n"
                        )

    print("\n========== SUMMARY ==========")
    for m in ("seacache", "hybrid"):
        if m in modes:
            if has_base:
                print(
                    f"  {m:9s} PSNR={summary[f'mean_psnr_{m}']:.3f} "
                    f"SSIM={summary[f'mean_ssim_{m}']:.4f} "
                    f"LPIPS={summary[f'mean_lpips_{m}']} "
                    f"time={summary[f'mean_time_{m}']:.2f}s "
                    f"TFLOPs={summary[f'tflops_{m}']:.1f} "
                    f"speedup={summary.get(f'mean_speedup_{m}')}x"
                )
            else:
                print(
                    f"  {m:9s} time={summary[f'mean_time_{m}']:.2f}s "
                    f"TFLOPs={summary[f'tflops_{m}']:.1f} "
                    f"theoretical_speedup={summary[f'theoretical_speedup_{m}']:.2f}x "
                    f"computed={summary[f'computed_{m}']}"
                )
    print(f"  Wrote {json_path}")
    print(f"  Wrote {md_path}")


if __name__ == "__main__":
    main()


