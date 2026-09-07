#!/usr/bin/env python3
"""Build all paper figures (CPU-only) from finished run artifacts.

Reads: outputs_*/comparison*.json, logs/*/fwd-skip lines, examples_flux/.
Writes: PDF plots + PNG strips to --out (default: /tmp/paper_respect/figs,
next to the paper source which lives outside the git repo by design).

Run: ./.venv/bin/python scripts/make_figures.py [--out DIR]
"""
import argparse
import glob
import json
import os
import re
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

# SeaCache-published baselines (marked dagger in paper)
FLUX_PUB = {"TeaCache d0.3": (1547, 20.76), "TaylorSeer S=3": (1191, 22.78),
            "SeaCache d0.3": (1098, 26.29)}
FLUX_OURS = {"ReSPect d0.3": (1241.29, 27.969), "ReSPect d0.6": (773.76, 21.630)}
WAN_PUB = {"SeaCache d0.2": (3942, 26.60), "SeaCache d0.35": (2793, 21.78)}
WAN_OURS = {"ReSPect d0.2": (4302.9, 28.382), "ReSPect d0.35": (3335.6, 26.700)}


def load(p):
    with open(p) as f:
        return json.load(f)


def fig2_quality_compute(out):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    for a, pub, ours, title, ref in [
            (ax[0], FLUX_PUB, FLUX_OURS, "FLUX.1-dev, DrawBench-200", 2976),
            (ax[1], WAN_PUB, WAN_OURS, "Wan2.1-1.3B, VBench-946", 8214)]:
        for name, (tf, ps) in pub.items():
            a.scatter([tf], [ps], marker="s", s=36, c="gray", zorder=3)
            a.annotate(name, (tf, ps), fontsize=7, color="dimgray",
                       xytext=(3, 4), textcoords="offset points")
        names = sorted(ours, key=lambda n: ours[n][0])
        xs = [ours[n][0] for n in names]
        ys = [ours[n][1] for n in names]
        a.plot(xs, ys, "o-", c="C0", ms=5, label="ReSPect (ours)")
        for n in names:
            a.annotate(n, ours[n], fontsize=7, color="C0",
                       xytext=(3, -9), textcoords="offset points")
        a.axvline(ref, ls=":", c="k", lw=1)
        a.text(ref, a.get_ylim()[0] if a.get_ylim()[0] else 0, " full",
               fontsize=7)
        a.set_xlabel("TFLOPs")
        a.set_title(title, fontsize=10)
    ax[0].set_ylabel("PSNR (dB) vs uncached base")
    ax[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_quality_compute.pdf"))
    print("wrote fig2_quality_compute.pdf")


def _flux_rows(which):
    d = load(os.path.join(ROOT, f"outputs_hybrid_{which}",
                          "comparison_vs_base.json"))
    return d["rows"], d["summary"]


def figA_ssim_lpips(out):
    r03, _ = _flux_rows("d03")
    r06, _ = _flux_rows("d06")
    w02 = load(os.path.join(ROOT, "outputs_video_wan_d02", "comparison.json"))
    w35 = load(os.path.join(ROOT, "outputs_video_wan_d035", "comparison.json"))
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
    # FLUX per-prompt clouds (same TFLOPs per config -> jitter x)
    rng = np.random.default_rng(0)
    for rows, tf, c, lab in [(r03, 1241.29, "C0", "d0.3"), (r06, 773.76, "C1", "d0.6")]:
        x = np.array([tf]) + rng.normal(0, 8, len(rows))
        ax[0, 0].scatter(x, [r["ssim_hybrid"] for r in rows], s=4, c=c,
                         alpha=0.25, label=lab)
        ax[0, 1].scatter(x, [r["lpips_hybrid"] for r in rows if r.get("lpips_hybrid") is not None],
                         s=4, c=c, alpha=0.25, label=lab)
    ax[0, 0].set_title("FLUX SSIM per prompt"); ax[0, 0].set_xlabel("TFLOPs")
    ax[0, 1].set_title("FLUX LPIPS per prompt"); ax[0, 1].set_xlabel("TFLOPs")
    for a in (ax[0, 0], ax[0, 1]):
        a.legend(fontsize=7, markerscale=3)
    # Wan per-prompt clouds colored by exact/mp4
    for d, tf, c in [(w02, 4302.9, "C0"), (w35, 3335.6, "C1")]:
        for via, m in [("exact", "x"), ("mp4", "o")]:
            sub = [r for r in d["rows"] if r["via"] == via]
            x = np.array([tf]) + rng.normal(0, 15, len(sub))
            ax[1, 0].scatter(x, [r["ssim"] for r in sub], s=4, c=c,
                             alpha=0.3, marker=m)
            ax[1, 1].scatter(x, [r["lpips"] for r in sub], s=4, c=c,
                             alpha=0.3, marker=m)
    ax[1, 0].set_title("Wan SSIM per prompt (x=exact, o=MP4)")
    ax[1, 0].set_xlabel("TFLOPs")
    ax[1, 1].set_title("Wan LPIPS per prompt"); ax[1, 1].set_xlabel("TFLOPs")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figA_ssim_lpips.pdf"))
    print("wrote figA_ssim_lpips.pdf")


def _prompt_skips(logname):
    """per-prompt (computed, skipped) from fwd=/skip= log lines."""
    out = []
    with open(os.path.join(ROOT, "logs", logname)) as f:
        for line in f:
            m = re.search(r"fwd=(\d+)\s+skip=(\d+)", line)
            if m:
                out.append((int(m.group(1)), int(m.group(2))))
    return out


def figA_skip(out):
    cfgs = [("wan_hybrid_d02.log", "Wan d0.2", "C0"),
            ("wan_hybrid_d035.log", "Wan d0.35", "C1"),
            ("wan_hybrid_d02_r65.log", "Wan d0.2 tail", "C0"),
            ("wan_hybrid_d035_r124.log", "Wan d0.35 tail", "C1")]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for log, lab, c in cfgs:
        try:
            runs = _prompt_skips(log)
        except FileNotFoundError:
            continue
        rates = [s / (co + s) for co, s in runs]
        ax[0].hist(rates, bins=24, alpha=0.45, label=f"{lab} (n={len(runs)})",
                   color=c)
    ax[0].set_xlabel("skip rate per prompt")
    ax[0].set_ylabel("prompts")
    ax[0].set_title("Compute concentrates; most prompts skip ~half")
    ax[0].legend(fontsize=7)
    # PSNR vs skip rate (Wan full-946 join)
    for cmpf, log_main, log_tail, c, lab in [
            ("outputs_video_wan_d02/comparison.json",
             "wan_hybrid_d02.log", "wan_hybrid_d02_r65.log", "C0", "d0.2"),
            ("outputs_video_wan_d035/comparison.json",
             "wan_hybrid_d035.log", "wan_hybrid_d035_r124.log", "C1", "d0.35")]:
        d = load(os.path.join(ROOT, cmpf))
        runs = _prompt_skips(log_main) + _prompt_skips(log_tail)
        assert len(runs) == 946, (cmpf, len(runs))
        xs = [s / (co + s) for co, s in runs]
        ax[1].scatter(xs, [r["psnr"] for r in d["rows"]], s=5, c=c,
                      alpha=0.3, label=lab)
    ax[1].set_xlabel("skip rate per prompt")
    ax[1].set_ylabel("PSNR (dB)")
    ax[1].set_title("Quality holds as skips deepen (d0.35)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figA_skip.pdf"))
    print("wrote figA_skip.pdf")


def _fmean(d):
    s = load(os.path.join(ROOT, d, "comparison.json"))
    rows = s["rows"]
    key = next(k for k in rows[0] if k.startswith("psnr_"))
    return float(np.mean([r[key] for r in rows]))

def _wmean(d):
    s = load(os.path.join(ROOT, d, "comparison.json"))
    return float(np.mean([r["psnr"] for r in s["rows"]]))

def fig4_ablation(out):
    """Delta-dB vs copy-reuse. FLUX d0.3 (n=20) + Wan d0.35 (n=20)."""
    # FLUX: reuse baseline + variants (all n=20, same env)
    f_reuse = _fmean("outputs_abl_seacache")
    flux = [
        ("w=0", _fmean("outputs_abl_w0") - f_reuse),
        ("w=0.5", _fmean("outputs_abl_ref") - f_reuse),
        ("w=1", _fmean("outputs_abl_w1") - f_reuse),
        ("wu8", _fmean("outputs_abl_mw8") - f_reuse),
        ("wu12", _fmean("outputs_abl_min12") - f_reuse),
        ("wu16", _fmean("outputs_abl_mw16") - f_reuse),
        ("wu24", _fmean("outputs_abl_mw24") - f_reuse),
    ]
    # Wan d0.35: reuse + w variants; ref = main-run rows 0..19
    wfull = load(os.path.join(ROOT, "outputs_video_wan_d035",
                              "comparison.json"))["rows"][:20]
    w_reuse = _wmean("outputs_ablwan_sc")
    wan = [
        ("w=0", _wmean("outputs_ablwan_w0") - w_reuse),
        ("w=0.5", float(np.mean([r["psnr"] for r in wfull])) - w_reuse),
        ("w=1", _wmean("outputs_ablwan_w1") - w_reuse),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for a, pairs, title in [(ax[0], flux, "FLUX d0.3, n=20, ΔdB vs reuse"),
                            (ax[1], wan, "Wan d0.35, n=20, ΔdB vs reuse")]:
        names = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        cols = ["C1" if v < 0 else "C0" for v in vals]
        a.bar(range(len(names)), vals, color=cols)
        a.axhline(0, c="k", lw=0.8)
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=28, ha="right", fontsize=7)
        a.set_title(title, fontsize=10)
    ax[0].set_ylabel("ΔdB vs copy-reuse")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_ablation.pdf"))
    print("wrote fig4_ablation.pdf",
          {k: round(v, 2) for k, v in flux + wan})


def _read(p, maxw=None):
    im = Image.open(p).convert("RGB")
    if maxw and im.width > maxw:
        im = im.resize((maxw, int(im.height * maxw / im.width)),
                       Image.LANCZOS)
    return im


def _wan_frame(idx, hybrid_dir, frame=32):
    """(base_png, hybrid_mp4_frame) PIL pair for a prompt idx."""
    # tail idx live in the remainder dirs, not the main output dir
    if hybrid_dir.endswith("outputs_video_wan_d035") and idx >= 822:
        hybrid_dir = os.path.join(ROOT, "outputs_video_wan_d035_r124")
    if hybrid_dir.endswith("outputs_video_wan_d02") and idx >= 881:
        hybrid_dir = os.path.join(ROOT, "outputs_video_wan_d02_r65")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from eval_flux import read_prompts, safe_filename
    prompts = read_prompts(os.path.join(ROOT, "prompts", "vbench946.txt"))
    slug = f"{idx:04d}-{safe_filename(prompts[idx])[:50]}"
    base = Image.open(os.path.join(
        ROOT, "outputs_video_wan_d02", "videos",
        f"{slug}_base_f{frame:03d}.png")).convert("RGB")
    hpath = os.path.join(ROOT, hybrid_dir, "videos", f"{slug}_hybrid.mp4")
    frames = imageio.mimread(hpath)
    hyb = Image.fromarray(frames[frame]).convert("RGB").resize(base.size)
    return base, hyb, prompts[idx][:60]


def video_frames(out):
    """Temporal consistency: 3 prompts x (base row / ours row) x 4 frames."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from eval_flux import read_prompts, safe_filename
    prompts = read_prompts(os.path.join(ROOT, "prompts", "vbench946.txt"))
    picks = [(834, "outputs_video_wan_d035_r124", "d0.35"),
             (860, "outputs_video_wan_d035_r124", "d0.35"),
             (852, "outputs_video_wan_d035_r124", "d0.35")]
    frames = [8, 24, 40, 56]
    TW = 320
    rows = []
    for idx, hdir, tag in picks:
        slug = f"{idx:04d}-{safe_filename(prompts[idx])[:50]}"
        bpaths = [os.path.join(ROOT, "outputs_video_wan_d02", "videos",
                               f"{slug}_base_f{f:03d}.png") for f in frames]
        base = [Image.open(p).convert("RGB") for p in bpaths]
        hpath = os.path.join(ROOT, hdir, "videos", f"{slug}_hybrid.mp4")
        hfr = imageio.mimread(hpath)
        hyb = [Image.fromarray(hfr[f]).convert("RGB").resize(base[0].size)
               for f in frames]
        lab = f"{idx} {prompts[idx][:52]} ({tag})"
        rows.append((lab, base, hyb))
    W = 4 * TW + 3 * 6
    th = int(rows[0][1][0].height * TW / rows[0][1][0].width)
    BAND = 44
    H = len(rows) * 2 * (th + BAND) + 10
    canvas = Image.new("RGB", (W, H), "white")
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    y = 0
    for lab, base, hyb in rows:
        _label(d, (6, y + 6), lab + "   [top: base / bottom: ours]",
               size=30)
        y += BAND
        for fr in base, hyb:
            x = 0
            for im in fr:
                canvas.paste(im.resize((TW, th), Image.LANCZOS), (x, y))
                x += TW + 6
            y += th
    canvas.save(os.path.join(out, "fig_video_frames.png"))
    print("wrote fig_video_frames.png", canvas.size)


def rebuild_strips(out=None):
    """Regenerate examples_flux strips with large readable captions."""
    from PIL import ImageFont
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from eval_flux import read_prompts, safe_filename
    prompts = read_prompts(os.path.join(ROOT, "prompts", "drawbench200.txt"))
    d = load(os.path.join(ROOT, "outputs_hybrid_d03",
                          "comparison_vs_base.json"))
    by_idx = {r["idx"]: r for r in d["rows"]}
    font_big = ImageFont.truetype(FONT, 40)
    font_med = ImageFont.truetype(FONT, 34)
    dest = os.path.join(ROOT, "examples_flux")
    names = {93: "flux_093_base_vs_ours.png",
             143: "flux_143_base_vs_ours.png",
             156: "flux_156_base_vs_ours.png"}
    for idx, fname in names.items():
        p = prompts[idx]
        slug = f"{idx:02d}-{safe_filename(p)[:60]}"
        b = Image.open(os.path.join(ROOT, "outputs_base", "images",
                                    f"{slug}_base.png")).convert("RGB")
        h = Image.open(os.path.join(ROOT, "outputs_hybrid_d03", "images",
                                    f"{slug}_hybrid.png")).convert("RGB")
        W = 1024
        bh = int(b.height * W / b.width)
        b, h = b.resize((W, bh), Image.LANCZOS), h.resize((W, bh),
                                                          Image.LANCZOS)
        r = by_idx[idx]
        cap = (f"idx {idx}  PSNR {r['psnr_hybrid']:.2f}dB  SSIM "
               f"{r['ssim_hybrid']:.3f}  LPIPS {r['lpips_hybrid']:.3f}")
        from PIL import ImageDraw
        top, bot = 64, 58
        canvas = Image.new("RGB", (2 * W + 12, top + bh + bot), "white")
        dr = ImageDraw.Draw(canvas)
        dr.text((8, 10), "BASE (FLUX.1-dev, 50 steps)", fill="black",
                font=font_big)
        dr.text((W + 20, 10), "OURS (hybrid, seacache_thresh=0.3)",
                fill="black", font=font_big)
        canvas.paste(b, (0, top))
        canvas.paste(h, (W + 12, top))
        dr.text((8, top + bh + 8), cap, fill="black", font=font_med)
        dr.text((8, top + bh + 8 + 40), p[:110], fill="black",
                font=font_med)
        canvas.save(os.path.join(dest, fname))
        print("rebuilt", fname, canvas.size)


def _side_by_side(base, hyb, h=420):
    w = int(base.width * h / base.height)
    b, hh = base.resize((w, h), Image.LANCZOS), hyb.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (2 * w + 6, h), "white")
    canvas.paste(b, (0, 0)); canvas.paste(hh, (w + 6, 0))
    return canvas


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _label(draw, xy, text, size=30):
    from PIL import ImageFont
    draw.text(xy, text, fill="black",
              font=ImageFont.truetype(FONT, size))


def _pair_strip(base, hyb, label, width=1400):
    """vertical base/top hybrid/bottom strip with labels."""
    w = width // 2
    h = int(base.height * w / base.width)
    b, hh = base.resize((w, h)), hyb.resize((w, h))
    canvas = Image.new("RGB", (2 * w + 8, h + 44), "white")
    canvas.paste(b, (0, 44)); canvas.paste(hh, (w + 8, 44))
    from PIL import ImageDraw
    _label(ImageDraw.Draw(canvas), (6, 6),
           f"base (left)  |  ours (right)   {label}", size=30)
    return canvas


def quals(out):
    # compact teaser: 3 base|ours pairs side by side (fits one row)
    pairs = []
    f143 = _read(os.path.join(ROOT, "examples_flux",
                              "flux_143_base_vs_ours.png"))
    # strips are base|ours halves already
    pairs.append(("flux fennec", f143))
    f093 = _read(os.path.join(ROOT, "examples_flux",
                              "flux_093_base_vs_ours.png"))
    pairs.append(("flux robot", f093))
    b, h, lab = _wan_frame(859, "outputs_video_wan_d02", frame=32)
    pairs.append(("wan volcano", _side_by_side(b, h)))
    H = 420
    cells = []
    for lab, im in pairs:
        w = int(im.width * H / im.height)
        cells.append(im.resize((w, H), Image.LANCZOS))
    W = sum(c.width for c in cells)
    teaser = Image.new("RGB", (W, H), "white")
    x = 0
    for c in cells:
        teaser.paste(c, (x, 0))
        x += c.width
    teaser.save(os.path.join(out, "teaser_compact.png"))
    print("wrote teaser_compact.png", W, "x", H)
    # fig3: flux text-heavy strip + 1 wan pair (fits one figure* page)
    f156 = _read(os.path.join(ROOT, "examples_flux",
                              "flux_156_base_vs_ours.png"), maxw=1400)
    f156 = f156.resize((1400, int(f156.height * 1400 / f156.width)),
                       Image.LANCZOS)
    b2, h2, lab2 = _wan_frame(918, "outputs_video_wan_d035", frame=32)
    row2 = _pair_strip(b2, h2, "wan " + lab2, width=1400)
    row2 = row2.crop((0, 44, 1400, row2.height))  # drop label band
    fig3 = Image.new("RGB", (1400, f156.height + row2.height), "white")
    fig3.paste(f156, (0, 0))
    fig3.paste(row2, (0, f156.height))
    fig3.save(os.path.join(out, "fig3_qual.png"))
    print("wrote fig3_qual.png", fig3.size)
    # appendix extra: fennec strip + volcano pair
    b4, h4, lab4 = _wan_frame(859, "outputs_video_wan_d02", frame=32)
    f143b = _read(os.path.join(ROOT, "examples_flux",
                               "flux_143_base_vs_ours.png"), maxw=1400)
    figA = Image.new("RGB", (1400, f143b.height + 480), "white")
    figA.paste(f143b.resize((1400, f143b.height)), (0, 0))
    figA.paste(_pair_strip(b4, h4, "wan " + lab4, width=1400),
               (0, f143b.height))
    figA.save(os.path.join(out, "figA_qual.png"))
    print("wrote figA_qual.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/paper_respect/figs")
    ap.add_argument("--rebuild_strips", action="store_true",
                    help="regenerate examples_flux strips with big fonts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig2_quality_compute(args.out)
    figA_ssim_lpips(args.out)
    figA_skip(args.out)
    quals(args.out)
    video_frames(args.out)
    fig4_ablation(args.out)
    if args.rebuild_strips:
        rebuild_strips()


if __name__ == "__main__":
    main()
