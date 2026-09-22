#!/usr/bin/env python3
"""Gate-schedule ablation figure: rolling vs last-compute SEA reference.

HunyuanVideo, 946 VBench prompts, 480p, 65 frames, 50 steps. Both gate
variants share an identical residual forecaster, so the gap is
attributable to the skip schedule alone.

Run: .venv/bin/python make_gate_ablation_fig.py
Writes: figs/fig_gate_ablation.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

BASE_TFLOPS = 14038.0
STEPS = 50

# (tflops, eval_steps, psnr, ssim, lpips, delta)
FROZEN = [(8984.3, 32.000, 35.939, 0.9574, 0.0352, 0.19),
          (6738.2, 24.000, 31.568, 0.9153, 0.0712, 0.35)]
ROLLING = [(6741.5, 24.012, 34.066, 0.9461, 0.0464, 0.27),
           (4495.4, 16.012, 29.203, 0.8907, 0.1001, 0.48)]
C_ROLL, C_FROZ = "#1b6ca8", "#c44e52"


def series(ax, data, idx, color, marker, ls, label, dlabels=None):
    xs = [d[1] for d in data]
    ys = [d[idx] for d in data]
    ax.plot(xs, ys, marker=marker, ls=ls, color=color, label=label,
            ms=6, lw=1.6, mec="white", mew=0.8, zorder=3)
    if dlabels:
        for d, off in zip(data, dlabels):
            ax.annotate(rf"$\delta{{=}}{d[5]}$", (d[1], d[idx]),
                        textcoords="offset points", xytext=off,
                        fontsize=7, color=color, zorder=4)
    return xs, ys


fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

# Only our own thresholds are annotated; the baselines' deltas are not
# comparable across implementations (see caption).
for ax, idx, ylab, offs in [
        (axes[0], 2, "PSNR (dB)", [(-40, 6), (6, -13)]),
        (axes[1], 4, "LPIPS", [(-42, -4), (7, 2)])]:
    series(ax, FROZEN, idx, C_FROZ, "s", "--", "Last-compute reference")
    series(ax, ROLLING, idx, C_ROLL, "o", "-", "Rolling reference (ours)",
           dlabels=offs)
    ax.set_xlabel("NFE per video (of 50)")
    ax.set_ylabel(ylab)
    ax.set_xlim(14.2, 34.2)

# Highlight the one budget where the two schedules overlap (24 steps).
ax = axes[0]
x0 = 24.0
ax.set_ylim(25.4, 37.4)
ax.axvline(x0, color="black", lw=0.8, ls=(0, (2, 3)), alpha=0.55, zorder=1)
ax.annotate("", xy=(x0, 34.066), xytext=(x0, 31.568),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.1))
ax.annotate("$+2.50$ dB\nsame budget,\nsame forecaster",
            xy=(x0 - 0.6, 32.8), xytext=(0, 0), textcoords="offset points",
            fontsize=7.5, ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.28", fc="white",
                      ec="black", lw=0.5, alpha=0.92), zorder=5)
ax.set_title("Quality vs. budget", fontsize=9)
axes[1].set_title("Perceptual distance", fontsize=9)
axes[0].legend(loc="lower right", fontsize=7.2, framealpha=0.95)

figs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
for ext in ("pdf", "png"):
    out = os.path.join(figs, f"fig_gate_ablation.{ext}")
    fig.savefig(out)
    print("wrote", out)
