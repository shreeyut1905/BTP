#!/usr/bin/env python3
"""Generate figs/fig_overview.pdf: ReSPect dataflow overview in the style
of SeaCache Fig. 3 -- (a) the spectral gate (WHEN to skip, inherited),
(b) the residual forecaster (WHAT to fill in, ours). CPU-only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

GRAY, GRAY_E = "#f2f2f2", "#888888"
BLUE, BLUE_E = "#e2efff", "#1f4e9c"
YELO, YELO_E = "#fff7e0", "#9c7a1f"
FS = 8.5   # box text size
FL = 7.0   # edge-label size

fig, ax = plt.subplots(figsize=(7.5, 4.9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.4)
ax.axis("off")


def box(x, y, w, h, text, fc=GRAY, ec=GRAY_E, fs=FS, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.35)


def arrow(x1, y1, x2, y2, label=None, color="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.1, color=color,
                                 shrinkA=1, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label,
                ha="center", va="bottom", fontsize=FL, color=color)


# ---------------- (a) spectral gate: WHEN to skip ----------------
ax.text(0.15, 5.95, "(a) spectral gate -- WHEN to skip (SeaCache, kept as is)",
        fontsize=9, style="italic")
BW, BH = 1.02, 0.52
y_top, y_bot = 5.05, 4.05
xs = [0.25, 1.62, 2.99, 4.36, 5.73]
box(xs[0], y_top, BW, BH, r"$I_t$")
box(xs[1], y_top, BW, BH, "FFT")
box(xs[2], y_top, BW, BH, r"$\times\,G_t^{\rm norm}$")
box(xs[3], y_top, BW, BH, "iFFT")
box(xs[4], y_top, BW, BH, r"$\mathcal{P}_t$")
box(xs[0], y_bot, BW, BH, r"$I_{t+1}$")
box(xs[1], y_bot, BW, BH, "FFT")
box(xs[2], y_bot, BW, BH, r"$\times\,G_{t+1}^{\rm norm}$")
box(xs[3], y_bot, BW, BH, "iFFT")
box(xs[4], y_bot, BW, BH, r"$\mathcal{P}_{t+1}$")
for k in range(4):
    arrow(xs[k] + BW, y_top + BH / 2, xs[k + 1], y_top + BH / 2)
    arrow(xs[k] + BW, y_bot + BH / 2, xs[k + 1], y_bot + BH / 2)
# distance + accumulator box
box(7.15, 4.05, 2.6, 1.52,
    r"${\rm L1_{rel}}(\mathcal{P}_t,\mathcal{P}_{t+1})\to\tilde{\Delta}_t$"
    "\naccumulate; fire if $>\\delta$", fc="#e9e9e9")
arrow(xs[4] + BW, y_top + BH / 2, 7.15, 5.15)
arrow(xs[4] + BW, y_bot + BH / 2, 7.15, 4.45)

# ---------------- (b) residual forecast: WHAT to fill in ----------------
ax.text(0.15, 3.45, "(b) residual forecast -- WHAT to fill in (ours)",
        fontsize=9, style="italic", color=BLUE_E)
# decision diamond
cx, cy = 5.0, 2.55
ax.add_patch(Polygon([[cx, cy + 0.5], [cx + 1.05, cy], [cx, cy - 0.5],
                      [cx - 1.05, cy]], closed=True, fc=YELO, ec=YELO_E, lw=1.1))
ax.text(cx, cy, "recompute?\n(accumulator $>\\delta$)",
        ha="center", va="center", fontsize=FS)
arrow(8.45, 4.05, cx + 0.55, cy + 0.42)  # gate decision flows down
# compute branch (left, inherits cached residual + updates forecaster)
box(0.35, 0.55, 3.6, 1.15,
    "run blocks: $h_{\\rm in}\\to h_{\\rm out}$\n"
    r"$r \leftarrow h_{\rm out}-h_{\rm in}$;  $\mathcal{F}.{\rm update}(t,r)$",
    fc=BLUE, ec=BLUE_E)
# skip branch (right, forecast)
box(5.95, 0.55, 3.7, 1.15,
    r"$\hat{r} \leftarrow \mathcal{F}.{\rm predict}(t)$"
    "\n" r"$h \leftarrow h_{\rm in}+\hat{r}$"
    "\nfallback: last $r$ ($<1$ obs.); Taylor-only ($<4$)",
    fc=BLUE, ec=BLUE_E)
ax.text(cx - 1.35, 1.85, "yes", fontsize=FL, style="italic")
ax.text(cx + 1.25, 1.85, "no", fontsize=FL, style="italic")
arrow(cx - 0.75, cy - 0.28, 1.6, 1.7)
arrow(cx + 0.75, cy - 0.28, 8.4, 1.7)
ax.text(cx, 0.28, "output projection always runs; forecast state is per-sample",
        ha="center", va="center", fontsize=FL, style="italic", color="#555555")
fig.tight_layout()
fig.savefig("figs/fig_overview.pdf", bbox_inches="tight")
print("wrote figs/fig_overview.pdf")
