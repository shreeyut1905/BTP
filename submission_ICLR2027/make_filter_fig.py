#!/usr/bin/env python3
"""Generate figs/figA_filters.pdf: SEA Wiener filter responses G_t(f) and
G_t^norm(f) along the flow schedule -- exact formula used by the gate
(util_seacache.apply_sea_from_ab): S_x(f) = 1/(|f|^p + eps),
G = a S_x / (a^2 S_x + b^2), flow mixing a = 1 - sigma, b = sigma,
mean-gain normalization over the FFT grid. CPU-only, no model needed.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-16
P = 2.0          # power_exp used for FLUX (video patches use p=3, same shape)
N = 64           # 1D grid size (matches a 64-token spatial axis)

# rfft radial frequencies on [0, 0.5]
f = np.fft.rfftfreq(N)
sigmas = [0.95, 0.8, 0.6, 0.4, 0.2, 0.05]   # early -> late along the 50-step flow

fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
cmap = plt.get_cmap("viridis")
for i, s in enumerate(sigmas):
    a, b = 1.0 - s, s
    Sx = 1.0 / (np.abs(f) ** P + EPS)
    G = (a * Sx) / (a * a * Sx + b * b + EPS)
    Gn = G / G.mean()
    c = cmap(i / (len(sigmas) - 1))
    lbl = rf"$\sigma={s}$"
    axes[0].plot(f, G, color=c, lw=1.8, label=lbl)
    axes[1].plot(f, Gn, color=c, lw=1.8, label=lbl)

axes[0].set_title(r"(a) raw response $G_t(f)$", fontsize=10)
axes[1].set_title(r"(b) mean-normalized $G_t^{\mathrm{norm}}(f)$", fontsize=10)
for ax in axes:
    ax.set_xlabel("radial frequency $f$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.5)
axes[0].set_ylabel("gain", fontsize=9)
axes[1].legend(fontsize=7.5, ncol=2, frameon=False)
fig.tight_layout()
fig.savefig("figs/figA_filters.pdf", bbox_inches="tight")
print("wrote figs/figA_filters.pdf")
