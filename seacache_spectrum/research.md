# SeaCache + Spectrum: Spectral Skip Scheduling with Feature Forecasting

This note records the hybrid acceleration experiment on **FLUX.1-dev**
(1024×1024): reuse SeaCache’s *frequency / SEA* skip schedule, but on
skipped denoising steps **forecast** the transformer residual the way
Spectrum does, instead of copying the last cached residual.

Code lives in `seacache_spectrum/`. Upstream clones:

- SeaCache: [jiwoogit/SeaCache](https://github.com/jiwoogit/SeaCache) (CVPR 2026 Oral)
- Spectrum: [hanjq17/Spectrum](https://github.com/hanjq17/Spectrum) (CVPR 2026)

---

## 1. Motivation

Both methods accelerate diffusion by skipping transformer forwards:

| | SeaCache | Spectrum |
|---|---|---|
| **When to skip** | Adaptive: SEA-filtered first-block modulation, accumulated relative L1 vs threshold δ | Fixed / flex window (`window_size`, `flex_window`) |
| **What to do on a skip** | **Reuse** last residual: `h ← h + r_prev` | **Forecast** post-block hidden states with Chebyshev ridge + discrete Taylor |
| **Frequency** | Yes — Wiener filter on spatial FFT using scheduler `(a_t, b_t)` | Spectral in *time*: Chebyshev polynomials over the denoising trajectory |

The hybrid question is: **keep SeaCache’s content-aware skip gate, but replace
copy-reuse with Spectrum’s time-series forecast.** If residuals evolve
smoothly, a forecast should track the uncached trajectory better than a
stale copy, raising PSNR vs the base model.

---

## 2. Method

### 2.1 SeaCache skip gate (unchanged)

On FLUX, after `x_embedder` / `time_text_embed` / `context_embedder`:

1. Take the first dual-stream block’s `norm1` modulation of image tokens.
2. Reshape to `(B, H, W, C)` and apply the SEA Wiener filter
   `apply_sea_with_scheduler` (power-law signal, scheduler mixing
   `a_t = 1-σ`, `b_t = σ` in flow mode, mean-normalized).
3. Accumulate relative L1 vs the previous filtered modulation.
4. If the accumulator is below `seacache_thresh` (δ = 0.3 here), **skip**
   the full transformer; else compute and reset the accumulator.
5. Always compute the first and last step.

### 2.2 Spectrum forecast (used only on skipped steps)

Spectrum’s FLUX forward (`Spectrum/src/pipelines/flux_forward.py`) does
**not** add a residual. On a skip it *replaces* hidden states with a
forecast of previously observed post-block features.

The forecaster (`Spectrum/src/utils/basis_utils.py` + `src/utils/__init__.py`):

- Window of up to **K = 100** observed `(t, h)` pairs.
- Chebyshev ridge regression, **M = 4** bases, **λ = 0.1**.
- Time mapped to τ ∈ [-1, 1] with a **fixed** range `[0, 50]` (not the
  buffer min/max).
- Blend with a first-order discrete Taylor / Newton step:

  `h_mix = (1 - w) * h_taylor + w * h_cheb`

Code default is **`w = 0.5`** (`src/utils/__init__.py`). The yaml
`configs/algo/spectrum.yaml` lists `w: 1.0`; we follow the **code**
default, as requested.

### 2.3 Hybrid rule

On a **computed** step (SeaCache said “run”):

- Run the full dual-stream + single-stream stack.
- Cache residual `r_t = h_out - h_in` (SeaCache).
- **Update** a per-sample Spectrum forecaster with `(t, r_t)`.

On a **skipped** step:

- If the forecaster has ≥ 1 observation, predict `r̂_t` and set
  `h ← h + r̂_t`.
- Else fall back to SeaCache copy `h ← h + r_prev`.

Chebyshev is mixed in only after **`min_cheb_obs = 4`** computed
residuals. With 1 observation, Taylor reduces to last-residual copy, so
early skips match SeaCache. After that, `w = 0.5` blends Taylor
extrapolation with the Chebyshev fit.

Skip decisions remain **per sample** so a batched run does not force
every prompt to skip on the same steps.

---

## 3. Implementation notes (bugs we hit)

A first hybrid was *worse* than SeaCache (~−0.65 dB mean). Causes:

1. **`w = 1.0`** (pure Chebyshev). A 4th-order ridge fit on a handful of
   points overshoots; Spectrum’s intended mix is `w = 0.5`.
2. **`K = 32`** instead of Spectrum’s **`K = 100`**.
3. Chebyshev mixed in from the **first** observation, so the earliest
   skips (where SeaCache copy is already a good local approximation)
   were polluted by an underdetermined polynomial.
4. Forecasting **hidden states as if they were residuals** (or the
   reverse) is easy to get wrong. SeaCache’s cached object is the
   residual; the hybrid therefore forecasts **residuals** and adds them,
   which is the drop-in replacement for copy-reuse.

After those fixes, hybrid PSNR was above SeaCache on **all four** images.

Batching: FLUX.1-dev 1024×1024 fits comfortably on an NVIDIA B200
(~38 GB). All four prompts are generated in **one pipeline call per
mode** (base / seacache / hybrid), with per-sample generators and
per-sample forecasters.

---

## 4. Experiment protocol

- **Model:** `black-forest-labs/FLUX.1-dev`
- **Resolution:** 1024 × 1024
- **Steps:** 50, guidance 3.5, bf16
- **SeaCache δ:** 0.3
- **Spectrum:** m=4, w=0.5, λ=0.1, K=100, taylor_order=1, min_cheb_obs=4
- **Seeds:** 0, 1, 2, 3 (one per prompt)
- **PSNR:** RGB, vs the uncached base image, `10 log10(1 / MSE)` on `[0, 1]` pixels
- **Hardware:** NVIDIA B200, 183 GB

Prompts:

1. a high-resolution photo of a panda drinking coffee in a cozy cafe
2. a photo of an astronaut riding a horse on the moon
3. a red double-decker bus driving through a rainy London street at night, cinematic lighting
4. a golden retriever sitting in a field of sunflowers under a bright blue sky

Reproduce:

```bash
cd seacache_spectrum
python compare_psnr.py \
  --width 1024 --height 1024 \
  --num_inference_steps 50 \
  --num_prompts 4 \
  --seacache_thresh 0.3 \
  --spectrum_w 0.5 --spectrum_k 100 \
  --output_dir ./outputs
```

FLUX.1-dev is gated; a Hugging Face token with access is required.

---

## 5. Results (1024×1024, vs base)

| # | Prompt | SeaCache PSNR | Hybrid PSNR | Δ (hybrid − SeaCache) |
|---|--------|---------------:|------------:|----------------------:|
| 0 | panda in cafe | 27.479 | **28.228** | **+0.749** |
| 1 | astronaut on moon | 24.375 | **25.657** | **+1.282** |
| 2 | London bus | 26.732 | **28.244** | **+1.512** |
| 3 | golden retriever / sunflowers | 30.401 | **32.019** | **+1.618** |
| | **mean** | **27.247** | **28.537** | **+1.290** |

Same skip budget for SeaCache and hybrid: **84 computed / 116 skipped**
sample-steps over the batch of 4 (21 computed / 29 skipped per 50-step
trajectory on average).

Wall-clock on B200, **batched B=4**:

| Mode | Batch wall | s / image | Speedup vs base |
|---|---:|---:|---:|
| Base (no cache) | 30.46 s | 7.61 | 1.00× |
| SeaCache | 13.03 s | 3.26 | **2.34×** |
| SeaCache + Spectrum | 13.40 s | 3.35 | **2.27×** |

Forecasting is cheap relative to a FLUX block stack (~0.09 s/image extra).
Quality improves; speed is essentially the SeaCache skip rate.

---

## 6. What we built

```
seacache_spectrum/
  spectrum_forecaster.py   Chebyshev + discrete Taylor (Spectrum port)
  util_seacache.py         SEA Wiener filter (copied from SeaCache/FLUX)
  flux_forward.py          patched FluxTransformer2DModel.forward
                           modes: base | seacache | hybrid
  compare_psnr.py          batched PSNR comparison vs base
  test_forecaster.py       CPU unit tests for the forecaster / PSNR
  prompts.txt              four evaluation prompts
  requirements.txt
  research.md              this note
```

`cached_flux_forward`:

- **base** — full stack every step (reference).
- **seacache** — SEA gate + residual copy.
- **hybrid** — same SEA gate + Spectrum residual forecast (`w=0.5`).

Per-sample skip masks so batching does not couple prompts.

---

## 7. Takeaways

- SeaCache’s frequency-aware gate is a strong *scheduler* of which
  steps to skip; the error on those steps is then “how well you
  reconstruct the skipped residual.”
- Copy-reuse is a 0th-order forecast. Spectrum’s 1st-order Taylor +
  Chebyshev mix (`w=0.5`) is a richer predictor of the same residual,
  and on these four FLUX.1-dev 1024×1024 images it is **+1.29 dB mean
  PSNR** vs SeaCache at the same skip count.
- Hyperparameters matter: `w=1.0` and mixing Chebyshev too early *lost*
  ~0.65 dB vs copy. The working recipe is Spectrum’s code defaults
  (`w=0.5`, `K=100`, `M=4`, `λ=0.1`) plus a short Taylor-only warmup.

Possible follow-ups: more prompts (DrawBench / COCO), FID/CLIP besides
PSNR, forecasting post-block hidden states as Spectrum does natively,
and sweeping δ so skip-rate vs quality is matched more carefully.

---

## 8. Full DrawBench-200 results (FLUX.1-dev, complete)

- **hybrid δ=0.3:** PSNR **27.969**, SSIM 0.9182, LPIPS 0.0699,
  3.33 s/img, **1241 TFLOPs** (`outputs_hybrid_d03/comparison_vs_base.json`)
- **hybrid δ=0.6:** PSNR **21.630**, SSIM 0.8225, LPIPS 0.1781,
  2.14 s/img, **774 TFLOPs** (`outputs_hybrid_d06/comparison_vs_base.json`)
- base 200 in `outputs_base/`. TFLOPs convention: 2976 × computed-ratio
  (SeaCache Table 1 reference). Seeds 0+i, guidance 3.5, bf16.
- vs SeaCache published: δ0.3 → 27.97 vs 26.29 (**+1.68 dB**);
  δ0.6 → 21.63 vs 21.33 (+0.30 dB) at identical 774 TFLOPs.

Qualitative base-vs-ours strips (δ=0.3): `examples_flux/flux_143_base_vs_ours.png`
(fennec, 33.79 dB), `examples_flux/flux_156_base_vs_ours.png` (statue,
32.15 dB), `examples_flux/flux_093_base_vs_ours.png` (robot, 32.43 dB).
