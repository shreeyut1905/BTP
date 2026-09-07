# SeaCache + Spectrum (FLUX.1-dev)

Hybrid caching for FLUX.1-dev:

- **SeaCache** supplies the *frequency / SEA* skip schedule: first-block
  modulation is Wiener-filtered with scheduler `(a_t, b_t)`, and a step is
  skipped when accumulated relative L1 stays below `seacache_thresh`.
- **Spectrum** replaces residual *reuse* on skipped steps: Chebyshev ridge
  regression (blended with a discrete Taylor term, **`w = 0.5`**) *forecasts*
  the transformer residual instead of copying the last cached residual.

See **[research.md](research.md)** for the method, bugs we hit, and the
1024×1024 PSNR table vs base FLUX.1-dev.

## DrawBench-200 results (FLUX.1-dev, 1024×1024, 50 steps, complete)

| Config | PSNR↑ | SSIM↑ | LPIPS↓ | TFLOPs | s/img (B200) |
|---|---|---|---|---|---|
| Hybrid δ=0.3 | 27.969 | 0.9182 | 0.0699 | 1241 | 3.33 |
| Hybrid δ=0.6 | 21.630 | 0.8225 | 0.1781 | 774 | 2.14 |

vs SeaCache-published at matched compute: δ0.3 → 27.97 vs 26.29 (**+1.68 dB**);
δ0.6 → 21.63 vs 21.33 (+0.30 dB) at identical 774 TFLOPs.
Details: `outputs_hybrid_d03/comparison_vs_base.json`,
`outputs_hybrid_d06/comparison_vs_base.json` (outputs/ dirs are gitignored;
numbers reproduced here for the record).

## Qualitative examples (base vs ours, δ=0.3)

![flux 143 base vs ours](examples_flux/flux_143_base_vs_ours.png)
*143 — baby fennec macro, PSNR 33.79, SSIM 0.971.*

![flux 156 base vs ours](examples_flux/flux_156_base_vs_ours.png)
*156 — Greek statue, PSNR 32.15, SSIM 0.981.*

![flux 093 base vs ours](examples_flux/flux_093_base_vs_ours.png)
*93 — robot, PSNR 32.43, SSIM 0.985.*

## 1024x1024 PSNR comparison

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

Outputs:

- `outputs/images/{id}_{base,seacache,hybrid}.png`
- `outputs/{id}_grid.png` — side-by-side strip
- `outputs/comparison.md` / `comparison.json`
