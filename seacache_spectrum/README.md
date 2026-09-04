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
