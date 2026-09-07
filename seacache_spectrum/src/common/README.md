# `src/common/` — shared skip gate + forecaster

Model-agnostic pieces used by the flux/wan/hunyuan patches. You never run
this folder directly; read it to understand (or retune) the method.

- `util_seacache.py` — the **skip gate** (from SeaCache). `apply_sea_with_scheduler`
  Wiener-filters the first block's modulation with the scheduler's `(a_t, b_t)`
  mixing; `rel_l1` accumulates relative-L1 vs the previous step. When the
  accumulator is below `seacache_thresh` (δ), the step is skipped.
- `spectrum_forecaster.py` — the **residual forecaster** (from Spectrum).
  `SpectrumResidualForecaster` keeps the last `K` computed `(t, residual)`
  pairs and, on a skipped step, predicts the residual with Chebyshev ridge
  regression (`M` bases, `λ`) blended with a discrete Taylor step (weight `w`).
  Pure Taylor until `min_cheb_obs` observations exist (warm-up).

Default knobs (same defaults in every patch):

| Arg | Default | Meaning |
|---|---|---|
| `seacache_thresh` (δ) | 0.3 flux / 0.2 video | skip aggressiveness: higher = faster, lower fidelity |
| `spectrum_m` | 4 | Chebyshev bases |
| `spectrum_k` | 100 | observation window |
| `spectrum_lam` | 0.1 | ridge strength |
| `spectrum_w` | 0.5 | Chebyshev vs Taylor blend |
| `spectrum_taylor_order` | 1 | warm-up stepper order |
| `spectrum_min_cheb_obs` | 4 | Taylor-only until this many observations |

Quick CPU sanity check (no weights, no GPU):

```bash
.venv/bin/python scripts/test_forecaster.py
```
