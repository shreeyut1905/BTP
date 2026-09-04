# FLUX.1-dev PSNR comparison (1024x1024)

PSNR is computed against the **base** (uncached) FLUX.1-dev image.

- Model: `black-forest-labs/FLUX.1-dev`
- Resolution: 1024x1024
- Steps: 50
- Guidance: 3.5
- Seed: 0
- SeaCache threshold: 0.3
- Spectrum: m=4, w=0.5, lam=0.1, taylor_order=1

| # | Prompt | SeaCache PSNR | Hybrid PSNR | Delta (hybrid-seacache) | SeaCache sample-forwards | Hybrid sample-forwards | Base s/img | SeaCache s/img | Hybrid s/img |
|---|--------|---------------|-------------|-------------------------|--------------------------|------------------------|------------|----------------|---------------|
| 0 | a high-resolution photo of a panda drinking coffee in a cozy cafe | 27.479 | 28.228 | +0.749 | 84 | 84 | 7.61 | 3.26 | 3.35 |
| 1 | a photo of an astronaut riding a horse on the moon | 24.375 | 25.657 | +1.282 | 84 | 84 | 7.61 | 3.26 | 3.35 |
| 2 | a red double-decker bus driving through a rainy London street at ni... | 26.732 | 28.244 | +1.512 | 84 | 84 | 7.61 | 3.26 | 3.35 |
| 3 | a golden retriever sitting in a field of sunflowers under a bright ... | 30.401 | 32.019 | +1.618 | 84 | 84 | 7.61 | 3.26 | 3.35 |

## Averages

- Mean SeaCache PSNR: **27.247 dB**
- Mean Hybrid (SeaCache+Spectrum) PSNR: **28.537 dB**
- Mean PSNR delta (hybrid - seacache): **+1.290 dB**
- Mean SeaCache speedup vs base: **2.34x**
- Mean Hybrid speedup vs base: **2.27x**

