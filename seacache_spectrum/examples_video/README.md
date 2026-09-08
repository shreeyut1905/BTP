# Wan2.1 example videos (VBench, 480x832x65f, 50 steps)

Base vs ReSPect hybrid (δ=0.35) pairs — same prompts/frames as Fig. 5
(`fig_video_frames.png`) in the paper:

- `0834-ocean_*` — person swimming in ocean
- `0852-indoor_swimming_pool_*` — indoor pool splash
- `0860-waterfall_*` — waterfall

`*_base.mp4` = uncached reference (seed 42+idx). `*_hybrid.mp4` = ReSPect
(SEA gate + Spectrum residual forecast, w=0.5, K=100). Full per-prompt
metrics in `../results/wan_d035_comparison.json`.
