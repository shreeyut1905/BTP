# BTP — Agent Memory (progress.md)

Improvement over **SeaCache** (CVPR 2026 Oral) for a CVPR-style submission.
Method name in paper: **ReSPect** — Residual Spectral Prediction for
Efficient Caching of Transformers (code: hybrid SeaCache-gate + Spectrum
residual forecast).

## 0. Starting point
- Upstreams (untracked in git): `SeaCache/` = jiwoogit/SeaCache,
  `Spectrum/` = hanjq17/Spectrum. Papers: arXiv 2602.18993 (SeaCache),
  2603.01623 (Spectrum). Both CVPR 2026.
- Target: beat SeaCache Table 1 (FLUX.1-dev), Table 3 (HunyuanVideo),
  Table 4 (Wan2.1-1.3B) at matched TFLOPs. Baselines are
  **SeaCache-published numbers** (no baseline reruns — explicit user call).
- GPU: single NVIDIA B200, 183 GB VRAM, 2.2 TB host RAM.
  Latencies ours differ from paper (A100 / RTX PRO 6000) — compare quality
  at matched TFLOPs, never seconds across hardware.
- Python: always `seacache_spectrum/.venv/bin/python` (torch 2.8 cu128,
  diffusers 0.40, pyiqa, skimage, calflops, triton).
  Exception: `/workspace/HunyuanVideo/` official-repo code also runs in
  this venv (torch-only forecaster is shared).

## 1. The research question
SeaCache answers **when to skip** (SEA-filtered content-drift gate) but
reconstructs by **copy-reuse** (order-0). Spectrum answers **how to
reconstruct** (Chebyshev ridge + Taylor forecast of hidden states) but
under a **fixed** schedule. The hybrid hypothesis: keep SeaCache's
content-aware gate (it fires exactly on smooth trajectory segments) and
replace copy-reuse with Spectrum-style forecasting — a predictor should
beat a stale copy precisely where the gate says the trajectory is smooth.

## 2. Method anatomy (what each piece does)
- **SEA gate (unchanged from SeaCache):** first dual-stream block's
  `norm1` modulation → reshape to spatial/temporal grid → Wiener filter
  `apply_sea_with_scheduler` (power-law signal prior, scheduler mixing
  `a_t=1-σ, b_t=σ` in flow mode, mean-normalized gain) → accumulate
  relative-L1 vs previous filtered modulation → skip while accumulator
  stays below δ. First and last steps always compute. Gates are
  **per-sample** so batched prompts never couple.
- **Spectrum forecaster (`spectrum_forecaster.py`, our port of
  Spectrum `src/utils/basis_utils.py` + `__init__.py`):** per-sample
  window of ≤K=100 observed (t, residual) pairs; Chebyshev ridge
  M=4 bases, λ=0.1, time mapped to τ∈[-1,1] over fixed [0,50];
  blended `h_mix = (1-w)·h_taylor + w·h_cheb` with **w=0.5** (code
  default; the yaml's w=1.0 is NOT what Spectrum runs).
- **Hybrid rule:** on computed steps run full stack, cache residual
  `r = h_out − h_in`, update forecaster; on skipped steps predict
  `r̂` if ≥1 observation else fall back to copy. Chebyshev mixes in only
  after `min_cheb_obs=4` computed residuals (before that Taylor-only,
  which with 1 obs reduces to copy — early skips match SeaCache).
- **Key modeling choice — forecast residuals, not hidden states:**
  Spectrum forecasts post-block states; SeaCache caches residuals.
  Forecasting residuals is the drop-in replacement for copy-reuse,
  keeps norms small/centered (better ridge conditioning), and avoids the
  silent `h_in`-shift bug of mixing the two (found the hard way).

## 3. Pilot study (4 prompts, FLUX 1024², 50 steps) — where the bugs died
First hybrid was **−0.65 dB worse** than SeaCache. Four root causes,
each verified:
1. `w=1.0` (pure Chebyshev): 4th-order ridge on few points overshoots.
   Spectrum's intended mix is w=0.5 → fixed.
2. `K=32` instead of Spectrum's `K=100` → fixed.
3. Chebyshev blended from the **first** observation → polluted early
   skips; added `min_cheb_obs=4` warm-up → fixed.
4. Forecasting hidden states as if residuals (shape/semantics mismatch)
   → forecast residuals → fixed.
After fixes, hybrid beat SeaCache on **all four** pilot prompts:

| prompt | SeaCache | Hybrid | Δ |
|---|---|---|---|
| panda in cafe | 27.479 | 28.228 | +0.749 |
| astronaut on moon | 24.375 | 25.657 | +1.282 |
| London bus | 26.732 | 28.244 | +1.512 |
| retriever/sunflowers | 30.401 | 32.019 | +1.618 |
| **mean** | **27.247** | **28.537** | **+1.290** |

Same skip budget (84 computed / 116 skipped sample-steps); forecast
overhead ≈0.09 s/image (SeaCache 3.26 vs hybrid 3.35 s/img on B200).
Documented in `seacache_spectrum/research.md`.

## 4. Full FLUX results (DrawBench 200, committed)
- hybrid δ=0.3: PSNR **27.969**, SSIM 0.9182, LPIPS 0.0699,
  3.33 s/img, **1241 TFLOPs** (`outputs_hybrid_d03/comparison_vs_base.json`)
- hybrid δ=0.6: PSNR **21.630**, SSIM 0.8225, LPIPS 0.1781,
  2.14 s/img, **774 TFLOPs** (`outputs_hybrid_d06/`)
- base 200 in `outputs_base/`. TFLOPs convention: 2976 × computed-ratio
  (SeaCache Table 1 reference). Seeds 0+i, guidance 3.5, bf16.
- vs SeaCache published: δ0.3 → 27.97 vs 26.29 (**+1.68 dB**);
  δ0.6 → 21.63 vs 21.33 (+0.30 dB) at identical 774 TFLOPs.

## 5. Video work
### 5a. Wan2.1-1.3B (diffusers path, fast) — DONE, full VBench-946
- `wan_hybrid_forward.py`: SEA gate on first-block modulation
  (`scale_shift_table` + temb chunk, grid from post-patch dims,
  power_exp=3.0, dims=(-4,-3,-2)) + residual forecast. CFG cond/uncond
  calls demuxed by **timestep equality** (parity is wrong when guidance
  is embedded). Smoke: PSNR 37.05 @10 steps; full-config base 65f = 56.7 s.
- `compare_video.py`: base vs hybrid, frame-averaged PSNR/SSIM/LPIPS,
  MP4 outputs, base-frame PNG save/reuse across thresholds,
  `--compile` (per-block, dynamic=False), `--attn cudnn|flash|auto`.
- Config: 480×832×65f, 50 steps, guidance 5.0, δ∈{0.2, 0.35}.
- FINAL (946/946, `outputs_video_wan_d02/comparison.json`,
  `outputs_video_wan_d035/comparison.json`):
  - δ=0.2: PSNR **28.382**, SSIM 0.9054, LPIPS 0.0656, **4302.9 TFLOPs**
    (vs SeaCache-published 26.60 @3942 → **+1.78 dB**)
  - δ=0.35: PSNR **26.700**, SSIM 0.8838, LPIPS 0.0841, **3335.6 TFLOPs**
    (vs SeaCache-published 21.78 @2793 → **+4.92 dB**, LPIPS halved)
  - Assembly: exact tails re-ran (d02 65/65 contig, 29.453 dB;
    d035 124/124 contig idx 822–945) + `/tmp/finalize_wan.py` phases A/B/C;
    LPIPS batched on GPU via pyiqa (1703 jobs). 881/822 hybrid rows came
    from MP4-decoded frames → ~0.1 dB codec caveat, footnoted in paper.
  - TFLOPs ref 8214, per-vid computed/full-calls convention.
- Example videos pushed to GitHub: `seacache_spectrum/examples_video/`
  (0834 ocean / 0852 pool / 0860 waterfall, base+hybrid δ0.35 pairs —
  same prompts as paper Fig. 5); full metrics in
  `seacache_spectrum/results/wan_d02_comparison.json`,
  `wan_d035_comparison.json` (+ FLUX `flux_d03/d06_comparison.json`).

### 5b. HunyuanVideo (official Tencent repo — user correction applied)
- Initial diffusers path (`hunyuanvideo-community` weights) was
  **~7.5 s/it** vs paper's 3.65 s/it: root cause = per-block bool
  attention mask kills flash ATTN (falls to slow kernels).
  User correctly redirected to the official repo (flash varlen).
- `/workspace/HunyuanVideo/` (clone, not in git) + `hybrid_generate.py`:
  SeaCache-forward clone with forecast-on-skip. Compat shims for
  transformers≥5 (LLM extractor, CLIP norm, Llama tokenizer_class).
  Smoke (SDPA torch mode): PSNR 33.02. Timings: 544p65f base 342 s;
  **480p65f base 341.8 s?? no — 480p run showed 3.39 s/it** (~170 s +
  decode). Paper config = 480×832×65f, embedded guidance 6.0,
  δ∈{0.19, 0.35}.
- Weights in `ckpts/`: transformer bf16+fp8, VAE, CLIP, LLaMA (via fixed
  `preprocess_llava.py`; tokenizer patch backed up as `*.bak5x`).
- FA4 detour (user-supplied cu130 stack in `/workspace/HunyuanVideo/.venv`):
  installed but API-mismatched (`flash_attn.cute` vs `flash_attn_interface`,
  cutlass 4.5.2 vs 4.6.2 — fixed by upgrading to 4.6.2) and then
  **reverted per user** ("skip FA4"). venv dormant. flash-attn v2.6.3
  source-build attempt killed mid-way. Do not resurrect without asking.
- LTX-2 discussion outcome: rejected — loses published SeaCache baseline,
  restarts integration, weaker story on distilled few-step models.

## 6. Paper (root `paper.pdf` + `supplementary.pdf`; source in
`/tmp/paper_respect/main.tex` + `supp.tex` + `app_body.tex` + `figs/` —
source deliberately outside git, only compiled PDFs tracked)
- Name evolution: ForeCache → **ReSPect** (user: less AI-looking, more math).
- Restyled Spectrum-like: 7pp 2-col main, teaser-first
  (`teaser_compact.png` banner + TikZ schedule-vs-reconstruct schematic),
  6pp single-col supplement (`xr` cross-refs, zero `??` floats fixed).
- Spine: predictor hierarchy (reuse Lipschitz bound Eq.5, Taylor remainder,
  Chebyshev uniform bound Eq.7, ridge closed form + stability Lemma 1),
  residual-space error identity (Eq.9) + ‖H‖_F conditioning argument,
  bias–variance blend optimum w⋆, degrees-of-freedom warm-up,
  per-sample gating + complexity. Ablations written as tests of theory.
- FLUX Tab. 1 filled (SeaCache-published baselines + ours, now
  `\scriptsize` + tabcolsep 3pt per user). Video Tab. 2: Wan rows complete,
  Hunyuan rows TBD (running). Qual figures are pairs-only (no reuse
  triplets, per user); FLUX strips rebuilt from original PNGs with
  DejaVuSans-Bold 30pt headers/captions (`--rebuild_strips`).
- Figures (all generated CPU-only by
  `seacache_spectrum/scripts/make_figures.py --out /tmp/paper_respect/figs`,
  pdflatex needs `texlive-pictures` for TikZ):
  `fig2_quality_compute.pdf`, `fig3_qual.png`, `fig_video_frames.png`
  (ocean 834 / waterfall 860 / pool 852 × 4 frames, base vs δ0.35),
  `fig4_ablation.pdf`, `figA_skip.pdf`, `figA_ssim_lpips.pdf`,
  `figA_qual.png`, `teaser_compact.png`.
- CLIP (ViT-B/32) + ImageReward measured on FLUX (base 31.66 / d03 31.69 /
  d06 31.59; IR −1.688/−1.691/−1.696, corr ≥0.995) but LEFT OUT of paper
  per user ("skip for now"). ImageReward ran from a `/tmp/ImageReward`
  source clone + 4 transformers-5 BLIP shims + `fairscale==0.4.13`.

## 7. Running now → pending (updated Sep 8)
- RUNNING Hunyuan base resume: `/workspace/HunyuanVideo/hybrid_generate.py
  --offset 112 --num_prompts 834 --modes base` → same
  `outputs_hunyuan_480p_d019/` dir (112 base videos already there),
  480×832×65f, 50 steps, guidance 6.0, seed 42+idx,
  `--compile --compile_mode max-autotune-no-cudagraphs --save_base_frames`.
  Log `seacache_spectrum/logs/hunyuan_480p_base_resume.log`, launcher
  `logs/run_hunyuan_resume.sh`. ~3.5 min/video → ~2 days for 834.
  (First try with `max-autotune` died on cudagraph error — sampler holds
  block outputs live across steps; fallback was pre-planned. See §10.)
- QUEUED after base: hybrid δ=0.19 (`--reuse_base` same dir) → hybrid
  δ=0.35 into `outputs_hunyuan_480p_d035/`.
- THEN: fill paper Tab. 2 Hunyuan rows; optional VBench-Quality/CycleReward.
- Commit hygiene: push only `seacache_spectrum/` + docs + `paper.pdf` +
  `supplementary.pdf` + `progress.md` (user explicitly requested
  progress.md tracked despite AGENTS.md). Never `SeaCache/`, `Spectrum/`,
  outputs, logs, venvs, `.inductor_cache/`. Never `git add -A` (untracked
  clones aren't gitignored) — stage explicit paths, `git add -f` for
  gitignored-but-wanted files (examples, results, progress.md).
- Git auth: if `No anonymous write access`, retry push with
  `VSCODE_GIT_IPC_HANDLE=` set to each live `/tmp/vscode-git-*.sock`
  (working one: `/tmp/vscode-git-63c5118d50.sock`).
- Repo restructured to `src/{common,flux,wan,hunyuan}/` + `scripts/` +
  `prompts/`; per-folder READMEs (`src/README.md`,
  `src/{common,flux,wan,hunyuan}/README.md`) with verified snippets.
  `AGENTS.md` created at root with all operating rules.

## 8. Gotchas (do not relearn)
- `enable_model_cpu_offload()` silently bypasses `__class__.forward`
  patches (zero counters). Use `pipe.to("cuda")` (Wan 19 GB, Hunyuan 62 GB).
- CFG demux by call parity is wrong for embedded-guidance models;
  demux by timestep equality (both video forwards do this).
- Single-use `sdpa_kernel()` contexts: fresh one per generation.
- torch.compile per-block, dynamic=False (single resolution).
- Never `pkill -f` with a literal pattern present in your own command
  line (self-kill twice so far). Kill GPU strays by PID from nvidia-smi.
- Long jobs: `setsid nohup … & disown`, poll logs only
  (`tail -c 600 log | tr '\r' '\n' | tail`).
- Forecaster params everywhere: m=4, k=100, λ=0.1, w=0.5,
  taylor_order=1, min_cheb_obs=4. Seeds: FLUX 0+i, video 42+i.
- `hybrid_generate.py` rows now record `"seed": 42+idx` per prompt and the
  summary records `seed_base` + `seed_formula` (added Sep 8 — previously
  seeds were only implicit; user asked for an explicit track record).
- Hunyuan base resume MUST use the official `/workspace/HunyuanVideo/`
  path (same pipeline/seeds/shapes as the first 112) — never mix with the
  diffusers `eval_video.py` Hunyuan branch mid-stream.
- `logs/run_hunyuan_full.sh` points at a stale `vbench_prompts.txt`
  (moved to `prompts/vbench946.txt`); slug order verified 112/112
  identical, resume uses the correct path with `--offset 112`.
- `torch.compile` mode `max-autotune` (with cudagraphs) crashes Hunyuan's
  sampler (`accessing tensor output of CUDAGraphs overwritten by a
  subsequent run`); use `max-autotune-no-cudagraphs`. Keep
  `TORCHINDUCTOR_CACHE_DIR` on workspace disk, not `/tmp` (d035 died on
  full `/tmp/torchinductor_root` once).
- After the repo restructure, `/workspace/HunyuanVideo/hybrid_generate.py`
  imports went stale (`spectrum_forecaster`, `compare_psnr`,
  `util_seacache` top-level); repointed to `common.*` + `scripts/eval_flux`
  (signatures verified). That file lives OUTSIDE git — runner-only edits.

## 9. Ablations (all vs copy-reuse, fixed 20-prompt subsets; §4.3/§5.4
written honest, not as grid-search wins)
- FLUX δ=0.3 (reuse 28.350 dB): w0 +1.79, w0.5 (paper setting) +0.65,
  w1 −0.42, warm-up 8 +0.66 / 12 +1.51 / 16 +1.77 / 24 +1.79.
- Wan δ=0.35 (reuse 28.888 dB): w0 +0.60, w0.5 +0.13, w1 −1.61.
- K=100 vs K=32: +0.02 dB (buffers rarely exceed ~40 obs — K is cheap
  insurance). Runs in `outputs_abl_*/`, `outputs_ablwan_*/`.

## 10. Hunyuan campaign log (official Tencent path, 480×832×65f)
- Sep 4: base+hybrid δ0.19 launched (`run_hunyuan_full.sh`); base reached
  112/946 (~345 s/video) then SIGTERM (rc=143). δ0.35 hybrid died on
  prompt 0: disk-full `/tmp/torchinductor_root` in `compile_wrapper`.
- Sep 8: resumed base at `--offset 112` (slugs verified 112/112 vs
  `prompts/vbench946.txt`; seeds continue 42+idx). Added `--compile_mode`
  (`max-autotune` → crashed on cudagraphs → `max-autotune-no-cudagraphs`,
  now denoising cleanly ~1.55 s/step). Added fwd/skip counters + TFLOPs
  (ref 14038) + per-prompt seed records to the script. Runner edits are
  local-only (`/workspace/HunyuanVideo/` not in git).
- SeaCache-reported targets to beat: 32.39 dB @6747 (δ0.19),
  26.46 dB @4598 (δ0.35). Reference clones: SeaCache example 720×1280×33f
  (no compile); Spectrum config 544×960×61f (no compile). Our 480×832×65f
  is the throughput compromise for 946 prompts — documented, not claimed
  as a paper match.

## 11. GitHub artifacts (pushed Sep 8)
- `seacache_spectrum/examples_video/`: 3 base+hybrid δ0.35 pairs
  (0834 ocean, 0852 pool, 0860 waterfall — paper Fig. 5 prompts) + README.
- `seacache_spectrum/results/`: `wan_d02/d035_comparison.json`,
  `flux_d03/d06_comparison.json` (full per-prompt rows).
- `seacache_spectrum/examples_flux/`: 3 rebuilt FLUX strips (large type).
- Root: `paper.pdf`, `supplementary.pdf`, `progress.md` (this file —
  user-requested tracked despite AGENTS.md default).
