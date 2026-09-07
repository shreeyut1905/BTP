# AGENTS.md — BTP / ReSPect

Technique repo: training-free diffusion cache (SeaCache skip gate + Spectrum
residual forecast). Read `README.md`, `seacache_spectrum/README.md`,
`seacache_spectrum/research.md` before changing method code.

## Layout (what matters)

- `seacache_spectrum/` is the entire project. Everything else at root is support.
- `seacache_spectrum/src/{common,flux,wan,hunyuan}/` — patches; `common/` is
  shared (SEA filter + forecaster). Scripts do `sys.path.insert(0, "src")` and
  import `common.*`, `flux.*`, `wan.*` — keep those package names.
- `seacache_spectrum/scripts/` — `eval_flux.py`, `eval_video.py`,
  `compute_metrics.py` (CPU-only image metrics), `test_forecaster.py`.
- `SeaCache/`, `Spectrum/` are **untracked upstream clones** (reference only).
  Never edit, never stage, never push them. They are NOT in `.gitignore`, so
  never `git add -A` — stage explicit paths only. Same for `progress.md`
  (local session memory, untracked, stays out of git).

## Python / GPU

- Only interpreter: `seacache_spectrum/.venv/bin/python` (torch 2.14.0+cu130).
  Never system python. Reinstalls: `uv pip install --python .venv/bin/python`.
- Single NVIDIA B200. One GPU job at a time — check `ps aux | grep eval_` and
  `tail` the relevant `seacache_spectrum/logs/*.log` before launching anything.
- Fast CPU check after touching `common/`:
  `seacache_spectrum/.venv/bin/python seacache_spectrum/scripts/test_forecaster.py`
  (CPU-only, no weights).

## Eval rules (learned the hard way)

- Compare quality at **matched TFLOPs, never wall seconds** (hardware differs
  from published baselines). Refs: FLUX 2976, Wan 8214, Hunyuan 14038.
- Video runs are long and get SIGKILLed: always use `--save_base_frames` on
  base runs, `--reuse_base <base_dir>` for delta configs, and
  `--offset N --num_prompts M` to resume — never redo finished prompts.
- Base reference = decoded base **PNG frames** (`*_base_f*.png`), not MP4s.
  Metrics off MP4-decoded hybrids carry a ~0.1 dB codec caveat — note it.
- Keep the model on CUDA: cpu-offload bypasses the patched forward.
  `torch.compile` per-block with `dynamic=False`; fresh `sdpa_kernel(...)`
  context per generation; Wan CFG streams demux by timestep equality.

## Git

- Branch `main`, remote `github.com/shreeyut1905/BTP.git`. Push only
  `seacache_spectrum/` + docs; outputs/logs/media are gitignored.
- If `git push` fails with `No anonymous write access`, the shell's
  `VSCODE_GIT_IPC_HANDLE` is a stale VS Code socket — retry the push with
  `VSCODE_GIT_IPC_HANDLE=` set to each live `/tmp/vscode-git-*.sock` in turn.
