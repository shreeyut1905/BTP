# `src/` — ReSPect patches

Drop-in transformer forwards that add training-free caching. One folder per
model, plus `common/` shared by all three.

| Folder | Model | Patch entrypoint |
|---|---|---|
| `common/` | (shared) | SEA skip gate + residual forecaster |
| `flux/` | FLUX.1-dev 1024×1024 | `flux_forward.cached_flux_forward` |
| `wan/` | Wan2.1-1.3B 480p×65f | `wan_forward.cached_wan_forward` |
| `hunyuan/` | HunyuanVideo 480p×65f | `hunyuan_forward.cached_hunyuan_forward` |

All scripts import from here with:

```python
import sys; sys.path.insert(0, "src")   # run from seacache_spectrum/
```

Modes (same everywhere): `base` = full compute (reference),
`seacache` = skip gate + copy last residual, `hybrid` = skip gate + forecast
residual (the ReSPect method).

Setup once (from `seacache_spectrum/`):

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/huggingface-cli login   # FLUX.1-dev is gated
```
