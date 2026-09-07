"""HunyuanVideoTransformer3DModel forward with SeaCache SEA gating + Spectrum forecast.

Modes: base (full), seacache (SEA schedule + residual reuse),
hybrid (SEA schedule + Spectrum residual forecast).

CFG note: HunyuanVideoPipeline calls the transformer twice per denoising
step (cond then uncond). Calls are demultiplexed by parity (call_idx % 2).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from common.util_seacache import apply_sea_with_scheduler, rel_l1
from common.spectrum_forecaster import SpectrumResidualForecaster

logger = logging.get_logger(__name__)


def _slot_state(self, slot: int) -> Dict[str, Any]:
    st = getattr(self, "hybrid_slots", None)
    if st is None or len(st) != 2:
        st = [{}, {}]
        self.hybrid_slots = st
    s = st[slot]
    if not s:
        s.update(
            {
                "cnt": 0,
                "acc": [0.0],
                "prev_mod": None,
                "prev_res": None,
                "forecasters": None,
                "computed": 0,
                "skipped": 0,
            }
        )
    return s


def _sea_gate_hunyuan(self, tokens, temb, Fp, Hp, Wp, slot: int) -> bool:
    s = _slot_state(self, slot)
    mod_out = self.transformer_blocks[0].norm1(tokens, emb=temb)
    mod_in = mod_out[0] if isinstance(mod_out, (tuple, list)) else mod_out
    B, N, C = mod_in.shape
    if Fp * Hp * Wp == N:
        filt = mod_in.reshape(B, Fp, Hp, Wp, C)
        dims = (-2, -3, -4)
    else:  # first-frame special patching: filter over token axis only
        filt = mod_in.reshape(B, 1, N, C)
        dims = (-2,)
    filt = apply_sea_with_scheduler(
        filt, self.scheduler, int(s["cnt"]),
        power_exp=3.0, dims=dims, norm_mode="mean",
    )
    filt = filt.reshape(B, -1, C)
    if s["cnt"] == 0 or s["cnt"] == self.num_steps - 1 or s["prev_mod"] is None:
        s["acc"] = [0.0] * B
        s["prev_mod"] = filt
        return True
    dist = rel_l1(filt[0:1], s["prev_mod"][0:1])
    s["acc"][0] += dist
    if s["acc"][0] < float(self.seacache_thresh):
        return False
    s["acc"][0] = 0.0
    s["prev_mod"] = filt
    return True


def cached_hunyuan_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[tuple, Transformer2DModelOutput]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    elif attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
        logger.warning(
            "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
        )

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = 1 * post_patch_height * post_patch_width

    image_rotary_emb = self.rope(hidden_states)

    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(
        encoder_hidden_states, timestep, encoder_attention_mask
    )

    latent_sequence_length = hidden_states.shape[1]
    condition_sequence_length = encoder_hidden_states.shape[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    attention_mask = torch.ones(
        batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
    )
    effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
    indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)
    mask_indices = indices >= effective_sequence_length.unsqueeze(1)
    attention_mask = attention_mask.masked_fill(mask_indices, False)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

    mode = getattr(self, "cache_mode", "base")
    # Demux CFG streams: cond+uncond calls within one denoising step share
    # the same timestep; a new (lower) timestep means a new step on slot 0.
    try:
        cur_t = float(timestep.flatten()[0].item())
    except Exception:
        cur_t = None
    last_t = getattr(self, "hybrid_last_t", None)
    if cur_t is not None and last_t is not None and cur_t == last_t:
        slot = 1
    else:
        slot = 0
    self.hybrid_last_t = cur_t
    s = _slot_state(self, slot)
    if s["forecasters"] is None and mode == "hybrid":
        s["forecasters"] = [
            SpectrumResidualForecaster(
                m=getattr(self, "spectrum_m", 4),
                k=getattr(self, "spectrum_k", 100),
                lam=getattr(self, "spectrum_lam", 0.1),
                w=getattr(self, "spectrum_w", 0.5),
                taylor_order=getattr(self, "spectrum_taylor_order", 1),
                min_cheb_obs=getattr(self, "spectrum_min_cheb_obs", 4),
                t_max=float(getattr(self, "num_steps", 50)),
            )
        ]

    should_calc = True
    step_idx = int(s["cnt"])
    if mode in ("seacache", "hybrid"):
        if batch_size != 1:
            raise ValueError(f"hybrid hunyuan forward supports B=1, got {batch_size}")
        should_calc = _sea_gate_hunyuan(
            self, hidden_states, temb,
            post_patch_num_frames, post_patch_height, post_patch_width, slot,
        )
        s["cnt"] = step_idx + 1
        if s["cnt"] == self.num_steps:
            s["cnt"] = 0

    if not should_calc:
        s["skipped"] += 1
        fc = s["forecasters"][0] if s["forecasters"] else None
        if mode == "hybrid" and fc is not None and fc.n_obs >= 1:
            pred = fc.predict(step_idx).to(dtype=hidden_states.dtype, device=hidden_states.device)
            hidden_states = hidden_states + pred
        elif s["prev_res"] is not None:
            hidden_states = hidden_states + s["prev_res"]
    else:
        s["computed"] += 1
        ori = hidden_states
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask,
                image_rotary_emb, token_replace_emb, first_frame_num_tokens,
            )
        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask,
                image_rotary_emb, token_replace_emb, first_frame_num_tokens,
            )
        residual = hidden_states - ori
        if mode in ("seacache", "hybrid"):
            s["prev_res"] = residual
            if mode == "hybrid" and s["forecasters"]:
                s["forecasters"][0].update(step_idx, residual)

    # 5. Output projection (always run)
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (hidden_states,)
    return Transformer2DModelOutput(sample=hidden_states)


def reset_hunyuan_cache_state(transformer, num_steps: int, mode: str, args=None) -> None:
    transformer.cache_mode = mode
    transformer.num_steps = int(num_steps)
    transformer.hybrid_last_t = None
    transformer.hybrid_slots = None
    transformer.seacache_thresh = float(getattr(args, "seacache_thresh", 0.19)) if args is not None else 0.19
    transformer.spectrum_m = int(getattr(args, "spectrum_m", 4)) if args is not None else 4
    transformer.spectrum_k = int(getattr(args, "spectrum_k", 100)) if args is not None else 100
    transformer.spectrum_lam = float(getattr(args, "spectrum_lam", 0.1)) if args is not None else 0.1
    transformer.spectrum_w = float(getattr(args, "spectrum_w", 0.5)) if args is not None else 0.5
    transformer.spectrum_taylor_order = int(getattr(args, "spectrum_taylor_order", 1)) if args is not None else 1
    transformer.spectrum_min_cheb_obs = int(getattr(args, "spectrum_min_cheb_obs", 4)) if args is not None else 4


def hunyuan_cache_totals(transformer) -> Dict[str, int]:
    comp = skip = 0
    for s in getattr(transformer, "hybrid_slots", None) or []:
        comp += int(s.get("computed", 0))
        skip += int(s.get("skipped", 0))
    return {"computed": comp, "skipped": skip}
