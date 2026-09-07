"""FLUX transformer forward with SeaCache gating and optional Spectrum prediction.

Modes
-----
base:
    Full transformer every step (reference).
seacache:
    SeaCache: SEA-filtered first-block modulation decides skips; skipped
    steps *reuse* the previous residual (original SeaCache).
hybrid:
    Same SEA skip schedule, but skipped steps *forecast* the cached residual
    with Spectrum (w=0.5 Chebyshev + discrete Taylor) instead of copying it.
    Early skips (few observations) fall back to last-residual / Taylor, so
    they match SeaCache; later skips use the Spectrum blend.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

from common.util_seacache import apply_sea_with_scheduler, rel_l1
from common.spectrum_forecaster import SpectrumResidualForecaster

logger = logging.get_logger(__name__)
def _run_transformer_blocks(
    self,
    hidden_states,
    encoder_hidden_states,
    temb,
    image_rotary_emb,
    joint_attention_kwargs,
    controlnet_block_samples,
    controlnet_single_block_samples,
    controlnet_blocks_repeat,
):
    """Full dual-stream + single-stream block stack (diffusers 0.40 API)."""
    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(hs, ehs, t_emb, rotary):
                    return module(
                        hidden_states=hs,
                        encoder_hidden_states=ehs,
                        temb=t_emb,
                        image_rotary_emb=rotary,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        if controlnet_block_samples is not None:
            interval_control = int(
                np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples))
            )
            if controlnet_blocks_repeat:
                hidden_states = (
                    hidden_states
                    + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                )
            else:
                hidden_states = (
                    hidden_states + controlnet_block_samples[index_block // interval_control]
                )

    for index_block, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward_single(module):
                def custom_forward(hs, ehs, t_emb, rotary):
                    return module(
                        hidden_states=hs,
                        encoder_hidden_states=ehs,
                        temb=t_emb,
                        image_rotary_emb=rotary,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

                return custom_forward

            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward_single(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        if controlnet_single_block_samples is not None:
            interval_control = int(
                np.ceil(
                    len(self.single_transformer_blocks)
                    / len(controlnet_single_block_samples)
                )
            )
            hidden_states = (
                hidden_states + controlnet_single_block_samples[index_block // interval_control]
            )

    return encoder_hidden_states, hidden_states


def _as_bool_mask(should_calc, batch_size, device):
    if isinstance(should_calc, torch.Tensor):
        return should_calc.to(device=device, dtype=torch.bool).reshape(-1)
    if isinstance(should_calc, (list, tuple)):
        return torch.tensor(list(should_calc), dtype=torch.bool, device=device)
    return torch.ones(batch_size, dtype=torch.bool, device=device) * bool(should_calc)


def _sea_gate(self, hidden_states, temb, img_ids):
    """Per-sample SeaCache skip decision using SEA-filtered first-block modulation.

    Returns a bool tensor of shape [B]; True means run the full transformer.
    """
    batch = hidden_states.shape[0]
    modulated_inp, _, _, _, _ = self.transformer_blocks[0].norm1(hidden_states, emb=temb)

    acc = getattr(self, "accumulated_rel_l1_distance", None)
    if not isinstance(acc, list) or len(acc) != batch:
        acc = [0.0] * batch
        self.accumulated_rel_l1_distance = acc

    force_full = (
        self.cnt == 0
        or self.cnt == self.num_steps - 1
        or self.previous_modulated_input is None
    )
    if force_full:
        self.accumulated_rel_l1_distance = [0.0] * batch
        self.previous_modulated_input = modulated_inp
        return torch.ones(batch, dtype=torch.bool, device=hidden_states.device)

    h = int(img_ids[:, 1].max().item() + 1)
    w = int(img_ids[:, 2].max().item() + 1)
    filtered = modulated_inp.reshape(batch, h, w, modulated_inp.shape[-1])
    filtered = apply_sea_with_scheduler(
        filtered,
        self.scheduler,
        getattr(self, "cnt", 0),
        power_exp=2.0,
        dims=(-2, -3),
        norm_mode="mean",
    )
    filtered = filtered.reshape(batch, -1, filtered.shape[-1])

    prev = self.previous_modulated_input
    should = []
    for i in range(batch):
        dist = rel_l1(filtered[i : i + 1], prev[i : i + 1])
        acc[i] += dist
        if acc[i] < float(self.seacache_thresh):
            should.append(False)
        else:
            should.append(True)
            acc[i] = 0.0
    self.accumulated_rel_l1_distance = acc
    self.previous_modulated_input = filtered
    return torch.tensor(should, dtype=torch.bool, device=hidden_states.device)

def cached_flux_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """Drop-in FluxTransformer2DModel.forward with SeaCache / Spectrum modes."""

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    elif joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
        logger.warning(
            "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
        )

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids is not None and txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    if img_ids is not None and img_ids.ndim == 3:
        img_ids = img_ids[0]

    if txt_ids is not None and img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)
    else:
        image_rotary_emb = None

    mode = getattr(self, "cache_mode", "base")
    batch = hidden_states.shape[0]
    should_mask = torch.ones(batch, dtype=torch.bool, device=hidden_states.device)
    if mode in ("seacache", "hybrid"):
        should_mask = _as_bool_mask(
            _sea_gate(self, hidden_states, temb, img_ids), batch, hidden_states.device
        )

    step_idx = int(getattr(self, "cnt", 0))
    if mode in ("seacache", "hybrid"):
        self.cnt = step_idx + 1
        if self.cnt == self.num_steps:
            self.cnt = 0

    skip_mask = ~should_mask
    out_hidden = hidden_states
    computed_now = 0
    skipped_now = 0

    if skip_mask.any() and mode in ("seacache", "hybrid"):
        skip_idx = torch.where(skip_mask)[0]
        skipped_now = int(skip_idx.numel())
        skip_hs = hidden_states.index_select(0, skip_idx)
        if mode == "hybrid" and getattr(self, "spectrum_forecasters", None):
            predicted = []
            for j, b in enumerate(skip_idx.tolist()):
                fc = self.spectrum_forecasters[b] if b < len(self.spectrum_forecasters) else None
                if fc is not None and fc.n_obs >= 1:
                    pred_res = fc.predict(step_idx).to(dtype=skip_hs.dtype, device=skip_hs.device)
                    predicted.append(skip_hs[j : j + 1] + pred_res)
                elif self.previous_residual is not None:
                    predicted.append(skip_hs[j : j + 1] + self.previous_residual[b : b + 1])
                else:
                    predicted.append(skip_hs[j : j + 1])
            skip_out = torch.cat(predicted, dim=0)
        elif self.previous_residual is not None:
            skip_out = skip_hs + self.previous_residual.index_select(0, skip_idx)
        else:
            skip_out = skip_hs
        out_hidden = out_hidden.clone()
        out_hidden.index_copy_(0, skip_idx, skip_out)

    if should_mask.any() or mode == "base":
        calc_idx = (
            torch.where(should_mask)[0]
            if mode in ("seacache", "hybrid")
            else torch.arange(batch, device=hidden_states.device)
        )
        computed_now = int(calc_idx.numel())
        calc_hs = hidden_states.index_select(0, calc_idx)
        calc_enc = encoder_hidden_states.index_select(0, calc_idx)
        calc_temb = temb.index_select(0, calc_idx) if temb.ndim > 1 else temb
        ori = calc_hs
        calc_enc, calc_hs = _run_transformer_blocks(
            self,
            calc_hs,
            calc_enc,
            calc_temb,
            image_rotary_emb,
            joint_attention_kwargs,
            controlnet_block_samples,
            controlnet_single_block_samples,
            controlnet_blocks_repeat,
        )
        residual = calc_hs - ori
        if mode in ("seacache", "hybrid"):
            if self.previous_residual is None or self.previous_residual.shape[0] != batch:
                self.previous_residual = torch.zeros_like(hidden_states)
            self.previous_residual.index_copy_(0, calc_idx, residual)
            if mode == "hybrid":
                if not getattr(self, "spectrum_forecasters", None):
                    self.spectrum_forecasters = [None] * batch
                for j, b in enumerate(calc_idx.tolist()):
                    fc = self.spectrum_forecasters[b]
                    if fc is not None:
                        fc.update(step_idx, residual[j : j + 1])
        if mode in ("seacache", "hybrid"):
            out_hidden = out_hidden.clone() if skip_mask.any() else calc_hs.new_empty(hidden_states.shape)
            if not skip_mask.any():
                out_hidden = calc_hs
            else:
                out_hidden.index_copy_(0, calc_idx, calc_hs)
        else:
            out_hidden = calc_hs

    hidden_states = out_hidden
    self.computed_steps = getattr(self, "computed_steps", 0) + computed_now
    self.skipped_steps = getattr(self, "skipped_steps", 0) + skipped_now

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def reset_cache_state(transformer, num_steps: int, mode: str, args=None, batch_size: int = 1) -> None:
    """Reset per-sample cache / forecaster state on the transformer."""
    transformer.cnt = 0
    transformer.num_steps = int(num_steps)
    transformer.accumulated_rel_l1_distance = [0.0] * int(batch_size)
    transformer.previous_modulated_input = None
    transformer.previous_residual = None
    transformer.computed_steps = 0
    transformer.skipped_steps = 0
    transformer.cache_mode = mode
    transformer.spectrum_forecaster = None
    transformer.spectrum_forecasters = None
    if mode == "hybrid":
        m = getattr(args, "spectrum_m", 4) if args is not None else 4
        k = getattr(args, "spectrum_k", 100) if args is not None else 100
        lam = getattr(args, "spectrum_lam", 0.1) if args is not None else 0.1
        w = getattr(args, "spectrum_w", 0.5) if args is not None else 0.5
        taylor_order = getattr(args, "spectrum_taylor_order", 1) if args is not None else 1
        min_cheb_obs = getattr(args, "spectrum_min_cheb_obs", 4) if args is not None else 4
        device = transformer.device if hasattr(transformer, "device") else None
        transformer.spectrum_forecasters = [
            SpectrumResidualForecaster(
                m=m,
                k=k,
                lam=lam,
                w=w,
                taylor_order=taylor_order,
                device=device,
                t_max=float(num_steps),
                min_cheb_obs=min_cheb_obs,
            )
            for _ in range(int(batch_size))
        ]

