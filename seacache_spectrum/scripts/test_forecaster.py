#!/usr/bin/env python3
"""Sanity checks for Spectrum residual forecasting (no GPU / no FLUX weights)."""

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)
from common.spectrum_forecaster import SpectrumResidualForecaster  # noqa: E402
from eval_flux import psnr  # noqa: E402
from PIL import Image


def test_forecaster_reproduces_observed():
    torch.manual_seed(0)
    # Low-degree polynomial trajectory in a small feature space so Chebyshev
    # interpolation (M=3, 8 observations) is well-conditioned.
    feat = torch.randn(4, dtype=torch.float32)
    f = SpectrumResidualForecaster(m=3, k=16, lam=1e-4, w=1.0, taylor_order=1, t_max=8)
    series = []
    for t in range(8):
        h = feat * (1.0 + 0.05 * t + 0.002 * t * t)
        series.append(h.to(torch.bfloat16))
        f.update(t, series[-1])
    pred = f.predict(4).float()
    ref = series[4].float()
    rel = (pred - ref).abs().mean() / (ref.abs().mean() + 1e-8)
    assert rel.item() < 0.15, f"interpolation rel-l1 too high: {rel.item()}"
    print(f"ok interpolation rel-l1={rel.item():.4f}")


def test_forecaster_extrapolates_smooth():
    torch.manual_seed(1)
    t = torch.arange(6, dtype=torch.float32)
    # Smooth linear residual trajectory.
    base = torch.randn(2, 16)
    series = [base + 0.1 * i * torch.ones_like(base) for i in range(6)]
    f = SpectrumResidualForecaster(m=2, k=16, lam=0.1, w=0.5, taylor_order=1, t_max=10)
    for i, h in enumerate(series):
        f.update(float(i), h.to(torch.bfloat16))
    pred = f.predict(6).float()
    assert pred.shape == series[0].shape
    assert torch.isfinite(pred).all()
    print(f"ok extrapolation shape={tuple(pred.shape)} mean={pred.mean().item():.4f}")


def test_psnr_identical():
    arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    assert psnr(img, img) >= 99.0 - 1e-6
    noisy = Image.fromarray(np.clip(arr.astype(np.int16) + 8, 0, 255).astype(np.uint8))
    val = psnr(img, noisy)
    assert 20 < val < 50, val
    print(f"ok psnr identical=99, noisy={val:.2f}")


if __name__ == "__main__":
    test_forecaster_reproduces_observed()
    test_forecaster_extrapolates_smooth()
    test_psnr_identical()
    print("all tests passed")
