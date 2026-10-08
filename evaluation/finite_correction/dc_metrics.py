#!/usr/bin/env python3
"""Exact image-domain metrics used by the archived W-TransUNet inference.

The reported metrics operate on the raw LoDoPaB-CT normalized image values.
They do not clip predictions, impose positivity, apply a reconstruction mask,
or convert to Hounsfield units.  The data range is computed independently for
each ground-truth image as ``max(gt) - min(gt)``.

Torch is imported lazily so that manifest and statistics commands remain usable
in a CPU-only environment without importing the reconstruction stack.
"""

from __future__ import annotations

from typing import Any

import numpy as np


PSNR_MSE_FLOOR = 1.0e-12
DATA_RANGE_FLOOR = 1.0e-8
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03


def _as_image_batch(array: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 ``(N, 1, H, W)`` array."""

    value = np.asarray(array)
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value[:, None]
    elif value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"expected (H,W), (N,H,W), or (N,1,H,W); got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("metric inputs contain NaN or infinity")
    return np.ascontiguousarray(value, dtype=np.float32)


def numpy_psnr_rmse(prediction: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute archived PSNR and RMSE definitions in NumPy.

    NumPy accumulation is intentionally float64.  The execution runner uses
    :func:`torch_image_metrics` for the exact archived float32 implementation;
    this function is a portable canary and pure-CPU fallback for PSNR/RMSE.
    """

    pred = _as_image_batch(prediction)
    gt = _as_image_batch(ground_truth)
    if pred.shape != gt.shape:
        raise ValueError(f"metric shape mismatch: {pred.shape} != {gt.shape}")
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    mse = np.mean(diff * diff, axis=(1, 2, 3))
    flat = gt.astype(np.float64).reshape(gt.shape[0], -1)
    data_range = np.maximum(flat.max(axis=1) - flat.min(axis=1), DATA_RANGE_FLOOR)
    psnr = 20.0 * np.log10(data_range) - 10.0 * np.log10(np.maximum(mse, PSNR_MSE_FLOOR))
    rmse = np.sqrt(mse + PSNR_MSE_FLOOR)
    return psnr, rmse


def _gaussian_kernel(torch: Any, window_size: int, sigma: float, device: Any, dtype: Any) -> Any:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, window_size, window_size).contiguous()


def torch_image_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Compute the exact archived PSNR, SSIM, and RMSE definitions.

    The Gaussian SSIM implementation, reflect padding, population local
    moments, stabilizers, and float32 arithmetic match ``src/inference.py``.
    Returned arrays are float64 merely to make downstream serialization stable.
    """

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - remote runtime dependency
        raise RuntimeError("torch is required for the exact SSIM implementation") from exc

    pred_np = _as_image_batch(prediction)
    gt_np = _as_image_batch(ground_truth)
    if pred_np.shape != gt_np.shape:
        raise ValueError(f"metric shape mismatch: {pred_np.shape} != {gt_np.shape}")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pred = torch.from_numpy(pred_np).to(target_device)
    gt = torch.from_numpy(gt_np).to(target_device)

    flat = gt.view(gt.shape[0], -1)
    data_range = (flat.max(dim=1).values - flat.min(dim=1).values).clamp(min=DATA_RANGE_FLOOR)
    mse = (pred - gt).pow(2).mean(dim=(1, 2, 3))
    psnr = 20.0 * torch.log10(data_range) - 10.0 * torch.log10(mse.clamp(min=PSNR_MSE_FLOOR))
    rmse = torch.sqrt(mse + PSNR_MSE_FLOOR)

    kernel = _gaussian_kernel(torch, SSIM_WINDOW_SIZE, SSIM_SIGMA, pred.device, pred.dtype)
    pad = SSIM_WINDOW_SIZE // 2
    pred_pad = functional.pad(pred, (pad, pad, pad, pad), mode="reflect")
    gt_pad = functional.pad(gt, (pad, pad, pad, pad), mode="reflect")
    mu_x = functional.conv2d(pred_pad, kernel)
    mu_y = functional.conv2d(gt_pad, kernel)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = functional.conv2d(
        functional.pad(pred * pred, (pad, pad, pad, pad), mode="reflect"), kernel
    ) - mu_x2
    sigma_y2 = functional.conv2d(
        functional.pad(gt * gt, (pad, pad, pad, pad), mode="reflect"), kernel
    ) - mu_y2
    sigma_xy = functional.conv2d(
        functional.pad(pred * gt, (pad, pad, pad, pad), mode="reflect"), kernel
    ) - mu_xy
    sigma_x2 = torch.clamp(sigma_x2, min=0.0)
    sigma_y2 = torch.clamp(sigma_y2, min=0.0)
    dr = data_range.view(-1, 1, 1, 1)
    c1 = (SSIM_K1 * dr) ** 2
    c2 = (SSIM_K2 * dr) ** 2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim = (numerator / (denominator + 1.0e-12)).mean(dim=(1, 2, 3))

    return {
        "psnr_db": psnr.detach().cpu().numpy().astype(np.float64),
        "ssim": ssim.detach().cpu().numpy().astype(np.float64),
        "rmse": rmse.detach().cpu().numpy().astype(np.float64),
    }


def metric_definition() -> dict[str, object]:
    """Machine-readable definition embedded in run manifests."""

    return {
        "domain": "raw LoDoPaB normalized image values, central 352x352 crop",
        "postprocessing": {
            "clipping": False,
            "positivity": False,
            "mask": False,
            "hu_conversion": False,
        },
        "data_range": "per-case max(ground_truth)-min(ground_truth), floor 1e-8",
        "psnr": "20*log10(data_range)-10*log10(max(MSE,1e-12))",
        "rmse": "sqrt(MSE+1e-12)",
        "ssim": {
            "window": SSIM_WINDOW_SIZE,
            "sigma": SSIM_SIGMA,
            "padding": "reflect",
            "k1": SSIM_K1,
            "k2": SSIM_K2,
            "local_moments": "population",
            "spatial_reduction": "mean",
        },
    }
