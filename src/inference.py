"""Later-stream inference for existing directional/repeated-FBP control checkpoints.

Machine paths are explicit. This metric stream is distinct from the historical
five-view evaluator in evaluate_cached.py.
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import os
import numpy as np
import math
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F

# headless png
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dival.config import set_config
# LoDoPaB location: set LODOPAB_PATH to point DIVal at your copy of the dataset
# P3 (Phase 3): honour LODOPAB_DATA first (the variable this study's run
# scripts export), then the README's LODOPAB_PATH, then the original
# explicit empty value; the entry point checks that the path was supplied.
LODOPAB_PATH = os.environ.get(
    'LODOPAB_DATA',
    os.environ.get('LODOPAB_PATH',
                   ''))
if LODOPAB_PATH:
    set_config('lodopab_dataset/data_path', LODOPAB_PATH)

from dival import get_standard_dataset

# --- import shim (import mechanics only; the networks are byte-identical) ----
# U-Net: the architecture now ships as src/unet.py (an unmodified copy of
# DIVal's dival/reconstructors/networks/unet.py), so it can be built without
# dragging in ODL/ASTRA through dival/__init__.py. DIVal remains the fallback.
try:
    from unet import get_unet_model
except ImportError:
    try:
        from dival.reconstructors.networks.unet import get_unet_model  # DiVAL style
    except Exception:
        try:
            from dival.networks.unet import get_unet_model
        except Exception as e:
            raise ImportError(
                "Cannot import get_unet_model from src/unet.py or from DiVAL. "
                "Please check your DiVAL version / module path."
            ) from e

# TransUNet: `dival.networks` is a directory that only ever existed in the
# author's working copy and is in no released version of DIVal, so this import
# failed for anyone else. The same files now ship in `src/`.
try:
    from vit_seg_modeling import VisionTransformer as ViT_seg
    from vit_seg_modeling import CONFIGS as CFG_ViT
except ImportError:
    from dival.networks.vit_seg_modeling import VisionTransformer as ViT_seg
    from dival.networks.vit_seg_modeling import CONFIGS as CFG_ViT

# Wavelet front end: single shared implementation (see src/wavelet_ops.py).
from wavelet_ops import (_upsample_like, haar_dwt_hvd, make_norm, ResBlock,
                         WavMixResNet)
# ---------------------------------------------------------------------------


# -----------------------------
# Args
# -----------------------------
def get_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument('--angle', type=int, required=True)
    p.add_argument('--cache_root', type=str,
                   default='./cache',
                   help='root folder containing "{angle}angle/cache_lodopab_test_fbp.npy"')
    p.add_argument('--cache_fbp', type=str, default='',
                   help='optional: direct path to cache_lodopab_test_fbp.npy (overrides cache_root)')

    # checkpoints
    p.add_argument('--ckpt_unet', type=str,
                   required=True)
    p.add_argument('--ckpt_transunet', type=str,
                   required=True)
    p.add_argument('--ckpt_wavres', type=str, required=True,
                   help='WavResTransUNet checkpoint (best_model.pth or epoch_xxx.pth)')

    # model build
    p.add_argument('--pretrained_npz', type=str,
                   default='')

    p.add_argument('--img_size', type=int, default=352)
    p.add_argument('--batch', type=int, default=16)

    # wavres options (must match training)
    p.add_argument('--wav_base_ch', type=int, default=64)
    p.add_argument('--wav_blocks', type=int, default=8)
    p.add_argument('--wav_norm', type=str, default='gn', choices=['none', 'bn', 'in', 'gn'])
    p.add_argument('--wav_upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    p.add_argument('--residual_out', action='store_true')
    # P11 (Phase 3, G1 gate): null control for the wavelet side channels.
    p.add_argument('--wav_null', action='store_true',
                   help='G1 control: replace H/V/D channels with copies of the FBP input')

    # UNet options (default: use sigmoid + use norm)
    p.add_argument('--unet_scales', type=int, default=5)
    p.add_argument('--unet_skip', type=int, default=4)
    p.add_argument('--unet_no_sigmoid', action='store_true',
                   help='if set, disable sigmoid at UNet output')
    p.add_argument('--unet_no_norm', action='store_true',
                   help='if set, disable normalization layers in UNet blocks')

    # output
    p.add_argument('--out_dir', type=str, default='./test_compare_out')
    p.add_argument('--dpi', type=int, default=150)

    # png saving control
    # P12 (Phase 3): decouple raw-NPY dumping from PNG rendering. R2b needs the
    # NPY for every test sample, but the PNG/5-panel figures only for a few.
    p.add_argument('--save_npy_all', action='store_true',
                   help='save raw NPY for ALL test samples (PNG still limited by png_n/save_all_png)')
    p.add_argument('--save_all_png', action='store_true',
                   help='save png for ALL test samples')
    p.add_argument('--png_n', type=int, default=50,
                   help='if not --save_all_png, save first N pngs (default 50)')
    p.add_argument('--png_indices', type=str, default='',
                   help='comma-separated indices to save, e.g. "0,5,10". Overrides png_n/save_all_png.')

    return p.parse_args()


# -----------------------------
# Device
# -----------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# -----------------------------
# crop helper
# -----------------------------
def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h0, w0 = img.shape
    dh, dw = (h0 - size) // 2, (w0 - size) // 2
    return img[dh:dh + size, dw:dw + size]


# -----------------------------
# Metrics (PSNR / RMSE / SSIM) in torch
#   - data_range: per-image (gt.max - gt.min) like skimage default for float
# -----------------------------
_GAUSS_CACHE = {}

def _get_gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (window_size, float(sigma), device.type, device.index, str(dtype))
    if key in _GAUSS_CACHE:
        return _GAUSS_CACHE[key]

    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :])
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.view(1, 1, window_size, window_size).contiguous()
    _GAUSS_CACHE[key] = kernel
    return kernel

def _data_range_from_gt(gt: torch.Tensor) -> torch.Tensor:
    """
    gt: (B,1,H,W)
    returns data_range: (B,) = max-min per image, clamped
    """
    flat = gt.view(gt.shape[0], -1)
    dr = flat.max(dim=1).values - flat.min(dim=1).values
    return torch.clamp(dr, min=1e-8)

def psnr_torch(pred: torch.Tensor, gt: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
    """
    pred/gt: (B,1,H,W)
    data_range: (B,)
    returns: (B,) PSNR in dB
    """
    mse = (pred - gt).pow(2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
    psnr = 20.0 * torch.log10(data_range) - 10.0 * torch.log10(mse)
    return psnr

def rmse_torch(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = (pred - gt).pow(2).mean(dim=(1, 2, 3))
    return torch.sqrt(mse + 1e-12)

def ssim_torch(pred: torch.Tensor,
               gt: torch.Tensor,
               data_range: torch.Tensor,
               window_size: int = 11,
               sigma: float = 1.5,
               k1: float = 0.01,
               k2: float = 0.03) -> torch.Tensor:
    """
    pred/gt: (B,1,H,W)
    data_range: (B,)  (per-image max-min)
    returns: (B,) SSIM
    """
    device, dtype = pred.device, pred.dtype
    kernel = _get_gaussian_kernel(window_size, sigma, device, dtype)
    pad = window_size // 2

    # reflect padding (closer to common SSIM implementations)
    pred_pad = F.pad(pred, (pad, pad, pad, pad), mode='reflect')
    gt_pad   = F.pad(gt,   (pad, pad, pad, pad), mode='reflect')

    mu_x = F.conv2d(pred_pad, kernel)
    mu_y = F.conv2d(gt_pad, kernel)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    pred2_pad = F.pad(pred * pred, (pad, pad, pad, pad), mode='reflect')
    gt2_pad   = F.pad(gt   * gt,   (pad, pad, pad, pad), mode='reflect')
    xy_pad    = F.pad(pred * gt,   (pad, pad, pad, pad), mode='reflect')

    sigma_x2 = F.conv2d(pred2_pad, kernel) - mu_x2
    sigma_y2 = F.conv2d(gt2_pad,   kernel) - mu_y2
    sigma_xy = F.conv2d(xy_pad,    kernel) - mu_xy

    # numeric stability
    sigma_x2 = torch.clamp(sigma_x2, min=0.0)
    sigma_y2 = torch.clamp(sigma_y2, min=0.0)

    dr = data_range.view(-1, 1, 1, 1)
    c1 = (k1 * dr) ** 2
    c2 = (k2 * dr) ** 2

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + 1e-12)

    return ssim_map.mean(dim=(1, 2, 3))


# -----------------------------
# Wavelet front end (Haar DWT details + WavResNet mixture CNN)
#   moved verbatim to src/wavelet_ops.py and imported at the top of this
#   file, so training and inference execute the same code instead of two
#   copies of it. Module and parameter names are unchanged, so existing
#   checkpoints still load with strict=True.
# -----------------------------

# -----------------------------
# TransUNet build (match training)
# -----------------------------
def build_transunet(img_size: int, pretrained_npz: str) -> nn.Module:
    cfg = CFG_ViT['R50-ViT-B_16']
    cfg.pretrained_path = pretrained_npz
    cfg.n_classes = 1
    cfg.n_skip = 3
    grid = img_size // cfg.patch_size
    cfg.patches.grid = (grid, grid)

    net = ViT_seg(cfg, img_size=img_size, num_classes=1)
    # pretrain load (will be overridden by ckpt anyway)
    net.load_from(np.load(cfg.pretrained_path))
    return net


# -----------------------------
# Full model: WavResTransUNet
# -----------------------------
class WavResTransUNet(nn.Module):
    def __init__(self,
                 img_size: int,
                 pretrained_npz: str,
                 wav_base_ch: int = 64,
                 wav_blocks: int = 8,
                 wav_norm: str = 'gn',
                 wav_upsample: str = 'bilinear',
                 residual_out: bool = False,
                 wav_null: bool = False):
        super().__init__()
        self.wav_upsample = wav_upsample
        self.mix = WavMixResNet(in_ch=4, out_ch=1, base_ch=wav_base_ch,
                                num_blocks=wav_blocks, norm=wav_norm)
        self.transunet = build_transunet(img_size, pretrained_npz)
        self.residual_out = residual_out
        self.wav_null = wav_null          # P11 — G1 null control

    def forward(self, x):
        _, h, v, d = haar_dwt_hvd(x)  # (B,1,H/2,W/2)
        if getattr(self, 'wav_null', False):          # P11 — G1 null control
            h, v, d = x.clone(), x.clone(), x.clone()
        h = _upsample_like(h, x, mode=self.wav_upsample)
        v = _upsample_like(v, x, mode=self.wav_upsample)
        d = _upsample_like(d, x, mode=self.wav_upsample)

        x4 = torch.cat([x, h, v, d], dim=1)  # (B,4,H,W)
        x_mix = self.mix(x4)                # (B,1,H,W)

        out = self.transunet(x_mix)         # (B,1,H,W)
        if self.residual_out:
            out = x_mix + out
        return out


# -----------------------------
# checkpoint loading (robust)
# -----------------------------
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ['state_dict', 'model_state_dict', 'model', 'net']:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj

def _maybe_strip_prefix(sd: dict, prefix: str) -> dict:
    if all(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd

def load_weights_strict_match(model: nn.Module, ckpt_path: str, device: torch.device):
    # P7: torch>=2.6 flips torch.load's weights_only default to True, which
    # rejects these checkpoints (they store args/optimizer state, not just
    # tensors). Be explicit so the code works on both sides of that change.
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd0 = _extract_state_dict(ckpt)
    if not isinstance(sd0, dict):
        raise RuntimeError(f'Checkpoint is not a state_dict/dict: {ckpt_path}')

    candidates = []
    candidates.append(("raw", sd0))
    candidates.append(("strip_module", _maybe_strip_prefix(sd0, "module.")))

    # strip first segment (wrapper)
    first_seg = list(sd0.keys())[0].split('.')[0]
    candidates.append((f"strip_{first_seg}.", _maybe_strip_prefix(sd0, first_seg + ".")))

    for pref in ["model.", "net.", "unet.", "transunet.", "generator.", "reconstructor."]:
        candidates.append((f"strip_{pref}", _maybe_strip_prefix(sd0, pref)))
        candidates.append((f"strip_module+{pref}", _maybe_strip_prefix(_maybe_strip_prefix(sd0, "module."), pref)))

    last_err = None
    for name, sd in candidates:
        try:
            model.load_state_dict(sd, strict=True)
            model.to(device)
            print(f'[LOAD] {ckpt_path} loaded with "{name}" mapping (strict=True)')
            return model
        except Exception as e:
            last_err = e

    # print hints
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(sd0.keys())
    print('[LOAD-ERROR] strict load failed.')
    print('  model key example:', model_keys[:5])
    print('  ckpt  key example:', ckpt_keys[:5])
    raise RuntimeError(f'Failed to strict-load checkpoint: {ckpt_path}\nLast error: {last_err}')


# -----------------------------
# Visualization
# -----------------------------
def _robust_vmin_vmax(arr: np.ndarray, lo=1.0, hi=99.0):
    vmin = float(np.percentile(arr, lo))
    vmax = float(np.percentile(arr, hi))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    return vmin, vmax

def save_5panel_png(out_path: Path,
                    gt: np.ndarray,
                    fbp: np.ndarray,
                    unet: np.ndarray,
                    transunet: np.ndarray,
                    wavres: np.ndarray,
                    idx: int,
                    psnr_fbp: float,
                    psnr_unet: float,
                    psnr_trans: float,
                    psnr_wavres: float,
                    dpi: int = 150):
    concat = np.concatenate([gt.ravel(), fbp.ravel(), unet.ravel(), transunet.ravel(), wavres.ravel()])
    vmin, vmax = _robust_vmin_vmax(concat, lo=1.0, hi=99.0)

    imgs = [gt, fbp, unet, transunet, wavres]
    titles = [
        "GT",
        f"FBP\nPSNR={psnr_fbp:.2f}",
        f"UNet\nPSNR={psnr_unet:.2f}",
        f"TransUNet\nPSNR={psnr_trans:.2f}",
        f"WavResTransUNet\nPSNR={psnr_wavres:.2f}",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=dpi)
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=10)
        ax.axis('off')

    fig.suptitle(f"TEST idx={idx}", fontsize=12)
    fig.tight_layout(rect=[0, 0.0, 1, 0.92])
    fig.savefig(out_path)
    plt.close(fig)

# (추가) 개별(single) PNG 저장 + (0~1 정규화 후) 5panel(PSNR) + 5panel(|diff|, RMSE) 저장
def _fmt_value(v: float, fmt: str = "{:.2f}") -> str:
    if v is None:
        return "nan"
    if isinstance(v, (float, np.floating)) and (np.isinf(v) or np.isnan(v)):
        return "inf" if np.isinf(v) else "nan"
    return fmt.format(float(v))

def save_single_png(out_path: Path,
                    img: np.ndarray,
                    vmin: float,
                    vmax: float):
    """
    (추가) 0~1 범위로 맞추기 전(=raw) 개별 PNG 저장용.
    - img는 np.ndarray (H,W)
    - vmin/vmax는 시각화용이며, raw npy 값은 그대로 별도 저장
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(out_path), img, cmap='gray', vmin=vmin, vmax=vmax)

def save_5panel_norm_psnr_png(out_path: Path,
                             gt01: np.ndarray,
                             fbp01: np.ndarray,
                             unet01: np.ndarray,
                             trans01: np.ndarray,
                             wav01: np.ndarray,
                             idx: int,
                             psnr_gt: float,
                             psnr_fbp: float,
                             psnr_unet: float,
                             psnr_trans: float,
                             psnr_wav: float,
                             dpi: int = 150):
    """
    (추가) 0~1 정규화 후 5panel 저장 (title에 PSNR).
    - data_range=1 기준 PSNR을 title로 표기
    - 시각화는 vmin=0, vmax=1 고정
    """
    imgs = [gt01, fbp01, unet01, trans01, wav01]
    titles = [
        f"GT\nPSNR={_fmt_value(psnr_gt)}",
        f"FBP\nPSNR={_fmt_value(psnr_fbp)}",
        f"UNet\nPSNR={_fmt_value(psnr_unet)}",
        f"TransUNet\nPSNR={_fmt_value(psnr_trans)}",
        f"WavTransUNet\nPSNR={_fmt_value(psnr_wav)}",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=dpi)
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im, cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_title(t, fontsize=10)
        ax.axis('off')

    fig.suptitle(f"TEST idx={idx} (0~1 norm)", fontsize=12)
    fig.tight_layout(rect=[0, 0.0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

def _robust_vmax_positive(arr: np.ndarray, hi: float = 99.0) -> float:
    vmax = float(np.percentile(arr, hi))
    if (not np.isfinite(vmax)) or (vmax <= 0.0):
        vmax = float(np.max(arr))
    if (not np.isfinite(vmax)) or (vmax <= 0.0):
        vmax = 1e-6
    return vmax

def save_5panel_diff_rmse_png(out_path: Path,
                             diff_gtgt: np.ndarray,
                             diff_gtfbp: np.ndarray,
                             diff_gtunet: np.ndarray,
                             diff_gttrans: np.ndarray,
                             diff_gtwav: np.ndarray,
                             idx: int,
                             rmse_gtgt: float,
                             rmse_gtfbp: float,
                             rmse_gtunet: float,
                             rmse_gttrans: float,
                             rmse_gtwav: float,
                             dpi: int = 150):
    """
    (추가) 0~1 정규화 후 |difference| 5panel 저장 (title에 RMSE).
    - difference는 |GT - X| (absolute difference)
    - 시각화는 vmin=0, vmax는 robust percentile로 자동
    (추가) 실제 저장/표시는 signed difference (GT - X) 를 사용하며, vmin=-vmax 로 0을 중간(gray)로 둠
    """
    diffs = [diff_gtgt, diff_gtfbp, diff_gtunet, diff_gttrans, diff_gtwav]
    concat = np.concatenate([d.ravel() for d in diffs], axis=0)

    # (추가) signed difference 시각화를 위해 |diff| 기반으로 robust vmax 산출 후 [-vmax, +vmax]로 표현
    vmax = _robust_vmax_positive(np.abs(concat), hi=99.0)

    titles = [
        f"GT-GT\nRMSE={rmse_gtgt:.5f}",
        f"GT-FBP\nRMSE={rmse_gtfbp:.5f}",
        f"GT-UNet\nRMSE={rmse_gtunet:.5f}",
        f"GT-TransUNet\nRMSE={rmse_gttrans:.5f}",
        f"GT-WavTransUNet\nRMSE={rmse_gtwav:.5f}",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=dpi)
    for ax, im, t in zip(axes, diffs, titles):
        ax.imshow(im, cmap='gray', vmin=-vmax, vmax=vmax)
        ax.set_title(t, fontsize=10)
        ax.axis('off')

    fig.suptitle(f"TEST idx={idx} | signed diff (0~1 norm)", fontsize=12)
    fig.tight_layout(rect=[0, 0.0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# (추가) 0~1 정규화 후 UNet/TransUNet/WavTransUNet difference 3개를
#       WavTransUNet이 가장 안보이도록(=UNet 기반 range) concatenate + colorbar 포함 PNG 저장
# (추가) difference가 signed 로 바뀌면서, colorbar range 기준을 WavTransUNet 기반으로 변경
def save_concat_3model_diff_colorbar_png(out_path: Path,
                                         diff_unet: np.ndarray,
                                         diff_trans: np.ndarray,
                                         diff_wav: np.ndarray,
                                         idx: int,
                                         dpi: int = 150,
                                         hi: float = 99.0):
    """
    (추가) 0~1 정규화 후 |difference| (GT-UNet, GT-TransUNet, GT-WavTransUNet) 3개를
           concatenate하여 1장의 PNG로 저장 (colorbar 포함).
    - range(vmin/vmax)는 UNet 기반으로 통일하여 WavTransUNet의 음영이 가장 덜 보이도록 설정
    - 각 모델별 p{hi}, max를 표시하여 UNet이 difference 범위가 넓음을 수치로 강조
    (추가) signed difference (GT - X) 표시를 위해 vmin=-vmax, vmax=+vmax 사용
    (추가) colorbar range는 WavTransUNet(가장 차이가 적을 것으로 예상)을 기준으로 설정
    """
    if (diff_unet.shape != diff_trans.shape) or (diff_unet.shape != diff_wav.shape):
        raise ValueError(f"diff shapes must match. Got: unet={diff_unet.shape}, trans={diff_trans.shape}, wav={diff_wav.shape}")

    H, W = diff_unet.shape

    # (추가) WavTransUNet 기반으로 colorbar range 설정: p{hi}(|diff_wav|)
    vmax_ref = _robust_vmax_positive(np.abs(diff_wav), hi=hi)

    # 수치 강조용: 각 모델의 p{hi}(|diff|) 및 max(|diff|)
    vmax_unet  = _robust_vmax_positive(np.abs(diff_unet),  hi=hi)
    vmax_trans = _robust_vmax_positive(np.abs(diff_trans), hi=hi)
    vmax_wav   = _robust_vmax_positive(np.abs(diff_wav),   hi=hi)

    max_unet  = float(np.max(np.abs(diff_unet)))
    max_trans = float(np.max(np.abs(diff_trans)))
    max_wav   = float(np.max(np.abs(diff_wav)))

    r_unet  = vmax_unet  / max(vmax_ref, 1e-12)
    r_trans = vmax_trans / max(vmax_ref, 1e-12)
    r_wav   = vmax_wav   / max(vmax_ref, 1e-12)

    # concatenate (UNet | TransUNet | WavTransUNet)
    concat = np.concatenate([diff_unet, diff_trans, diff_wav], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(18, 5), dpi=dpi)
    im = ax.imshow(concat, cmap='gray', vmin=-vmax_ref, vmax=vmax_ref)
    ax.axis('off')

    # 경계선 (concatenate 구간 구분)
    ax.axvline(W - 0.5, color='k', linewidth=1.0)
    ax.axvline(2 * W - 0.5, color='k', linewidth=1.0)

    # 각 패널 라벨(상단) + 수치 표시 (UNet이 가장 넓은 범위임을 강조)
    ax.text(1/6, 1.02,
            f"UNet\np{int(hi)}(|d|)={vmax_unet:.4f} ({r_unet:.2f}x)\nmax(|d|)={max_unet:.4f}",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=10, clip_on=False)
    ax.text(3/6, 1.02,
            f"TransUNet\np{int(hi)}(|d|)={vmax_trans:.4f} ({r_trans:.2f}x)\nmax(|d|)={max_trans:.4f}",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=10, clip_on=False)
    ax.text(5/6, 1.02,
            f"WavTransUNet\np{int(hi)}(|d|)={vmax_wav:.4f} ({r_wav:.2f}x)\nmax(|d|)={max_wav:.4f}",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=10, clip_on=False)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("GT - X (signed, 0~1 norm)", fontsize=10)

    # 제목: scale이 WavTransUNet 기반임을 명시(=WavTransUNet이 가장 잘 보이는 범위)
    fig.suptitle(
        f"TEST idx={idx} | signed diff compare (shared range = WavTransUNet p{int(hi)}(|d|)={vmax_ref:.4f}) | UNet widest diff expected",
        fontsize=12
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.0, 1, 0.90])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# -----------------------------
# helpers: report
# -----------------------------
def mean_std(arr):
    a = np.asarray(arr, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))

def write_report(out_dir: Path, angle: int, ckpts: dict, metrics: dict):
    """
    metrics[model]['psnr'/'ssim'/'rmse'] = list(float)
    """
    lines = []
    lines.append("LoDoPaB-CT TEST Metric Report")
    lines.append("------------------------------------------------------------")
    lines.append(f"Angle: {angle}")
    lines.append("Checkpoints:")
    for k, v in ckpts.items():
        lines.append(f"  - {k}: {v}")
    lines.append("------------------------------------------------------------")

    # table header
    header = f"{'Model':<18} | {'PSNR(dB) mean±std':<22} | {'SSIM mean±std':<18} | {'RMSE mean±std':<18}"
    lines.append(header)
    lines.append("-" * len(header))

    rows = []
    for model_name in ["FBP_INPUT", "UNet", "TransUNet", "WavResTransUNet"]:
        psnr_m, psnr_s = mean_std(metrics[model_name]["psnr"])
        ssim_m, ssim_s = mean_std(metrics[model_name]["ssim"])
        rmse_m, rmse_s = mean_std(metrics[model_name]["rmse"])

        row = (f"{model_name:<18} | "
               f"{psnr_m:>7.2f} ± {psnr_s:<7.2f} | "
               f"{ssim_m:>6.4f} ± {ssim_s:<6.4f} | "
               f"{rmse_m:>7.5f} ± {rmse_s:<7.5f}")
        rows.append(row)
        lines.append(row)

    lines.append("------------------------------------------------------------")

    txt_path = out_dir / "metrics_report.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    # csv
    csv_path = out_dir / "metrics_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "psnr_mean_db", "psnr_std_db", "ssim_mean", "ssim_std", "rmse_mean", "rmse_std"])
        for model_name in ["FBP_INPUT", "UNet", "TransUNet", "WavResTransUNet"]:
            psnr_m, psnr_s = mean_std(metrics[model_name]["psnr"])
            ssim_m, ssim_s = mean_std(metrics[model_name]["ssim"])
            rmse_m, rmse_s = mean_std(metrics[model_name]["rmse"])
            w.writerow([model_name, psnr_m, psnr_s, ssim_m, ssim_s, rmse_m, rmse_s])

    print(f"[SAVE] metrics_report.txt -> {txt_path}")
    print(f"[SAVE] metrics_report.csv -> {csv_path}")


def write_per_image_metrics(out_dir: Path, metrics: dict, angle: int = 0):
    """Dump one row per test slice: metrics_per_test.csv.

    The aggregate report above only carries mean +- SD, which is not enough for
    a paired analysis: bootstrap CIs, Wilcoxon signed-rank tests and effect
    sizes all need the per-image values, and a reviewer asking for them cannot
    be answered without rerunning inference.

    The variant of this script that was released omitted the dump, although the
    script that produced the published table wrote exactly this file
    (`inference_test_from_cache_compare_5panel.py`, lines 935-964). The columns
    and their order are kept identical to that file so the CSVs are
    interchangeable; the only change is that this version uses `csv` rather
    than pandas, to avoid adding a dependency for one table.
    """
    model_cols = [
        ("FBP_INPUT", "fbp"),
        ("UNet", "unet"),
        ("TransUNet", "transunet"),
        ("WavResTransUNet", "wavtransunet"),
    ]

    num_samples = len(metrics["FBP_INPUT"]["psnr"])
    for model_name, _ in model_cols:
        n = len(metrics[model_name]["psnr"])
        if n != num_samples:
            raise RuntimeError(
                f"per-image metric length mismatch: {model_name} has {n} entries, "
                f"FBP_INPUT has {num_samples}")

    header = ["test_idx"]
    for _, prefix in model_cols:
        header += [f"{prefix}_psnr", f"{prefix}_ssim", f"{prefix}_rmse"]

    csv_path = out_dir / "metrics_per_test.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(num_samples):
            row = [i]
            for model_name, _ in model_cols:
                row += [metrics[model_name]["psnr"][i],
                        metrics[model_name]["ssim"][i],
                        metrics[model_name]["rmse"][i]]
            w.writerow(row)

    print(f"[SAVE] metrics_per_test.csv ({num_samples} rows) -> {csv_path}")

    # P10 (long form): the wide table above keeps the column layout of the file
    # that produced the published table, so the two are interchangeable. The
    # paired-statistics tooling groups by (angle, index, model) instead, so the
    # same numbers are also written one row per (slice, model).
    long_path = out_dir / "per_image_metrics.csv"
    with open(long_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["angle", "index", "model", "psnr", "ssim", "rmse"])
        for model_name, _ in model_cols:
            for i in range(num_samples):
                w.writerow([angle, i, model_name,
                            metrics[model_name]["psnr"][i],
                            metrics[model_name]["ssim"][i],
                            metrics[model_name]["rmse"][i]])
    print(f"[SAVE] per_image_metrics.csv "
          f"({len(model_cols) * num_samples} rows) -> {long_path}")


# -----------------------------
# (추가) 0~1 정규화 헬퍼: GT 기준 min-max로 일괄 스케일링
# -----------------------------
def normalize_0_1_by_gt(img: torch.Tensor, gt_min: torch.Tensor, gt_range: torch.Tensor) -> torch.Tensor:
    """
    (추가) per-sample GT(min/max) 기준으로 img를 0~1로 맞춤.
    - img: (B,1,H,W)
    - gt_min, gt_range: (B,1,1,1)
    """
    return torch.clamp((img - gt_min) / gt_range, 0.0, 1.0)


# -----------------------------
# main
# -----------------------------
def main():
    args = get_args()
    device = get_device()
    print(f'[INFO] device={device}')

    # sanity checks
    if (args.img_size % 16) != 0:
        raise ValueError(f'img_size must be multiple of 16. Got {args.img_size}')
    if (args.img_size % 2) != 0:
        raise ValueError(f'img_size must be even (Haar DWT). Got {args.img_size}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # (기존) 5panel PNG 디렉토리
    # (추가) 0~1 정규화 후 PSNR 5panel / |diff|+RMSE 5panel / raw 개별 PNG/NPY 디렉토리 생성
    png_dir_norm_psnr = out_dir / "png_5panel_norm_psnr"
    png_dir_norm_psnr.mkdir(parents=True, exist_ok=True)

    png_dir_norm_diff = out_dir / "png_5panel_norm_diff_rmse"
    png_dir_norm_diff.mkdir(parents=True, exist_ok=True)

    raw_png_root = out_dir / "raw_individual_png"
    raw_npy_root = out_dir / "raw_individual_npy"
    raw_png_root.mkdir(parents=True, exist_ok=True)
    raw_npy_root.mkdir(parents=True, exist_ok=True)

    raw_names = ["GT", "FBP", "UNet", "TransUNet", "WavTransUNet"]
    raw_png_dirs = {n: (raw_png_root / n) for n in raw_names}
    raw_npy_dirs = {n: (raw_npy_root / n) for n in raw_names}
    for n in raw_names:
        raw_png_dirs[n].mkdir(parents=True, exist_ok=True)
        raw_npy_dirs[n].mkdir(parents=True, exist_ok=True)

    # (추가) 0~1 정규화 후 GT&FBP, GT&UNet, GT&TransUNet, GT&WavTransUNet difference 개별 PNG/NPY 저장 디렉토리
    diff_png_root = out_dir / "norm_diff_individual_png"
    diff_npy_root = out_dir / "norm_diff_individual_npy"
    diff_png_root.mkdir(parents=True, exist_ok=True)
    diff_npy_root.mkdir(parents=True, exist_ok=True)

    diff_pairs = ["GT_FBP", "GT_UNet", "GT_TransUNet", "GT_WavTransUNet"]
    diff_png_dirs = {n: (diff_png_root / n) for n in diff_pairs}
    diff_npy_dirs = {n: (diff_npy_root / n) for n in diff_pairs}
    for n in diff_pairs:
        diff_png_dirs[n].mkdir(parents=True, exist_ok=True)
        diff_npy_dirs[n].mkdir(parents=True, exist_ok=True)

    # (추가) 0~1 정규화 후 UNet/TransUNet/WavTransUNet difference 3개 concatenate + colorbar 저장 디렉토리
    png_dir_norm_diff_compare3 = out_dir / "png_concat_3panel_norm_diff_compare_colorbar"
    png_dir_norm_diff_compare3.mkdir(parents=True, exist_ok=True)

    # resolve cache path
    if args.cache_fbp.strip():
        cache_fbp_path = Path(args.cache_fbp)
    else:
        cache_fbp_path = Path(args.cache_root) / f"{args.angle}angle" / "cache_lodopab_test_fbp.npy"

    if not cache_fbp_path.is_file():
        raise FileNotFoundError(f"FBP cache not found: {cache_fbp_path}")

    print(f"[INFO] Loading cached TEST FBP: {cache_fbp_path}")
    fbp_cache = np.load(cache_fbp_path, mmap_mode='r')  # (N,H,W)
    n_test = int(fbp_cache.shape[0])
    print(f"[INFO] #test samples from cache: {n_test}")

    # dataset only for GT (test) - stream in same order
    if not LODOPAB_PATH:
        raise ValueError('Set LODOPAB_DATA or LODOPAB_PATH explicitly')
    dataset = get_standard_dataset('lodopab', impl='astra_cuda', num_angles=args.angle)

    # make gt iterator in test order (avoid random access)
    try:
        gt_pairs_iter = iter(dataset.get_data_pairs('test', n_test))
    except TypeError:
        gt_pairs_iter = iter(dataset.get_data_pairs('test'))

    # decide which indices to save as png
    if args.png_indices.strip():
        sel = [int(s) for s in args.png_indices.split(',') if s.strip() != ""]
        save_indices = set([i for i in sel if 0 <= i < n_test])
        print(f"[INFO] PNG indices explicitly set: {sorted(save_indices)[:20]} (total {len(save_indices)})")
    else:
        if args.save_all_png:
            save_indices = set(range(n_test))
            print("[INFO] Will save PNG/NPY outputs for ALL test samples.")
        else:
            k = max(0, min(args.png_n, n_test))
            save_indices = set(range(k))
            print(f"[INFO] Will save PNG/NPY outputs for first {k} test samples.")

    # build models
    # 1) UNet
    unet_model = get_unet_model(
        in_ch=1,
        out_ch=1,
        scales=args.unet_scales,
        skip=args.unet_skip,
        use_sigmoid=(not args.unet_no_sigmoid),
        use_norm=(not args.unet_no_norm)
    )

    # 2) TransUNet (plain)
    transunet_model = build_transunet(args.img_size, args.pretrained_npz)

    # 3) WavResTransUNet
    wavres_model = WavResTransUNet(
        img_size=args.img_size,
        pretrained_npz=args.pretrained_npz,
        wav_base_ch=args.wav_base_ch,
        wav_blocks=args.wav_blocks,
        wav_norm=args.wav_norm,
        wav_upsample=args.wav_upsample,
        residual_out=args.residual_out,
        wav_null=bool(args.wav_null),
    )
    print(f"[P11] wav_null={bool(args.wav_null)}")

    # load weights (strict)
    unet_model = load_weights_strict_match(unet_model, args.ckpt_unet, device=device).eval()
    transunet_model = load_weights_strict_match(transunet_model, args.ckpt_transunet, device=device).eval()
    wavres_model = load_weights_strict_match(wavres_model, args.ckpt_wavres, device=device).eval()

    # metrics storage
    metrics = {
        "FBP_INPUT": {"psnr": [], "ssim": [], "rmse": []},
        "UNet": {"psnr": [], "ssim": [], "rmse": []},
        "TransUNet": {"psnr": [], "ssim": [], "rmse": []},
        "WavResTransUNet": {"psnr": [], "ssim": [], "rmse": []},
    }

    num_batches = math.ceil(n_test / args.batch)
    print(f"[INFO] Start inference: batch={args.batch}, total_batches={num_batches}")

    # inference loop
    for b in range(num_batches):
        s = b * args.batch
        e = min((b + 1) * args.batch, n_test)
        idxs = list(range(s, e))
        B = len(idxs)

        # build batch
        xs = []
        ys = []
        for _idx in idxs:
            # FBP from cache (aligned by index)
            x = center_crop(np.asarray(fbp_cache[_idx]), args.img_size)

            # GT from iterator (same order)
            try:
                _, gt = next(gt_pairs_iter)
            except StopIteration:
                raise RuntimeError(
                    "GT iterator ended before reaching n_test. "
                    "Your cache length and dataset test length may not match."
                )
            y = center_crop(np.asarray(gt), args.img_size)

            xs.append(x)
            ys.append(y)

        x_np = np.stack(xs, axis=0).astype(np.float32)  # (B,H,W)  (raw before norm)
        y_np = np.stack(ys, axis=0).astype(np.float32)  # (B,H,W)  (raw before norm)

        x_t = torch.from_numpy(x_np).unsqueeze(1).to(device)  # (B,1,H,W)
        y_t = torch.from_numpy(y_np).unsqueeze(1).to(device)

        with torch.no_grad():
            out_unet  = unet_model(x_t)
            out_trans = transunet_model(x_t)
            out_wav   = wavres_model(x_t)

        # per-image data_range from GT (기존 metric 계산용)
        dr = _data_range_from_gt(y_t)  # (B,)

        # compute metrics (torch) - 기존 보고서용(정규화 전 스케일)
        psnr_fbp  = psnr_torch(x_t, y_t, dr)
        rmse_fbp  = rmse_torch(x_t, y_t)
        ssim_fbp  = ssim_torch(x_t, y_t, dr)

        psnr_unet = psnr_torch(out_unet, y_t, dr)
        rmse_unet = rmse_torch(out_unet, y_t)
        ssim_unet = ssim_torch(out_unet, y_t, dr)

        psnr_tr   = psnr_torch(out_trans, y_t, dr)
        rmse_tr   = rmse_torch(out_trans, y_t)
        ssim_tr   = ssim_torch(out_trans, y_t, dr)

        psnr_wav  = psnr_torch(out_wav, y_t, dr)
        rmse_wav  = rmse_torch(out_wav, y_t)
        ssim_wav  = ssim_torch(out_wav, y_t, dr)

        # store metrics (CPU float lists)
        metrics["FBP_INPUT"]["psnr"].extend(psnr_fbp.detach().cpu().numpy().tolist())
        metrics["FBP_INPUT"]["ssim"].extend(ssim_fbp.detach().cpu().numpy().tolist())
        metrics["FBP_INPUT"]["rmse"].extend(rmse_fbp.detach().cpu().numpy().tolist())

        metrics["UNet"]["psnr"].extend(psnr_unet.detach().cpu().numpy().tolist())
        metrics["UNet"]["ssim"].extend(ssim_unet.detach().cpu().numpy().tolist())
        metrics["UNet"]["rmse"].extend(rmse_unet.detach().cpu().numpy().tolist())

        metrics["TransUNet"]["psnr"].extend(psnr_tr.detach().cpu().numpy().tolist())
        metrics["TransUNet"]["ssim"].extend(ssim_tr.detach().cpu().numpy().tolist())
        metrics["TransUNet"]["rmse"].extend(rmse_tr.detach().cpu().numpy().tolist())

        metrics["WavResTransUNet"]["psnr"].extend(psnr_wav.detach().cpu().numpy().tolist())
        metrics["WavResTransUNet"]["ssim"].extend(ssim_wav.detach().cpu().numpy().tolist())
        metrics["WavResTransUNet"]["rmse"].extend(rmse_wav.detach().cpu().numpy().tolist())

        # (추가) 0~1 범위 정규화 (GT 기준 min-max) 후 PSNR/RMSE 계산 및 시각화용 배열 준비
        with torch.no_grad():
            gt_min = y_t.amin(dim=(2, 3), keepdim=True)                  # (B,1,1,1)
            gt_max = y_t.amax(dim=(2, 3), keepdim=True)                  # (B,1,1,1)
            gt_rng = (gt_max - gt_min).clamp(min=1e-8)                   # (B,1,1,1)

            y_01 = normalize_0_1_by_gt(y_t, gt_min, gt_rng)              # (B,1,H,W)
            x_01 = normalize_0_1_by_gt(x_t, gt_min, gt_rng)
            unet_01  = normalize_0_1_by_gt(out_unet,  gt_min, gt_rng)
            trans_01 = normalize_0_1_by_gt(out_trans, gt_min, gt_rng)
            wav_01   = normalize_0_1_by_gt(out_wav,   gt_min, gt_rng)

            dr01 = torch.ones((B,), device=device, dtype=y_t.dtype)      # data_range=1

            # PSNR (0~1 정규화 후, GT 대비)
            psnr_fbp_01  = psnr_torch(x_01,     y_01, dr01)
            psnr_unet_01 = psnr_torch(unet_01,  y_01, dr01)
            psnr_tr_01   = psnr_torch(trans_01, y_01, dr01)
            psnr_wav_01  = psnr_torch(wav_01,   y_01, dr01)

            # RMSE (0~1 정규화 후, GT&X)
            rmse_gtgt_01  = rmse_torch(y_01,     y_01)
            rmse_fbp_01   = rmse_torch(x_01,     y_01)
            rmse_unet_01  = rmse_torch(unet_01,  y_01)
            rmse_tr_01    = rmse_torch(trans_01, y_01)
            rmse_wav_01   = rmse_torch(wav_01,   y_01)

        # numpy outputs (raw before norm) for saving
        out_unet_np  = out_unet[:, 0].detach().cpu().numpy().astype(np.float32)
        out_trans_np = out_trans[:, 0].detach().cpu().numpy().astype(np.float32)
        out_wav_np   = out_wav[:, 0].detach().cpu().numpy().astype(np.float32)

        # numpy outputs (0~1 after norm) for saving
        y01_np     = y_01[:, 0].detach().cpu().numpy().astype(np.float32)
        x01_np     = x_01[:, 0].detach().cpu().numpy().astype(np.float32)
        unet01_np  = unet_01[:, 0].detach().cpu().numpy().astype(np.float32)
        trans01_np = trans_01[:, 0].detach().cpu().numpy().astype(np.float32)
        wav01_np   = wav_01[:, 0].detach().cpu().numpy().astype(np.float32)

        # PSNR (0~1 after norm) numpy
        psnr_fbp01_np  = psnr_fbp_01.detach().cpu().numpy()
        psnr_unet01_np = psnr_unet_01.detach().cpu().numpy()
        psnr_tr01_np   = psnr_tr_01.detach().cpu().numpy()
        psnr_wav01_np  = psnr_wav_01.detach().cpu().numpy()

        # RMSE (0~1 after norm) numpy
        rmse_gtgt01_np = rmse_gtgt_01.detach().cpu().numpy()
        rmse_fbp01_np  = rmse_fbp_01.detach().cpu().numpy()
        rmse_unet01_np = rmse_unet_01.detach().cpu().numpy()
        rmse_tr01_np   = rmse_tr_01.detach().cpu().numpy()
        rmse_wav01_np  = rmse_wav_01.detach().cpu().numpy()

        # save per-sample outputs (raw individual + norm 5panel(psnr) + norm diff 5panel(rmse))
        for i, idx in enumerate(idxs):
            # P12: NPY for all (when requested), PNG only for selected indices.
            do_npy = bool(args.save_npy_all) or (idx in save_indices)
            do_png = (idx in save_indices)
            if not (do_npy or do_png):
                continue

            # -----------------------------------------
            # (추가-1) 0~1 범위로 맞추기 전 개별 이미지 PNG + NPY 저장
            # -----------------------------------------
            # npy raw 저장 (P12: gated by do_npy)
            if do_npy:
                np.save(raw_npy_dirs["GT"] / f"test_idx_{idx:05d}.npy", y_np[i].astype(np.float32))
                np.save(raw_npy_dirs["FBP"] / f"test_idx_{idx:05d}.npy", x_np[i].astype(np.float32))
                np.save(raw_npy_dirs["UNet"] / f"test_idx_{idx:05d}.npy", out_unet_np[i].astype(np.float32))
                np.save(raw_npy_dirs["TransUNet"] / f"test_idx_{idx:05d}.npy", out_trans_np[i].astype(np.float32))
                np.save(raw_npy_dirs["WavTransUNet"] / f"test_idx_{idx:05d}.npy", out_wav_np[i].astype(np.float32))

            # P12: everything below is figure rendering (raw PNG, 5-panel,
            #      difference maps). Skip it for samples that are NPY-only.
            if not do_png:
                continue

            # raw 시각화 vmin/vmax는 (해당 샘플의 5개 이미지 concat 기반) robust로 통일
            concat_raw = np.concatenate([
                y_np[i].ravel(),
                x_np[i].ravel(),
                out_unet_np[i].ravel(),
                out_trans_np[i].ravel(),
                out_wav_np[i].ravel()
            ], axis=0)
            vmin_raw, vmax_raw = _robust_vmin_vmax(concat_raw, lo=1.0, hi=99.0)

            # png raw 저장
            save_single_png(raw_png_dirs["GT"] / f"test_idx_{idx:05d}.png", y_np[i], vmin_raw, vmax_raw)
            save_single_png(raw_png_dirs["FBP"] / f"test_idx_{idx:05d}.png", x_np[i], vmin_raw, vmax_raw)
            save_single_png(raw_png_dirs["UNet"] / f"test_idx_{idx:05d}.png", out_unet_np[i], vmin_raw, vmax_raw)
            save_single_png(raw_png_dirs["TransUNet"] / f"test_idx_{idx:05d}.png", out_trans_np[i], vmin_raw, vmax_raw)
            save_single_png(raw_png_dirs["WavTransUNet"] / f"test_idx_{idx:05d}.png", out_wav_np[i], vmin_raw, vmax_raw)

            # -----------------------------------------
            # (추가-2) 0~1 정규화 후 GT/FBP/UNet/TransUNet/WavTransUNet 5panel + PSNR title 저장
            # -----------------------------------------
            out_path_psnr = png_dir_norm_psnr / f"test_idx_{idx:05d}_5panel_psnr.png"
            # GT의 PSNR은 정의상 무한대(동일 영상)라서 title 표기만 inf로 처리
            save_5panel_norm_psnr_png(
                out_path=out_path_psnr,
                gt01=y01_np[i],
                fbp01=x01_np[i],
                unet01=unet01_np[i],
                trans01=trans01_np[i],
                wav01=wav01_np[i],
                idx=idx,
                psnr_gt=float("inf"),
                psnr_fbp=float(psnr_fbp01_np[i]),
                psnr_unet=float(psnr_unet01_np[i]),
                psnr_trans=float(psnr_tr01_np[i]),
                psnr_wav=float(psnr_wav01_np[i]),
                dpi=args.dpi
            )

            # -----------------------------------------
            # (추가-3) 0~1 정규화 후 |difference| 5panel + RMSE title 저장
            # -----------------------------------------
            # (추가) abs 대신 signed difference(GT - X)로 변경하여 음/양 차이를 그대로 표현
            diff_gtgt  = y01_np[i] - y01_np[i]          # np.abs(y01_np[i] - y01_np[i])
            diff_gtfbp = y01_np[i] - x01_np[i]          # np.abs(y01_np[i] - x01_np[i])
            diff_gtun  = y01_np[i] - unet01_np[i]       # np.abs(y01_np[i] - unet01_np[i])
            diff_gttr  = y01_np[i] - trans01_np[i]      # np.abs(y01_np[i] - trans01_np[i])
            diff_gtwv  = y01_np[i] - wav01_np[i]        # np.abs(y01_np[i] - wav01_np[i])

            # (추가) 0~1 범위로 맞춘 뒤 GT&FBP, GT&UNet, GT&TransUNet, GT&WavTransUNet difference를
            #       각각 PNG + NPY로 개별 저장
            # NOTE: 현재는 |GT - X| (absolute difference) 저장.
            #       signed difference를 원하면 np.abs(...) 대신 (y01_np[i] - x01_np[i]) 처럼 바꾸면 됨.
            # (추가) 본 버전은 이미 signed difference(GT - X)를 사용하므로 vmin=-vmax 로 저장
            diff_map = {
                "GT_FBP": diff_gtfbp,
                "GT_UNet": diff_gtun,
                "GT_TransUNet": diff_gttr,
                "GT_WavTransUNet": diff_gtwv,
            }

            # 시각화 스케일 통일(해당 샘플 내 4개 diff 기준) : vmin=0, vmax=robust
            concat_diff = np.concatenate([d.ravel() for d in diff_map.values()], axis=0)
            # (추가) signed difference 이므로 |diff| 기반 vmax 산출 후 [-vmax, +vmax]로 저장
            vmax_diff = _robust_vmax_positive(np.abs(concat_diff), hi=99.0)

            for name, dimg in diff_map.items():
                np.save(diff_npy_dirs[name] / f"test_idx_{idx:05d}.npy", dimg.astype(np.float32))
                save_single_png(diff_png_dirs[name] / f"test_idx_{idx:05d}.png", dimg, -vmax_diff, vmax_diff)

            out_path_diff = png_dir_norm_diff / f"test_idx_{idx:05d}_5panel_diff_rmse.png"
            save_5panel_diff_rmse_png(
                out_path=out_path_diff,
                diff_gtgt=diff_gtgt,
                diff_gtfbp=diff_gtfbp,
                diff_gtunet=diff_gtun,
                diff_gttrans=diff_gttr,
                diff_gtwav=diff_gtwv,
                idx=idx,
                rmse_gtgt=float(rmse_gtgt01_np[i]),
                rmse_gtfbp=float(rmse_fbp01_np[i]),
                rmse_gtunet=float(rmse_unet01_np[i]),
                rmse_gttrans=float(rmse_tr01_np[i]),
                rmse_gtwav=float(rmse_wav01_np[i]),
                dpi=args.dpi
            )

            # (추가) UNet/TransUNet/WavTransUNet difference 3개를 UNet 기반 range로 맞춰서
            #       (WavTransUNet 음영이 가장 덜 보이도록) concatenate + colorbar 포함 PNG 저장
            # (추가) WavTransUNet 기준 range로 변경(=Wav 차이가 가장 작을 것으로 예상)하여 UNet의 clipping을 통해 범위가 넓음을 강조
            out_path_cmp3 = png_dir_norm_diff_compare3 / f"test_idx_{idx:05d}_concat_diff_UNet_Trans_Wav_colorbar.png"
            save_concat_3model_diff_colorbar_png(
                out_path=out_path_cmp3,
                diff_unet=diff_gtun,
                diff_trans=diff_gttr,
                diff_wav=diff_gtwv,
                idx=idx,
                dpi=args.dpi,
                hi=99.0
            )

        if (b + 1) % 10 == 0 or (b + 1) == num_batches:
            print(f"[PROGRESS] batch {b+1}/{num_batches} | done {e}/{n_test}")

    # report
    ckpts = {
        "UNet": args.ckpt_unet,
        "TransUNet": args.ckpt_transunet,
        "WavResTransUNet": args.ckpt_wavres
    }

    print("\n====================== METRIC REPORT (TEST) ======================")
    for model_name in ["FBP_INPUT", "UNet", "TransUNet", "WavResTransUNet"]:
        psnr_m, psnr_s = mean_std(metrics[model_name]["psnr"])
        ssim_m, ssim_s = mean_std(metrics[model_name]["ssim"])
        rmse_m, rmse_s = mean_std(metrics[model_name]["rmse"])
        print(f"{model_name:>15s} | PSNR {psnr_m:7.2f} ± {psnr_s:6.2f} dB"
              f" | SSIM {ssim_m:6.4f} ± {ssim_s:6.4f}"
              f" | RMSE {rmse_m:7.5f} ± {rmse_s:7.5f}")
    print("==================================================================\n")

    write_report(out_dir=out_dir, angle=args.angle, ckpts=ckpts, metrics=metrics)
    write_per_image_metrics(out_dir=out_dir, metrics=metrics, angle=args.angle)

    print(f"[SAVE] raw individual PNG root: {raw_png_root}")
    print(f"[SAVE] raw individual NPY root: {raw_npy_root}")
    print(f"[SAVE] 0~1 norm PSNR 5panel dir: {png_dir_norm_psnr}")
    print(f"[SAVE] 0~1 norm |diff|+RMSE 5panel dir: {png_dir_norm_diff}")
    print(f"[SAVE] 0~1 norm diff compare(concat+colorbar) dir: {png_dir_norm_diff_compare3}")


if __name__ == "__main__":
    main()
