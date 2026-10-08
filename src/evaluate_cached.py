"""Cached five-view paper evaluation with explicit paths.

Portability changes only: local architecture imports, explicit cache/checkpoint
paths and CPU checkpoint deserialization. Metric definitions, crop, model
operations and checkpoint-matching rules retain the historical implementation.
The historical executed command and byte-frozen source were not recovered.
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
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

from unet import get_unet_model
from vit_seg_modeling import VisionTransformer as ViT_seg
from vit_seg_modeling import CONFIGS as CFG_ViT


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

    p.add_argument('--cache_gt', type=str, required=True,
                   help='reference cache in the same official test order as FBP')

    # checkpoints
    p.add_argument('--ckpt_unet', type=str,
                   required=True)
    p.add_argument('--ckpt_transunet', type=str,
                   required=True)
    p.add_argument('--ckpt_wavres', type=str,
                   required=True,
                   help='WavResTransUNet checkpoint (epoch_150.pth or epoch_xxx.pth)')

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

    # UNet options (default: use sigmoid + use norm)
    p.add_argument('--unet_scales', type=int, default=5)
    p.add_argument('--unet_skip', type=int, default=4)
    p.add_argument('--unet_no_sigmoid', action='store_true',
                   help='if set, disable sigmoid at UNet output')
    p.add_argument('--unet_no_norm', action='store_true',
                   help='if set, disable normalization layers in UNet blocks')

    # output
    p.add_argument('--out_dir', type=str, default='./test_compare_out')
    p.add_argument('--dpi', type=int, default=300)

    # png saving control
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
    rmse = torch.sqrt(mse + 1e-12)
    rmse_abs_map = torch.abs(pred - gt)
    return rmse, rmse_abs_map

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
# Wavelet helpers (match training)
# -----------------------------
def _upsample_like(x: torch.Tensor, ref: torch.Tensor, mode: str) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    if mode == 'nearest':
        return F.interpolate(x, size=ref.shape[-2:], mode='nearest')
    return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

def haar_dwt_hvd(x: torch.Tensor):
    B, C, H, W = x.shape
    if (H % 2 != 0) or (W % 2 != 0):
        raise ValueError(f'Haar DWT requires even H,W. Got {H}x{W}')

    y = F.pixel_unshuffle(x, 2)              # (B, 4C, H/2, W/2)
    y = y.view(B, C, 4, H // 2, W // 2)      # (B, C, 4, h, w)
    x00 = y[:, :, 0]
    x01 = y[:, :, 1]
    x10 = y[:, :, 2]
    x11 = y[:, :, 3]

    ll = (x00 + x01 + x10 + x11) * 0.5
    h  = (x00 - x01 + x10 - x11) * 0.5
    v  = (x00 + x01 - x10 - x11) * 0.5
    d  = (x00 - x01 - x10 + x11) * 0.5
    return ll, h, v, d


# -----------------------------
# WavResNet (MATCH training names!)
# -----------------------------
def make_norm(norm: str, num_ch: int):
    if norm == 'none':
        return nn.Identity()
    if norm == 'bn':
        return nn.BatchNorm2d(num_ch)
    if norm == 'in':
        return nn.InstanceNorm2d(num_ch, affine=True)
    if norm == 'gn':
        g = 8 if num_ch >= 8 else 1
        return nn.GroupNorm(num_groups=g, num_channels=num_ch)
    raise ValueError(norm)

class ResBlock(nn.Module):
    def __init__(self, ch: int, norm: str = 'gn'):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm1 = make_norm(norm, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm2 = make_norm(norm, ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + h
        x = self.act(x)
        return x

class WavMixResNet(nn.Module):
    def __init__(self, in_ch: int = 4, out_ch: int = 1, base_ch: int = 64,
                 num_blocks: int = 8, norm: str = 'gn'):
        super().__init__()
        self.in_proj  = nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False)
        self.in_norm  = make_norm(norm, base_ch)
        self.act      = nn.ReLU(inplace=True)

        self.blocks = nn.Sequential(*[ResBlock(base_ch, norm=norm) for _ in range(num_blocks)])

        self.out_proj = nn.Conv2d(base_ch, out_ch, 3, padding=1, bias=True)
        self.skip_proj = nn.Conv2d(in_ch, out_ch, 1, bias=True)

    def forward(self, x4):
        y = self.in_proj(x4)
        y = self.in_norm(y)
        y = self.act(y)
        y = self.blocks(y)
        y = self.out_proj(y)
        return y + self.skip_proj(x4)


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
                 residual_out: bool = False):
        super().__init__()
        self.wav_upsample = wav_upsample
        self.mix = WavMixResNet(in_ch=4, out_ch=1, base_ch=wav_base_ch,
                                num_blocks=wav_blocks, norm=wav_norm)
        self.transunet = build_transunet(img_size, pretrained_npz)
        self.residual_out = residual_out

    def forward(self, x):
        _, h, v, d = haar_dwt_hvd(x)  # (B,1,H/2,W/2)
        h = _upsample_like(h, x, mode=self.wav_upsample)
        v = _upsample_like(v, x, mode=self.wav_upsample)
        d = _upsample_like(d, x, mode=self.wav_upsample)

        x4 = torch.cat([x, h, v, d], dim=1)  # (B,4,H,W)
        x_mix = self.mix(x4)                # (B,1,H,W)

        out = self.transunet(x_mix)         # (B,1,H,W)
        if self.residual_out:
            out = x_mix + out
        return out, h, v, d


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


def normalize_01(img):
    """영상 하나를 자체 min‑max 기준으로 0–1 스케일링"""
    mn, mx = img.min(), img.max()
    if mx == mn:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - mn) / (mx - mn)).astype(np.float32)

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
    # concat = np.concatenate([gt.ravel(), fbp.ravel(), unet.ravel(), transunet.ravel(), wavres.ravel()])
    # vmin, vmax = _robust_vmin_vmax(concat, lo=1.0, hi=99.0)

    # fbp = np.clip(fbp, 0, 1)
    # unet = np.clip(unet, 0, 1)
    # transunet = np.clip(transunet, 0, 1)
    # wavres = np.clip(wavres, 0, 1)

    fbp = normalize_01(fbp)
    unet = normalize_01(unet)
    transunet = normalize_01(transunet)
    wavres = normalize_01(wavres)


    imgs = [gt, fbp, unet, transunet, wavres]
    titles = [
        "GT",
        f"FBP\nPSNR={psnr_fbp:.2f}",
        f"UNet\nPSNR={psnr_unet:.2f}",
        f"TransUNet\nPSNR={psnr_trans:.2f}",
        f"WavResTransUNet\nPSNR={psnr_wavres:.2f}",
    ]
    print("loading ... ...")

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=dpi)
    for ax, im, t in zip(axes, imgs, titles):
        # ax.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
        ax.imshow(im, cmap='gray')
        ax.set_title(t, fontsize=10)
        ax.axis('off')

    fig.suptitle(f"TEST idx={idx}", fontsize=12)
    fig.tight_layout(rect=[0, 0.0, 1, 0.92])
    fig.savefig(out_path)
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
    png_dir = out_dir /f"{args.angle}"/ "png_5panel_test"
    png_dir.mkdir(parents=True, exist_ok=True)

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

    cache_gt_path = Path(args.cache_gt)
    gt_cache = np.load(cache_gt_path, mmap_mode='r')
    if len(gt_cache) != n_test:
        raise ValueError('FBP and reference caches must have identical case counts')


    # make gt iterator in test order (avoid random access)

    # decide which indices to save as png
    if args.png_indices.strip():
        sel = [int(s) for s in args.png_indices.split(',') if s.strip() != ""]
        save_indices = set([i for i in sel if 0 <= i < n_test])
        print(f"[INFO] PNG indices explicitly set: {sorted(save_indices)[:20]} (total {len(save_indices)})")
    else:
        if args.save_all_png:
            save_indices = set(range(n_test))
            print("[INFO] Will save PNG for ALL test samples.")
        else:
            k = max(0, min(args.png_n, n_test))
            save_indices = set(range(k))
            print(f"[INFO] Will save PNG for first {k} test samples.")

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
        residual_out=args.residual_out
    )

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
            y = center_crop(np.asarray(gt_cache[_idx]), args.img_size)

            xs.append(x)
            ys.append(y)

        x_np = np.stack(xs, axis=0).astype(np.float32)  # (B,H,W)
        y_np = np.stack(ys, axis=0).astype(np.float32)

        x_t = torch.from_numpy(x_np).unsqueeze(1).to(device)  # (B,1,H,W)
        y_t = torch.from_numpy(y_np).unsqueeze(1).to(device)

        with torch.no_grad():
            out_unet  = unet_model(x_t)
            out_trans = transunet_model(x_t)
            out_wav, h, v, d = wavres_model(x_t)

        wav_h = h[:, 0].detach().cpu().numpy()
        wav_v = v[:, 0].detach().cpu().numpy()
        wav_d = d[:, 0].detach().cpu().numpy()

        # per-image data_range from GT
        dr = _data_range_from_gt(y_t)  # (B,)

        # compute metrics (torch)
        psnr_fbp = psnr_torch(x_t, y_t, dr)
        rmse_fbp, rmse_fbp_map = rmse_torch(x_t, y_t)
        ssim_fbp  = ssim_torch(x_t, y_t, dr)

        psnr_unet = psnr_torch(out_unet, y_t, dr)
        rmse_unet, rmse_unet_map = rmse_torch(out_unet, y_t)
        ssim_unet = ssim_torch(out_unet, y_t, dr)

        psnr_tr = psnr_torch(out_trans, y_t, dr)
        rmse_tr, rmse_trans_map = rmse_torch(out_trans, y_t)
        ssim_tr   = ssim_torch(out_trans, y_t, dr)

        psnr_wav  = psnr_torch(out_wav, y_t, dr)
        rmse_wav, rmse_wav_map = rmse_torch(out_wav, y_t)
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

        # PNG save (per-sample)
        out_unet_np  = out_unet[:, 0].detach().cpu().numpy()
        out_trans_np = out_trans[:, 0].detach().cpu().numpy()
        out_wav_np   = out_wav[:, 0].detach().cpu().numpy()



        psnr_fbp_np  = psnr_fbp.detach().cpu().numpy()
        psnr_unet_np = psnr_unet.detach().cpu().numpy()
        psnr_tr_np   = psnr_tr.detach().cpu().numpy()
        psnr_wav_np  = psnr_wav.detach().cpu().numpy()

        rmse_fbp = rmse_fbp.detach().cpu().numpy()
        rmse_unet = rmse_unet.detach().cpu().numpy()
        rmse_tr = rmse_tr.detach().cpu().numpy()
        rmse_wav = rmse_wav.detach().cpu().numpy()

        rmse_fbp_map = rmse_fbp_map.detach().cpu().numpy()
        rmse_unet_map = rmse_unet_map.detach().cpu().numpy()
        rmse_trans_map   = rmse_trans_map.detach().cpu().numpy()
        rmse_wav_map  = rmse_wav_map.detach().cpu().numpy()

        for i, idx in enumerate(idxs):
            if idx in save_indices:
                out_path = png_dir / f"test_idx_{idx:05d}.png"
                save_5panel_png(
                    out_path=out_path,
                    gt=y_np[i],
                    fbp=x_np[i],
                    unet=out_unet_np[i],
                    transunet=out_trans_np[i],
                    wavres=out_wav_np[i],
                    idx=idx,
                    psnr_fbp=float(psnr_fbp_np[i]),
                    psnr_unet=float(psnr_unet_np[i]),
                    psnr_trans=float(psnr_tr_np[i]),
                    psnr_wavres=float(psnr_wav_np[i]),
                    dpi=args.dpi
                )

                # # concat = np.concatenate([out_wav_np[i].ravel(), wav_h[i].ravel(), wav_v[i].ravel(), wav_d[i].ravel()])
                # concat = out_wav_np[i].ravel()
                # vmin, vmax = _robust_vmin_vmax(concat, lo=1.0, hi=99.0)

                imgs = {
                    "FBP" : x_np[i],
                    "Horizontal": wav_h[i],
                    "Vertical": wav_v[i],
                    "Diagonal": wav_d[i],
                    "TransUNet": out_trans_np[i],
                    "WavTransUNet" : out_wav_np[i]
                }

                for name, img in imgs.items():
                    out_path = png_dir / f"test_idx_{idx:05d}_{name}.png"
                    plt.figure(figsize=(4, 4), dpi=args.dpi)
                    # plt.imshow(img, cmap="gray",  vmin=vmin, vmax=vmax)
                    plt.imshow(img, cmap="gray")
                    plt.axis("off")
                    # plt.title(name, fontsize=10)
                    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
                    plt.close()

                rmse_imgs = {
                    "rmse_FBP": rmse_fbp_map[i, 0],
                    "rmse_UNet": rmse_unet_map[i, 0],
                    "rmse_TransUNet": rmse_trans_map[i, 0],
                    "rmse_WavTransUNet": rmse_wav_map[i, 0],
                }

                # concat = np.concatenate([rmse_unet_map[i,0].ravel(), rmse_trans_map[i,0].ravel(), rmse_wav_map[i,0].ravel()])
                # vmin, vmax = _robust_vmin_vmax(concat, lo=1.0, hi=99.0)


                rmse_vals = {
                    "rmse_FBP": rmse_fbp[i],
                    "rmse_UNet": rmse_unet[i],
                    "rmse_TransUNet": rmse_tr[i],
                    "rmse_WavTransUNet": rmse_wav[i],
                }

                for model, rmse_img in rmse_imgs.items():
                    out_path = png_dir / f"test_idx_{idx:05d}_{model}.png"
                    plt.figure(figsize=(4, 4), dpi=args.dpi)
                    rmse_img = normalize_01(rmse_img)
                    # plt.imshow(rmse_img, cmap="gray", vmin=vmin, vmax=vmax)
                    plt.imshow(rmse_img, cmap="gray")
                    plt.axis("off")
                    plt.title(f"{model} : {rmse_vals[model]:.4f}", fontsize=10)
                    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
                    plt.close()


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
    print(f"[SAVE] PNG directory: {png_dir}")

    import pandas as pd

    num_samples = len(metrics["FBP_INPUT"]["psnr"])

    rows = []
    for i in range(num_samples):
        rows.append({
            "test_idx": i,

            "fbp_psnr": metrics["FBP_INPUT"]["psnr"][i],
            "fbp_ssim": metrics["FBP_INPUT"]["ssim"][i],
            "fbp_rmse": metrics["FBP_INPUT"]["rmse"][i],

            "unet_psnr": metrics["UNet"]["psnr"][i],
            "unet_ssim": metrics["UNet"]["ssim"][i],
            "unet_rmse": metrics["UNet"]["rmse"][i],

            "transunet_psnr": metrics["TransUNet"]["psnr"][i],
            "transunet_ssim": metrics["TransUNet"]["ssim"][i],
            "transunet_rmse": metrics["TransUNet"]["rmse"][i],

            "wavtransunet_psnr": metrics["WavResTransUNet"]["psnr"][i],
            "wavtransunet_ssim": metrics["WavResTransUNet"]["ssim"][i],
            "wavtransunet_rmse": metrics["WavResTransUNet"]["rmse"][i],
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "metrics_per_test.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] Metrics CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
