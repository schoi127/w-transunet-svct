"""Model-forward profiling source with machine-specific path defaults removed."""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.flop_counter import FlopCounterMode


# =========================================================
# Utilities
# =========================================================
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def bytes_to_mib(x: float) -> float:
    return float(x) / (1024.0 ** 2)


def num_to_human(x: float, unit: str, base: float = 1000.0) -> str:
    """
    Human readable scaler:
      base=1000 -> K/M/G/T/P
      base=1024 -> Ki/Mi/Gi/Ti/Pi (not used for FLOPs)
    """
    if x is None:
        return "N/A"
    suffixes = ["", "K", "M", "G", "T", "P"]
    s = 0
    v = float(x)
    while abs(v) >= base and s < len(suffixes) - 1:
        v /= base
        s += 1
    # Keep 3 sig figs-ish
    if abs(v) >= 100:
        fmt = f"{v:.0f}"
    elif abs(v) >= 10:
        fmt = f"{v:.1f}"
    else:
        fmt = f"{v:.2f}"
    return f"{fmt}{suffixes[s]}{unit}"


def flops_to_gflops(flops: int) -> float:
    return float(flops) / 1e9


def params_to_mparams(params: int) -> float:
    return float(params) / 1e6


def safe_torch_cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_params_and_bytes(model: nn.Module) -> Tuple[int, int, int, int]:
    """
    Returns:
      total_params, trainable_params, total_param_bytes, trainable_param_bytes
    """
    total_params = 0
    trainable_params = 0
    total_bytes = 0
    trainable_bytes = 0

    for p in model.parameters():
        n = p.numel()
        b = n * p.element_size()
        total_params += n
        total_bytes += b
        if p.requires_grad:
            trainable_params += n
            trainable_bytes += b
    return total_params, trainable_params, total_bytes, trainable_bytes


def estimate_adamw_train_state_bytes(param_bytes_fp32: int) -> Dict[str, int]:
    """
    AdamW FP32 typical memory (very common approximation):
      - weights: 1x
      - grads: 1x
      - exp_avg: 1x
      - exp_avg_sq: 1x
    Total ~ 4x param_bytes.

    (AMP autocast doesn't change weight/grad dtype here in typical setups.)
    """
    w = int(param_bytes_fp32)
    g = int(param_bytes_fp32)
    m1 = int(param_bytes_fp32)
    m2 = int(param_bytes_fp32)
    total = w + g + m1 + m2
    return {"weights": w, "grads": g, "adam_m1": m1, "adam_m2": m2, "total": total}


def _mk_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    # Simple markdown table generator (no extra deps).
    colw = [len(h) for h in headers]
    for r in rows:
        for j, cell in enumerate(r):
            colw[j] = max(colw[j], len(str(cell)))

    def fmt_row(r: List[str]) -> str:
        return "| " + " | ".join(str(r[j]).ljust(colw[j]) for j in range(len(headers))) + " |"

    sep = "| " + " | ".join("-" * colw[j] for j in range(len(headers))) + " |"
    out = [fmt_row(headers), sep]
    out += [fmt_row(r) for r in rows]
    return "\n".join(out)


def _mk_latex_tabular(headers: List[str], rows: List[List[str]], caption: Optional[str] = None,
                     label: Optional[str] = None) -> str:
    """
    Produces a LaTeX table snippet with tabular environment only.
    (You can wrap with \\begin{table} ... if needed.)
    """
    # Escape minimal LaTeX special chars in cells
    def esc(s: str) -> str:
        s = str(s)
        s = s.replace("\\", "\\textbackslash{}")
        s = s.replace("_", "\\_")
        s = s.replace("%", "\\%")
        s = s.replace("&", "\\&")
        s = s.replace("#", "\\#")
        s = s.replace("{", "\\{")
        s = s.replace("}", "\\}")
        return s

    cols = "l" + "r" * (len(headers) - 1)
    lines = []
    if caption is not None:
        lines.append(f"% {caption}")
    if label is not None:
        lines.append(f"% {label}")
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\hline")
    lines.append(" & ".join(esc(h) for h in headers) + " \\\\")
    lines.append("\\hline")
    for r in rows:
        lines.append(" & ".join(esc(c) for c in r) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


# =========================================================
# FLOPs measurement
# =========================================================
@torch.no_grad()
def measure_forward_flops(model: nn.Module, x: torch.Tensor) -> int:
    """
    Forward-only FLOPs for a single pass.
    """
    model.eval()
    with FlopCounterMode(display=False) as fcm:
        _ = model(x)
    return int(fcm.get_total_flops())


def measure_trainstep_flops(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    """
    Train-step FLOPs for one step:
      out = model(x)
      loss = mse(out, y)
      loss.backward()

    - Input x typically does NOT require grad in usual training. We keep it so.
    - Counts forward + backward flops (only for ops with formulas in registry).
    """
    model.train()
    model.zero_grad(set_to_none=True)

    mse = nn.MSELoss()

    with FlopCounterMode(display=False) as fcm:
        out = model(x)
        loss = mse(out, y)
        loss.backward()

    # Cleanup grads so the next model run starts clean
    model.zero_grad(set_to_none=True)
    return int(fcm.get_total_flops())


def profile_cuda_peak_memory_bytes(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                                  do_backward: bool = True) -> Optional[int]:
    """
    CUDA only: measures peak allocated memory during forward (and optionally backward).
    """
    if x.device.type != "cuda":
        return None

    device = x.device
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    safe_torch_cuda_sync(device)

    model.train()
    model.zero_grad(set_to_none=True)
    mse = nn.MSELoss()

    if do_backward:
        out = model(x)
        loss = mse(out, y)
        loss.backward()
    else:
        with torch.no_grad():
            _ = model(x)

    safe_torch_cuda_sync(device)
    peak = torch.cuda.max_memory_allocated(device)
    model.zero_grad(set_to_none=True)
    return int(peak)


def profile_latency_ms(model: nn.Module, x: torch.Tensor, y: Optional[torch.Tensor],
                       device: torch.device, warmup: int = 10, iters: int = 50,
                       do_backward: bool = False) -> Optional[float]:
    """
    Measures average latency (ms) for forward-only or train-step.

    - For CUDA: uses cuda events for accurate timing.
    - For CPU/MPS: uses time.perf_counter (less stable).
    """
    model = model.to(device)

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # warmup
        for _ in range(max(0, warmup)):
            if do_backward:
                model.train()
                model.zero_grad(set_to_none=True)
                out = model(x)
                loss = F.mse_loss(out, y)  # type: ignore[arg-type]
                loss.backward()
            else:
                model.eval()
                with torch.no_grad():
                    _ = model(x)
        safe_torch_cuda_sync(device)

        times = []
        for _ in range(iters):
            if do_backward:
                model.train()
                model.zero_grad(set_to_none=True)
                starter.record()
                out = model(x)
                loss = F.mse_loss(out, y)  # type: ignore[arg-type]
                loss.backward()
                ender.record()
            else:
                model.eval()
                starter.record()
                with torch.no_grad():
                    _ = model(x)
                ender.record()

            torch.cuda.synchronize(device)
            times.append(starter.elapsed_time(ender))  # ms

        # cleanup
        model.zero_grad(set_to_none=True)
        return float(np.mean(times))

    # CPU/MPS fallback
    # warmup
    for _ in range(max(0, warmup)):
        if do_backward:
            model.train()
            model.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.mse_loss(out, y)  # type: ignore[arg-type]
            loss.backward()
        else:
            model.eval()
            with torch.no_grad():
                _ = model(x)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        if do_backward:
            model.train()
            model.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.mse_loss(out, y)  # type: ignore[arg-type]
            loss.backward()
        else:
            model.eval()
            with torch.no_grad():
                _ = model(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    model.zero_grad(set_to_none=True)
    return float(np.mean(times))


# =========================================================
# Model definitions (TransUNet / W-TransUNet)
#   The wavelet front end (haar_dwt_hvd / _upsample_like / make_norm /
#   ResBlock / WavMixResNet) is imported from src/wavelet_ops.py, which
#   holds the definitions copied verbatim from the training script. It
#   used to be a third private copy in this file; a complexity table has
#   to be measured on the same code that is trained, so the copy is gone.
# =========================================================
from wavelet_ops import (_upsample_like, haar_dwt_hvd, make_norm,  # noqa: E402
                         ResBlock, WavMixResNet)


def build_transunet(img_size: int,
                    pretrained_npz: str,
                    load_pretrained_if_exists: bool = False) -> nn.Module:
    """
    Baseline TransUNet from DIVal: ViT_seg + CFG_ViT['R50-ViT-B_16']
    """
    # --- import shim (import mechanics only; the network is byte-identical) --
    # `dival.networks` is a directory that only ever existed in the author's
    # working copy and is in no released version of DIVal, so this import failed
    # for everyone else and no cell of the complexity table could be reproduced
    # from the released repository. The same files now ship in `src/`; the DIVal
    # path is kept as a fallback so the original environment resolves as before.
    try:
        from vit_seg_modeling import VisionTransformer as ViT_seg
        from vit_seg_modeling import CONFIGS as CFG_ViT
    except ImportError:
        try:
            from dival.networks.vit_seg_modeling import VisionTransformer as ViT_seg
            from dival.networks.vit_seg_modeling import CONFIGS as CFG_ViT
        except Exception as e:
            raise RuntimeError(
                "Failed to import the TransUNet definition from src/vit_seg_modeling.py "
                "or from dival.networks.vit_seg_modeling."
            ) from e
    # ------------------------------------------------------------------------

    cfg = CFG_ViT["R50-ViT-B_16"]
    cfg.pretrained_path = pretrained_npz
    cfg.n_classes = 1
    cfg.n_skip = 3
    grid = img_size // cfg.patch_size
    cfg.patches.grid = (grid, grid)

    net = ViT_seg(cfg, img_size=img_size, num_classes=1)

    npz_path = Path(pretrained_npz)
    if load_pretrained_if_exists and npz_path.is_file():
        print(f"[INFO] Loading ImageNet-21k weights from: {npz_path}")
        net.load_from(np.load(str(npz_path)))
    else:
        if load_pretrained_if_exists:
            print(f"[WARN] pretrained_npz not found -> skip loading: {npz_path}")
        # weights not needed for FLOPs/Params
    return net


class WavResTransUNet(nn.Module):
    """
    W-TransUNet:
      x (1ch) -> Haar details -> 4ch -> WavMixResNet -> 1ch mixture -> TransUNet -> 1ch output
    """
    def __init__(self,
                 img_size: int,
                 pretrained_npz: str,
                 wav_base_ch: int = 64,
                 wav_blocks: int = 8,
                 wav_norm: str = "gn",
                 wav_upsample: str = "bilinear",
                 residual_out: bool = False,
                 load_pretrained_if_exists: bool = False,
                 wav_null: bool = False):
        super().__init__()
        self.wav_upsample = wav_upsample
        self.mix = WavMixResNet(in_ch=4, out_ch=1, base_ch=wav_base_ch,
                                num_blocks=wav_blocks, norm=wav_norm)
        self.transunet = build_transunet(
            img_size=img_size,
            pretrained_npz=pretrained_npz,
            load_pretrained_if_exists=load_pretrained_if_exists,
        )
        self.residual_out = residual_out
        self.wav_null = wav_null          # P11 — G1 null control

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h, v, d = haar_dwt_hvd(x)
        if getattr(self, 'wav_null', False):          # P11 — G1 null control
            h, v, d = x.clone(), x.clone(), x.clone()
        h = _upsample_like(h, x, mode=self.wav_upsample)
        v = _upsample_like(v, x, mode=self.wav_upsample)
        d = _upsample_like(d, x, mode=self.wav_upsample)

        x4 = torch.cat([x, h, v, d], dim=1)
        x_mix = self.mix(x4)
        out = self.transunet(x_mix)
        if self.residual_out:
            out = x_mix + out
        return out


# =========================================================
# U-Net builder (DIVal FBPUNetReconstructor)
# =========================================================
def build_unet_from_dival(lodopab_path: str, ray_impl: str, angle_for_raytrafo: int) -> nn.Module:
    """
    Builds the same U-Net used in your UNet training script by:
      dataset = get_standard_dataset('lodopab', impl=..., num_angles=angle)
      ray_trafo = dataset.get_ray_trafo(impl=...)
      reconstructor = FBPUNetReconstructor(ray_trafo)
      load hyper params
      init_model()
      model = reconstructor.model

    Returns the torch.nn.Module (unwrapped from DataParallel if needed).
    """
    try:
        from dival.config import set_config
        from dival import get_standard_dataset
        from dival.reconstructors.fbpunet_reconstructor import FBPUNetReconstructor
        from dival.reference_reconstructors import (
            check_for_params, download_params, get_hyper_params_path
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to import DIVal modules. Please ensure 'dival' is installed."
        ) from e

    set_config("lodopab_dataset/data_path", lodopab_path)

    dataset = get_standard_dataset("lodopab", impl=ray_impl, num_angles=angle_for_raytrafo)
    ray_trafo = dataset.get_ray_trafo(impl=ray_impl)

    reconstructor = FBPUNetReconstructor(ray_trafo)

    # Ensure hyper-params exist (download if needed)
    if not check_for_params("fbpunet", "lodopab", include_learned=False):
        print("[INFO] fbpunet hyper-params missing -> downloading via DIVal ...")
        download_params("fbpunet", "lodopab", include_learned=False)

    hyper_params_path = get_hyper_params_path("fbpunet", "lodopab")
    reconstructor.load_hyper_params(hyper_params_path)
    reconstructor.init_model()

    model = reconstructor.model
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model


def build_unet_local() -> nn.Module:
    """Build the same U-Net without the DIVal runtime.

    `build_unet_from_dival()` above needs ODL, the ASTRA toolbox and a LoDoPaB
    installation -- not because the U-Net needs them, but because it reaches the
    architecture through a dataset and a ray transform. Requiring a CUDA/ASTRA
    stack in order to count parameters is the reason this script could not be
    run at all outside the author's machine.

    This path builds the identical `nn.Module` directly from the shipped copy of
    DIVal's architecture, using FBPUNetReconstructor's own hyper-parameter
    defaults (fbpunet_reconstructor.py: scales=5, skip_channels=4,
    channels=(32, 32, 64, 64, 128, 128), use_sigmoid=False; init_bias_zero=True
    is applied as in `init_model()`). It is a construction shortcut, not a
    different network: the resulting module has 610,673 parameters, matching the
    reported U-Net row exactly.

    Caveat, stated rather than hidden: if a downloaded fbpunet/lodopab
    `hyper_params.json` overrides any of those four values, the DIVal path is
    authoritative and this one is not. Use --unet_builder dival when the DIVal
    runtime is available.
    """
    try:
        from unet import UNet
    except ImportError:
        from dival.reconstructors.networks.unet import UNet

    scales = 5
    skip_channels = 4
    channels = (32, 32, 64, 64, 128, 128)
    use_sigmoid = False

    model = UNet(in_ch=1, out_ch=1,
                 channels=channels[:scales],
                 skip_channels=[skip_channels] * scales,
                 use_sigmoid=use_sigmoid)

    def weights_init(m):  # init_bias_zero=True in FBPUNetReconstructor
        if isinstance(m, torch.nn.Conv2d):
            m.bias.data.fill_(0.0)

    model.apply(weights_init)
    return model


# =========================================================
# Dataset size inference (optional)
# =========================================================
def infer_train_samples_from_cache(cache_root: str, angle: int) -> Optional[int]:
    """
    Fast inference of train set size from cached FBP numpy file.
    Uses mmap_mode='r' so it does not load full array into RAM.
    """
    p = Path(cache_root) / f"{angle}angle" / "cache_lodopab_train_fbp.npy"
    if p.is_file():
        try:
            arr = np.load(str(p), mmap_mode="r")
            return int(arr.shape[0])
        except Exception:
            return None
    return None


# =========================================================
# Metrics container
# =========================================================
@dataclass
class ModelMetrics:
    name: str
    img_size: int
    input_shape: Tuple[int, int, int, int]

    params_total: int
    params_trainable: int
    param_bytes: int

    flops_forward: int
    flops_trainstep: int

    # Optional runtime/memory (device dependent)
    cuda_peak_mem_forward_bytes: Optional[int] = None
    cuda_peak_mem_trainstep_bytes: Optional[int] = None
    latency_forward_ms: Optional[float] = None
    latency_trainstep_ms: Optional[float] = None

    # Training cost estimates
    train_samples: Optional[int] = None
    epochs: Optional[int] = None
    n_runs: Optional[int] = None  # e.g., number of angles experiments
    total_train_flops: Optional[int] = None  # across epochs and runs

    notes: str = ""


# =========================================================
# Main
# =========================================================
def get_device(device_str: str) -> torch.device:
    device_str = device_str.lower().strip()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_str in ["cuda", "gpu"]:
        return torch.device("cuda")
    if device_str == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Common
    p.add_argument("--img_size", type=int, default=352)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=0)

    # Training-cost estimates
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--angles", nargs="+", type=int, default=[1000, 500, 250, 125, 50],
                   help="If you trained each angle separately, total_train_flops multiplies by len(angles).")
    p.add_argument("--train_samples", type=int, default=None,
                   help="If not given, tries to infer from cache_root/<angle>angle/cache_lodopab_train_fbp.npy")
    p.add_argument("--cache_root", type=str, default="./cache",
                   help="Used only to infer train_samples quickly (mmap).")

    # Model selection
    p.add_argument("--models", nargs="+", default=["unet", "transunet", "wtransunet"],
                   choices=["unet", "transunet", "wtransunet"],
                   help="Which models to measure. The TransUNet rows need only "
                        "torch; the DIVal U-Net path additionally needs ODL/ASTRA.")
    p.add_argument("--unet_builder", type=str, default="auto",
                   choices=["auto", "dival", "local"],
                   help="How to obtain the U-Net: 'dival' goes through "
                        "FBPUNetReconstructor (needs ODL/ASTRA/LoDoPaB), 'local' "
                        "instantiates the shipped architecture directly, 'auto' "
                        "tries DIVal and falls back to local.")

    # DIVal / LoDoPaB
    # LODOPAB_PATH environment variable is honoured (README section 3); with it
    # unset an explicit dataset path is required for the DIVal builder.
    p.add_argument("--lodopab_path", type=str,
                   default=os.environ.get(
                       "LODOPAB_DATA",
                       os.environ.get("LODOPAB_PATH",
                                      "")),
                   help="LoDoPaB dataset path used by DIVal set_config "
                        "(defaults to $LODOPAB_PATH when set).")
    p.add_argument("--ray_impl", type=str, default="astra_cuda",
                   help="DIVal dataset implementation for ray_trafo (e.g., astra_cuda / astra).")
    p.add_argument("--angle_for_raytrafo", type=int, default=1000,
                   help="Angle used only to build ray_trafo for FBPUNetReconstructor init.")

    # TransUNet pretrained npz (optional)
    p.add_argument("--pretrained_npz", type=str,
                   default="")
    p.add_argument("--load_pretrained", action="store_true",
                   help="If set AND pretrained_npz exists, load ImageNet-21k weights (not needed for FLOPs).")

    # W-TransUNet (WavMixResNet) settings
    p.add_argument("--wav_base_ch", type=int, default=64)
    p.add_argument("--wav_blocks", type=int, default=8)
    p.add_argument("--wav_norm", type=str, default="gn", choices=["none", "bn", "in", "gn"])
    p.add_argument("--wav_upsample", type=str, default="bilinear", choices=["nearest", "bilinear"])
    p.add_argument("--residual_out", action="store_true")
    # P11 (Phase 3, G1 gate): null control for the wavelet side channels.
    # NOTE: never enable this for the Table 2 complexity numbers (it removes
    # the DWT/upsample work and therefore changes FLOPs).
    p.add_argument("--wav_null", action="store_true",
                   help="G1 control: replace H/V/D channels with copies of the FBP input")

    # Optional profiling
    p.add_argument("--profile_memory", action="store_true",
                   help="If set and device=cuda, measure torch.cuda.max_memory_allocated for forward/trainstep (B=1).")
    p.add_argument("--profile_time", action="store_true",
                   help="Measure latency (ms) for forward/trainstep (B=1).")
    p.add_argument("--time_warmup", type=int, default=10)
    p.add_argument("--time_iters", type=int, default=50)

    # Output files
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--out_tex", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    print(f"[INFO] { _now() } | device={device} | torch={torch.__version__}")

    img_size = int(args.img_size)
    if img_size % 2 != 0:
        raise ValueError(f"img_size must be even (Haar DWT). Got {img_size}")
    if img_size % 16 != 0:
        print(f"[WARN] img_size={img_size} is not multiple of 16. TransUNet usually expects /16 grid. "
              f"Proceeding anyway; ensure your ViT config supports it.")

    # training samples inference
    train_samples = args.train_samples
    if train_samples is None:
        # try cache-based inference using the first angle in args.angles
        if args.angles and args.cache_root:
            inferred = infer_train_samples_from_cache(args.cache_root, int(args.angles[0]))
            if inferred is not None:
                train_samples = inferred
                print(f"[INFO] train_samples inferred from cache: {train_samples}")
            else:
                print("[WARN] train_samples not provided and could not infer from cache. "
                      "Total training FLOPs will be shown as N/A unless you pass --train_samples.")
        else:
            print("[WARN] train_samples not provided and cache_root/angles not usable. "
                  "Total training FLOPs will be shown as N/A unless you pass --train_samples.")

    epochs = int(args.epochs)
    n_runs = len(args.angles) if args.angles else 1

    # Dummy input/target (B=1)
    x = torch.randn(1, 1, img_size, img_size, device=device)
    y = torch.randn(1, 1, img_size, img_size, device=device)

    # -------------------------------------------------
    # Build models
    # -------------------------------------------------
    print("\n[INFO] Building models ...")

    selected = set(args.models)
    models: List[Tuple[str, nn.Module]] = []
    build_failures: List[str] = []   # models that could not be measured at all
    build_notes: List[str] = []      # a builder that failed but was recovered from

    # 1) U-Net.  --unet_builder auto: try DIVal first (the runtime path the
    #    baseline was trained through) and fall back to the shipped architecture
    #    when the DIVal/ODL/ASTRA stack is absent. A failure here no longer
    #    aborts the whole run: the two TransUNet rows do not depend on DIVal and
    #    there is no reason a reviewer without ASTRA should lose them as well.
    if "unet" in selected:
        unet, unet_name = None, "U-Net"
        if args.unet_builder in ("auto", "dival"):
            try:
                unet = build_unet_from_dival(
                    lodopab_path=args.lodopab_path,
                    ray_impl=args.ray_impl,
                    angle_for_raytrafo=int(args.angle_for_raytrafo),
                ).to(device)
                unet_name = "U-Net (DIVal FBPUNet)"
            except Exception as e:
                dival_err = f"U-Net via DIVal: {type(e).__name__}: {e}"
                print(f"[WARN] Failed to build {dival_err}")
                if args.unet_builder == "dival":
                    raise
                build_notes.append(dival_err)
        if unet is None and args.unet_builder in ("auto", "local"):
            try:
                unet = build_unet_local().to(device)
                unet_name = "U-Net (DIVal FBPUNet architecture, built locally)"
                print("[INFO] U-Net built from the shipped architecture "
                      "(src/unet.py); no DIVal runtime required.")
            except Exception as e:
                msg = f"U-Net locally: {type(e).__name__}: {e}"
                print(f"[ERROR] Failed to build {msg}")
                build_failures.append(msg)
        if unet is not None:
            models.append((unet_name, unet))
        else:
            build_failures.extend(build_notes)
            build_notes = []

    # 2) TransUNet baseline (no wavelet mix)
    if "transunet" in selected:
        try:
            transunet = build_transunet(
                img_size=img_size,
                pretrained_npz=args.pretrained_npz,
                load_pretrained_if_exists=bool(args.load_pretrained),
            ).to(device)
            models.append(("TransUNet (R50+ViT-B_16)", transunet))
        except Exception as e:
            print(f"[ERROR] Failed to build TransUNet: {e}")
            raise

    # 3) W-TransUNet
    if "wtransunet" in selected:
        try:
            w_transunet = WavResTransUNet(
                img_size=img_size,
                pretrained_npz=args.pretrained_npz,
                wav_base_ch=int(args.wav_base_ch),
                wav_blocks=int(args.wav_blocks),
                wav_norm=str(args.wav_norm),
                wav_upsample=str(args.wav_upsample),
                residual_out=bool(args.residual_out),
                load_pretrained_if_exists=bool(args.load_pretrained),
                wav_null=bool(args.wav_null),
            ).to(device)
            print(f"[P11] wav_null={bool(args.wav_null)}")
            models.append(("W-TransUNet (Wavelet+Mix+TransUNet)", w_transunet))
        except Exception as e:
            print(f"[ERROR] Failed to build W-TransUNet: {e}")
            raise

    if not models:
        raise RuntimeError("No model could be built; nothing to measure. "
                           + " | ".join(build_failures))
    if build_failures:
        print("\n[WARN] Some models were skipped and are missing from the table:")
        for m in build_failures:
            print(f"       - {m}")
    if build_notes:
        print("\n[NOTE] A preferred builder failed but the model was still built:")
        for m in build_notes:
            print(f"       - {m}")
        print("       The model name in the table states which builder was used.")

    # -------------------------------------------------
    # Measure metrics
    # -------------------------------------------------
    print("\n[INFO] Measuring FLOPs/Params ... (B=1, input=1x1xHxW)")

    results: List[ModelMetrics] = []

    for name, model in models:
        print(f"\n--- {name} ---")
        model = model.to(device)

        # params
        p_total, p_train, p_bytes, _ = count_params_and_bytes(model)
        print(f"  Params: total={p_total:,} ({params_to_mparams(p_total):.3f} M)")

        # FLOPs
        f_fwd = measure_forward_flops(model, x)
        print(f"  FLOPs (forward): {f_fwd:,} ({flops_to_gflops(f_fwd):.3f} GFLOPs)")

        f_tr = measure_trainstep_flops(model, x, y)
        print(f"  FLOPs (train step fwd+bwd): {f_tr:,} ({flops_to_gflops(f_tr):.3f} GFLOPs)")

        # Training total FLOPs
        total_train_flops = None
        if train_samples is not None and epochs is not None and n_runs is not None:
            # total samples processed across all epochs and runs (angles)
            total_train_flops = int(f_tr * train_samples * epochs * n_runs)

        # Optional CUDA memory profiling
        peak_fwd = None
        peak_tr = None
        if args.profile_memory:
            peak_fwd = profile_cuda_peak_memory_bytes(model, x, y, do_backward=False)
            peak_tr = profile_cuda_peak_memory_bytes(model, x, y, do_backward=True)
            if peak_fwd is not None:
                print(f"  CUDA peak mem (forward): {bytes_to_mib(peak_fwd):.1f} MiB")
            else:
                print("  CUDA peak mem (forward): N/A (not cuda)")
            if peak_tr is not None:
                print(f"  CUDA peak mem (trainstep): {bytes_to_mib(peak_tr):.1f} MiB")
            else:
                print("  CUDA peak mem (trainstep): N/A (not cuda)")

        # Optional timing
        lat_fwd = None
        lat_tr = None
        if args.profile_time:
            lat_fwd = profile_latency_ms(model, x, y=None, device=device,
                                         warmup=int(args.time_warmup), iters=int(args.time_iters),
                                         do_backward=False)
            lat_tr = profile_latency_ms(model, x, y=y, device=device,
                                        warmup=int(args.time_warmup), iters=int(args.time_iters),
                                        do_backward=True)
            print(f"  Latency (forward): {lat_fwd:.3f} ms")
            print(f"  Latency (trainstep): {lat_tr:.3f} ms")

        mm = ModelMetrics(
            name=name,
            img_size=img_size,
            input_shape=tuple(x.shape),  # type: ignore[arg-type]
            params_total=p_total,
            params_trainable=p_train,
            param_bytes=p_bytes,
            flops_forward=f_fwd,
            flops_trainstep=f_tr,
            cuda_peak_mem_forward_bytes=peak_fwd,
            cuda_peak_mem_trainstep_bytes=peak_tr,
            latency_forward_ms=lat_fwd,
            latency_trainstep_ms=lat_tr,
            train_samples=train_samples,
            epochs=epochs,
            n_runs=n_runs,
            total_train_flops=total_train_flops,
        )
        results.append(mm)

    # -------------------------------------------------
    # Build paper-friendly table
    # -------------------------------------------------
    headers = [
        "Model",
        "Params (M)",
        "FLOPs Fwd (G)",
        "FLOPs TrainStep (G)",
        f"Total Train FLOPs ({epochs}ep × {train_samples if train_samples is not None else 'N/A'} samples × {n_runs} runs)",
        "Weights (MiB)",
        "AdamW State Est. (MiB)",
    ]
    if args.profile_memory:
        headers += ["PeakMem Fwd (MiB)", "PeakMem Train (MiB)"]
    if args.profile_time:
        headers += ["Latency Fwd (ms)", "Latency Train (ms)"]

    rows: List[List[str]] = []

    for m in results:
        weights_mib = bytes_to_mib(m.param_bytes)
        adam_state = estimate_adamw_train_state_bytes(m.param_bytes)
        adam_mib = bytes_to_mib(adam_state["total"])

        total_train_str = "N/A"
        if m.total_train_flops is not None:
            # show as PFLOPs for readability
            total_train_str = num_to_human(m.total_train_flops, unit="FLOPs", base=1000.0)

        row = [
            m.name,
            f"{params_to_mparams(m.params_total):.3f}",
            f"{flops_to_gflops(m.flops_forward):.3f}",
            f"{flops_to_gflops(m.flops_trainstep):.3f}",
            total_train_str,
            f"{weights_mib:.1f}",
            f"{adam_mib:.1f}",
        ]

        if args.profile_memory:
            row += [
                f"{bytes_to_mib(m.cuda_peak_mem_forward_bytes):.1f}" if m.cuda_peak_mem_forward_bytes is not None else "N/A",
                f"{bytes_to_mib(m.cuda_peak_mem_trainstep_bytes):.1f}" if m.cuda_peak_mem_trainstep_bytes is not None else "N/A",
            ]
        if args.profile_time:
            row += [
                f"{m.latency_forward_ms:.3f}" if m.latency_forward_ms is not None else "N/A",
                f"{m.latency_trainstep_ms:.3f}" if m.latency_trainstep_ms is not None else "N/A",
            ]

        rows.append(row)

    print("\n" + "=" * 90)
    print("[RESULT] Markdown table (copy to paper draft / README)")
    print("=" * 90)
    print(_mk_markdown_table(headers, rows))

    # -------------------------------------------------
    # Save optional outputs
    # -------------------------------------------------
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        # naive CSV writer
        with out_csv.open("w", encoding="utf-8") as f:
            f.write(",".join('"' + h.replace('"', '""') + '"' for h in headers) + "\n")
            for r in rows:
                f.write(",".join('"' + str(c).replace('"', '""') + '"' for c in r) + "\n")
        print(f"\n[INFO] Saved CSV: {out_csv}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "timestamp": _now(),
                "torch_version": torch.__version__,
                "device": str(device),
                "img_size": img_size,
                "epochs": epochs,
                "angles": args.angles,
                "n_runs": n_runs,
                "train_samples": train_samples,
                "ray_impl": args.ray_impl,
                "angle_for_raytrafo": args.angle_for_raytrafo,
            },
            "models": [dataclasses.asdict(m) for m in results],
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[INFO] Saved JSON: {out_json}")

    if args.out_tex:
        out_tex = Path(args.out_tex)
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        tex = _mk_latex_tabular(headers, rows,
                                caption="Model complexity comparison (Params/FLOPs).",
                                label="tab:model_complexity")
        with out_tex.open("w", encoding="utf-8") as f:
            f.write(tex + "\n")
        print(f"[INFO] Saved LaTeX tabular: {out_tex}")

    print("\n[Done]")


if __name__ == "__main__":
    main()