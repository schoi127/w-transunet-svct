# Archival source; complete checkpoint-generating run provenance is unavailable.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------
# 0) IMPORT 및 환경 설정
# ------------------------------------------------------------------
from pathlib import Path
import os, sys, math, random, argparse, re
from typing import Optional, Dict, Any

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# headless 환경 PNG 저장을 위해 Agg backend 권장
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dival.config import set_config
# LoDoPaB location: set LODOPAB_PATH to point DIVal at your copy of the dataset
# (README section 3). With the variable unset the original hard-coded path is
# used, so behaviour on the machine that produced the paper is unchanged.
# P3 (Phase 3): honour LODOPAB_DATA first (the variable this study's run
# scripts export), then the README's LODOPAB_PATH, then the original
# hard-coded path so behaviour on the paper machine is unchanged.
LODOPAB_PATH = os.environ.get(
    'LODOPAB_DATA',
    os.environ.get('LODOPAB_PATH',
                   '/home/schoi/15_DIVAL/dival/dival/lodopab1'))
set_config('lodopab_dataset/data_path', LODOPAB_PATH)

from dival import get_standard_dataset
from dival.measure import PSNR
from odl.tomo import fbp_op

# --- import shim (import mechanics only; the network is byte-identical) ------
# The published runs imported the TransUNet definition from `dival.networks`,
# a directory that only ever existed in the author's working copy and is in no
# released version of DIVal, so this import failed for anyone else. The same
# files now ship in `src/`; the DIVal path is kept as a fallback so the original
# environment still resolves exactly as before.
try:
    from vit_seg_modeling import VisionTransformer as ViT_seg
    from vit_seg_modeling import CONFIGS as CFG_ViT
except ImportError:
    from dival.networks.vit_seg_modeling import VisionTransformer as ViT_seg
    from dival.networks.vit_seg_modeling import CONFIGS as CFG_ViT
# ---------------------------------------------------------------------------

# Wavelet front end: single shared implementation (see src/wavelet_ops.py).
# Previously duplicated verbatim in this file, inference.py and
# compute_metrics_models.py.
from wavelet_ops import (_upsample_like, haar_dwt_hvd, make_norm, ResBlock,
                         WavMixResNet, wavelet_detail_loss)

# Shared seeding / cuDNN controls (src/repro.py). Defaults reproduce this
# script's published behaviour: seed 0 and cudnn.benchmark = True.
from repro import add_repro_args, apply_cudnn_regime, seed_everything


# ------------------------------------------------------------------
# 1) Argument parser
# ------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()

    # TransUNet / 학습 기본
    parser.add_argument('--pretrained_npz', type=str,
                        default='/home/schoi/15_DIVAL/dival/dival/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz',
                        help='absolute path to R50+ViT-B_16.npz')
    parser.add_argument('--img_size', type=int, default=352,
                        help='input resolution (16의 배수 권장, 짝수 권장)')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--base_lr', type=float, default=3e-4)

    # angle 실험
    parser.add_argument('--angles', nargs='+', type=int,
                        default=[1000, 500, 250, 125, 50],
                        help='list of projection angle counts (1000의 약수여야 함)')
    parser.add_argument('--cache_root', type=str, default='./cache',
                        help='root dir to store FBP caches')
    parser.add_argument('--log_root', type=str, default='./logs/lodopab_wavres_transunet',
                        help='root dir for checkpoints')

    # DataLoader
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', action='store_true', help='pin_memory for CUDA')

    # AMP
    parser.add_argument('--amp', action='store_true', help='use torch.cuda.amp (CUDA only)')

    # WavResNet (mixture CNN) 설정
    parser.add_argument('--wav_base_ch', type=int, default=64,
                        help='base channels for WavResNet')
    parser.add_argument('--wav_blocks', type=int, default=8,
                        help='number of residual blocks in WavResNet')
    parser.add_argument('--wav_norm', type=str, default='gn',
                        choices=['none', 'bn', 'in', 'gn'],
                        help='normalization in WavResNet blocks')
    parser.add_argument('--wav_upsample', type=str, default='bilinear',
                        choices=['nearest', 'bilinear'],
                        help='upsampling mode for H/V/D details to img_size')

    # Loss
    parser.add_argument('--wav_loss_weight', type=float, default=0.05,
                        help='weight for wavelet-detail loss (0이면 비활성)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='gradient clipping max norm (0이면 비활성)')

    # 출력 residual 옵션(선택)
    parser.add_argument('--residual_out', action='store_true',
                        help='if set, final output = mixture_input + transunet_output')
    # P11 (Phase 3, G1 gate): null control for the wavelet side channels.
    parser.add_argument('--wav_null', action='store_true',
                        help='G1 control: replace H/V/D channels with copies of the FBP input')

    #  저장/시각화 주기 옵션
    parser.add_argument('--save_every', type=int, default=10,
                        help='save checkpoint every N epochs')
    parser.add_argument('--viz_every', type=int, default=10,
                        help='save validation comparison PNG every N epochs')
    parser.add_argument('--viz_n', type=int, default=5,
                        help='number of validation samples to visualize')
    parser.add_argument('--viz_seed', type=int, default=0,
                        help='seed to choose fixed validation samples for visualization')

    #  재시작/복구 옵션
    parser.add_argument('--resume', type=str, default='none',
                        help=("Resume training. "
                              "none(default) | auto | last | best | <path_to_ckpt>. "
                              "auto: last.pth -> latest epoch_*.pth -> best_model.pth"))
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help=("LEGACY weight-only ckpt용: "
                              "resume 파일에 epoch 정보가 없을 때 '마지막 완료 epoch'을 직접 지정. "
                              "예) best_model.pth가 epoch 68에서 저장되었다면 --resume_epoch 68"))
    parser.add_argument('--resume_strict', action='store_true',
                        help='use strict=True when loading state_dict (default: False)')

    # 재현성: seed / cuDNN 레짐 (기본값 = 논문 학습 시 사용된 값)
    add_repro_args(parser, default_cudnn='benchmark')

    return parser.parse_args()


# ------------------------------------------------------------------
# 2) Device & 재현성
# ------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

# `seed_everything` now lives in src/repro.py (imported above) so that the three
# training scripts seed identically. The body is unchanged.


# ------------------------------------------------------------------
# 3) 헬퍼 함수: 디렉토리, FBP 캐시, 체크포인트
# ------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def atomic_torch_save(obj: Any, path: Path):
    """중간에 죽어도 파일이 깨질 확률을 줄이기 위한 atomic save."""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)

def load_or_create_fbp(dataset, ray_trafo, split: str, cache_path: Path) -> np.ndarray:
    """
    split(train|validation)에 대해
    cache(.npy)이 있으면 로드, 없으면 FBP 계산 후 저장.
    """
    ensure_dir(cache_path.parent)
    if cache_path.is_file():
        print(f'  – load {split:10s} FBP from cache: {cache_path}')
        return np.load(cache_path)

    print(f'  – cache missing → computing {split:10s} FBP')
    sins = dataset.get_data_pairs(split)
    op_fbp  = fbp_op(ray_trafo, filter_type='Ram-Lak', frequency_scaling=1.0)
    fbp_split = [np.asarray(op_fbp(sin)) for sin, _ in tqdm(sins)]
    arr = np.stack(fbp_split, axis=0)
    np.save(cache_path, arr)
    return arr


def _infer_epoch_from_filename(p: Path) -> Optional[int]:
    m = re.search(r'epoch_(\d+)\.pth$', p.name)
    if m:
        return int(m.group(1))
    return None

def _find_latest_epoch_ckpt(log_dir: Path) -> Optional[Path]:
    candidates = sorted(log_dir.glob('epoch_*.pth'))
    if not candidates:
        return None
    # epoch_XXX.pth 중 XXX가 가장 큰 것 선택
    best = None
    best_ep = -1
    for p in candidates:
        ep = _infer_epoch_from_filename(p)
        if ep is not None and ep > best_ep:
            best_ep = ep
            best = p
    return best

def resolve_resume_path(log_dir: Path, resume_arg: str) -> Optional[Path]:
    if resume_arg is None:
        return None
    key = str(resume_arg).strip()
    if key == '' or key.lower() in ['none', 'no', 'false', '0']:
        return None

    key_l = key.lower()
    if key_l == 'best':
        p = log_dir / 'best_model.pth'
        return p if p.is_file() else None

    if key_l == 'last':
        p = log_dir / 'last.pth'
        if p.is_file():
            return p
        return _find_latest_epoch_ckpt(log_dir)

    if key_l == 'auto':
        p = log_dir / 'last.pth'
        if p.is_file():
            return p
        p = _find_latest_epoch_ckpt(log_dir)
        if p is not None:
            return p
        p = log_dir / 'best_model.pth'
        return p if p.is_file() else None

    # treat as explicit path
    p = Path(key)
    if p.is_file():
        return p
    # also try relative to log_dir
    p2 = log_dir / key
    if p2.is_file():
        return p2
    return None


def validate_psnr(net: nn.Module, va_loader: DataLoader, device: torch.device) -> float:
    net.eval()
    psnr_sum = 0.0
    n = len(va_loader.dataset)
    with torch.no_grad():
        for x, y in va_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = net(x)
            out_np = out[:, 0].detach().cpu().numpy()
            y_np   = y[:, 0].detach().cpu().numpy()
            for i in range(out_np.shape[0]):
                psnr_sum += PSNR(out_np[i], y_np[i])
    return psnr_sum / float(n)


def prepare_onecycle_param_groups_for_resume(
    optimizer: torch.optim.Optimizer,
    max_lr: float,
    div_factor: float,
    final_div_factor: float,
    base_momentum: float = 0.85,
    max_momentum: float = 0.95,
):
    """
    OneCycleLR를 last_epoch != -1로 생성하려면(=resume step 점프)
    optimizer.param_groups에 아래 key들이 이미 있어야 함.
    (원래는 scheduler state_dict로 복구됨)
    """
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', max_lr / div_factor)
        group.setdefault('max_lr', max_lr)
        group.setdefault('min_lr', (max_lr / div_factor) / final_div_factor)

        # OneCycleLR 기본: cycle_momentum=True
        group.setdefault('base_momentum', base_momentum)
        group.setdefault('max_momentum', max_momentum)


# ------------------------------------------------------------------
# 4) Dataset: center-crop (TransUNet 요구사항: 16의 배수)
# ------------------------------------------------------------------
class FBPCrop(Dataset):
    """Center‑crop to a multiple‑of‑16 patch"""
    def __init__(self, fbp: np.ndarray, gt: np.ndarray, size: int = 352):
        self.fbp, self.gt, self.size = fbp, gt, size

    def __len__(self):
        return len(self.fbp)

    def _crop(self, img: np.ndarray) -> np.ndarray:
        h0, w0 = img.shape
        dh, dw = (h0 - self.size) // 2, (w0 - self.size) // 2
        return img[dh:dh + self.size, dw:dw + self.size]

    def __getitem__(self, idx):
        x = torch.from_numpy(self._crop(self.fbp[idx])).unsqueeze(0).float()  # (1,H,W)
        y = torch.from_numpy(self._crop(self.gt[idx])).unsqueeze(0).float()   # (1,H,W)
        return x, y


# ------------------------------------------------------------------
# 5)-6) Wavelet front end (Haar DWT details + WavResNet mixture CNN)
#       moved verbatim to src/wavelet_ops.py and imported at the top of this
#       file, so training, inference and the complexity script all execute the
#       same code instead of three copies of it.
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 7) TransUNet 빌드
# ------------------------------------------------------------------
def build_transunet(img_size: int, pretrained_npz: str) -> nn.Module:
    cfg = CFG_ViT['R50-ViT-B_16']
    cfg.pretrained_path = pretrained_npz
    cfg.n_classes = 1
    cfg.n_skip = 3
    grid = img_size // cfg.patch_size
    cfg.patches.grid = (grid, grid)

    net = ViT_seg(cfg, img_size=img_size, num_classes=1)
    print('[INFO] Loading ImageNet‑21k weights …')
    net.load_from(np.load(cfg.pretrained_path))
    return net


# ------------------------------------------------------------------
# 8) WavResTransUNet (전체 모델)
# ------------------------------------------------------------------
class WavResTransUNet(nn.Module):
    """
    forward:
      x (FBP, 1ch) → wavelet(H,V,D) 추출 → (FBP,H,V,D)=4ch
      → WavResNet mixture (1ch) → TransUNet → output(1ch)
    """
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


# `wavelet_detail_loss` also lives in src/wavelet_ops.py (imported above).


# ------------------------------------------------------------------
# 8.5)  Validation PNG 저장 유틸
# ------------------------------------------------------------------
def _robust_vmin_vmax(arr: np.ndarray, lo=1.0, hi=99.0):
    """시각화용 robust min/max (percentile)"""
    vmin = float(np.percentile(arr, lo))
    vmax = float(np.percentile(arr, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax <= vmin):
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    return vmin, vmax

def save_val_comparison_png(
    net: nn.Module,
    ds_va: Dataset,
    device: torch.device,
    log_dir: Path,
    epoch: int,
    indices: np.ndarray,
    angle: int
):
    """
    지정 indices에 대해 FBP/GT/Output 비교 PNG 저장.
    한 row = 한 샘플, col = [FBP, GT, Output]
    """
    net.eval()

    xs, ys = [], []
    for idx in indices:
        x, y = ds_va[int(idx)]
        xs.append(x)
        ys.append(y)

    x = torch.stack(xs, dim=0).to(device)  # (N,1,H,W)
    y = torch.stack(ys, dim=0).to(device)

    with torch.no_grad():
        out = net(x)

    x_np = x[:, 0].detach().cpu().numpy()
    y_np = y[:, 0].detach().cpu().numpy()
    o_np = out[:, 0].detach().cpu().numpy()

    n = x_np.shape[0]
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(12, 4*n), dpi=150)

    # n=1일 때 axes 형태 보정
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["FBP", "GT", "Output"]
    for j in range(3):
        axes[0, j].set_title(col_titles[j], fontsize=12)

    for i in range(n):
        # 3개 이미지의 공통 스케일(샘플별)로 비교가 깔끔해짐
        concat = np.concatenate([x_np[i].ravel(), y_np[i].ravel(), o_np[i].ravel()])
        vmin, vmax = _robust_vmin_vmax(concat, lo=1.0, hi=99.0)

        psnr_i = PSNR(o_np[i], y_np[i])

        axes[i, 0].imshow(x_np[i], cmap='gray', vmin=vmin, vmax=vmax)
        axes[i, 1].imshow(y_np[i], cmap='gray', vmin=vmin, vmax=vmax)
        axes[i, 2].imshow(o_np[i], cmap='gray', vmin=vmin, vmax=vmax)

        axes[i, 0].set_ylabel(f"idx={int(indices[i])}\nPSNR={psnr_i:.2f} dB", fontsize=10)

        for j in range(3):
            axes[i, j].axis('off')

    fig.suptitle(f'Angle {angle} | Epoch {epoch:03d} | (FBP vs GT vs Output)', fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out_path = log_dir / f'val_compare_epoch_{epoch:03d}.png'
    fig.savefig(out_path)
    plt.close(fig)
    print(f'    [PNG] Saved validation comparison: {out_path}')


# ------------------------------------------------------------------
# 9) Train one angle
# ------------------------------------------------------------------
def train_one_angle(args, device, angle: int):
    print(f'\n{"="*70}\n[ANGLE {angle}]')

    # 9-1) Dataset & geometry
    dataset = get_standard_dataset('lodopab', impl='astra_cuda', num_angles=angle)
    ray_trafo = dataset.get_ray_trafo(impl='astra_cuda')

    # 9-2) FBP 캐시
    cache_dir = Path(args.cache_root) / f'{angle}angle'
    fbp_train = load_or_create_fbp(dataset, ray_trafo, 'train',
                                   cache_dir / 'cache_lodopab_train_fbp.npy')
    fbp_val   = load_or_create_fbp(dataset, ray_trafo, 'validation',
                                   cache_dir / 'cache_lodopab_validation_fbp.npy')

    # 9-3) GT 로드
    n_tr = fbp_train.shape[0]
    n_va = fbp_val.shape[0]
    gt_train = np.stack([gt for _, gt in dataset.get_data_pairs('train', n_tr)], axis=0)
    gt_val   = np.stack([gt for _, gt in dataset.get_data_pairs('validation', n_va)], axis=0)

    # 9-4) DataLoader
    ds_tr = FBPCrop(fbp_train, gt_train, size=args.img_size)
    ds_va = FBPCrop(fbp_val,   gt_val,   size=args.img_size)

    pin = bool(args.pin_memory) and (device.type == 'cuda')
    tr_loader = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                           num_workers=args.num_workers, pin_memory=pin,
                           persistent_workers=(args.num_workers > 0))
    va_loader = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                           num_workers=args.num_workers, pin_memory=pin,
                           persistent_workers=(args.num_workers > 0))

    # 9-5) Model
    if (args.img_size % 16) != 0:
        raise ValueError(f'img_size must be multiple of 16. Got {args.img_size}')
    if (args.img_size % 2) != 0:
        raise ValueError(f'img_size should be even for Haar DWT. Got {args.img_size}')

    net = WavResTransUNet(
        img_size=args.img_size,
        pretrained_npz=args.pretrained_npz,
        wav_base_ch=args.wav_base_ch,
        wav_blocks=args.wav_blocks,
        wav_norm=args.wav_norm,
        wav_upsample=args.wav_upsample,
        residual_out=args.residual_out,
        wav_null=bool(args.wav_null),
    ).to(device)
    print(f"[P11] wav_null={bool(args.wav_null)}")

    mse = nn.MSELoss()
    use_amp = bool(args.amp) and (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 9-7) Logging dir
    log_dir = Path(args.log_root) / f'{angle}angle'
    ensure_dir(log_dir)

    #  시각화용 validation 샘플 index 고정(매 epoch 동일 샘플로 비교)
    n_viz = int(min(args.viz_n, len(ds_va)))
    rng = np.random.RandomState(args.viz_seed)
    viz_indices = rng.choice(len(ds_va), size=n_viz, replace=False)
    viz_indices = np.sort(viz_indices)

    # -------------------------
    #  RESUME: checkpoint 로드
    # -------------------------
    resume_path = resolve_resume_path(log_dir, args.resume)
    resume_is_full = False
    resume_last_completed_epoch = 0  # "마지막 완료 epoch" (1-based). e.g., 60이면 다음은 61부터.
    best_psnr = -1.0

    resume_model_state_dict: Optional[Dict[str, torch.Tensor]] = None
    resume_ckpt_obj: Optional[Any] = None

    if resume_path is not None:
        print(f'[RESUME] requested: {args.resume} -> {resume_path}')
        # P7: torch>=2.6 flips torch.load's weights_only default to True, which
        # rejects these checkpoints (they store args/optimizer state, not just
        # tensors). Be explicit so the code works on both sides of that change.
        resume_ckpt_obj = torch.load(resume_path, map_location='cpu',
                                     weights_only=False)

        if isinstance(resume_ckpt_obj, dict) and ('model' in resume_ckpt_obj):
            #  full checkpoint
            resume_is_full = True
            resume_last_completed_epoch = int(resume_ckpt_obj.get('epoch', 0))
            best_psnr = float(resume_ckpt_obj.get('best_psnr', -1.0))

            state = resume_ckpt_obj['model']
            strict = bool(args.resume_strict)
            incompatible = net.load_state_dict(state, strict=strict)
            if (not strict) and (len(incompatible.missing_keys) > 0 or len(incompatible.unexpected_keys) > 0):
                print('[RESUME][WARN] Non-strict load. missing_keys / unexpected_keys:')
                print('  missing_keys   :', incompatible.missing_keys)
                print('  unexpected_keys:', incompatible.unexpected_keys)

            print(f'[RESUME] full checkpoint loaded. last_completed_epoch={resume_last_completed_epoch} '
                  f'best_psnr={best_psnr:.2f}')

        else:
            #  legacy: weight-only state_dict
            resume_is_full = False
            resume_model_state_dict = resume_ckpt_obj
            strict = bool(args.resume_strict)
            incompatible = net.load_state_dict(resume_model_state_dict, strict=strict)
            if (not strict) and (len(incompatible.missing_keys) > 0 or len(incompatible.unexpected_keys) > 0):
                print('[RESUME][WARN] Non-strict load. missing_keys / unexpected_keys:')
                print('  missing_keys   :', incompatible.missing_keys)
                print('  unexpected_keys:', incompatible.unexpected_keys)

            # epoch 추정: (1) --resume_epoch (2) 파일명(epoch_060.pth) (3) 0
            if args.resume_epoch is not None:
                resume_last_completed_epoch = int(args.resume_epoch)
            else:
                inferred = _infer_epoch_from_filename(resume_path)
                resume_last_completed_epoch = int(inferred) if inferred is not None else 0

            # legacy는 best_psnr 정보가 없으므로, 기존 best_model.pth가 있으면 그 PSNR로 초기화(덮어쓰기 방지)
            best_path = log_dir / 'best_model.pth'
            if best_path.is_file():
                try:
                    best_obj = torch.load(best_path, map_location='cpu',
                                          weights_only=False)  # P7
                    if isinstance(best_obj, dict) and ('best_psnr' in best_obj):
                        best_psnr = float(best_obj['best_psnr'])
                    else:
                        # weight-only best_model: 잠깐 로드해서 PSNR 계산
                        best_weights = best_obj['model'] if (isinstance(best_obj, dict) and 'model' in best_obj) else best_obj
                        net.load_state_dict(best_weights, strict=strict)
                        best_psnr = validate_psnr(net, va_loader, device)
                        net.load_state_dict(resume_model_state_dict, strict=strict)
                except Exception as e:
                    print(f'[RESUME][WARN] Failed to read best_model.pth to init best_psnr: {e}')
                    best_psnr = -1.0

            # best_psnr가 여전히 음수면(=best 파일 없음) 현재 모델 PSNR로 세팅
            if best_psnr < 0:
                best_psnr = validate_psnr(net, va_loader, device)

            print(f'[RESUME] legacy weights loaded. last_completed_epoch={resume_last_completed_epoch} '
                  f'(next epoch={resume_last_completed_epoch+1}) best_psnr_init={best_psnr:.2f}')

    else:
        best_psnr = -1.0

    # 9-6) Optimizer / Scheduler (resume 여부에 따라 last_epoch 처리)
    optimizer = AdamW(net.parameters(), lr=args.base_lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(ds_tr) / args.batch)
    print(f'[N2] epochs={args.epochs} steps_per_epoch={steps_per_epoch} total_steps={args.epochs * steps_per_epoch}')

    # OneCycle 설정(원본과 동일)
    pct_start = 0.1
    div_factor = 10.0
    final_div_factor = 100.0
    base_momentum = 0.85
    max_momentum = 0.95

    total_steps = args.epochs * steps_per_epoch

    if resume_path is not None and (not resume_is_full):
        # legacy: optimizer/scheduler state가 없으니, last_epoch를 점프해서 스케줄만 맞춤
        start_step = resume_last_completed_epoch * steps_per_epoch
        if start_step > 0:
            prepare_onecycle_param_groups_for_resume(
                optimizer,
                max_lr=args.base_lr,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
                base_momentum=base_momentum,
                max_momentum=max_momentum,
            )
            scheduler = OneCycleLR(
                optimizer,
                max_lr=args.base_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
                base_momentum=base_momentum,
                max_momentum=max_momentum,
                last_epoch=start_step - 1,
            )
            print(f'[RESUME] OneCycleLR fast-forward: start_step={start_step} (epoch {resume_last_completed_epoch} done)')
        else:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=args.base_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
                base_momentum=base_momentum,
                max_momentum=max_momentum,
            )
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.base_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
        )

    # full resume면 optimizer/scheduler/scaler 복구
    if resume_path is not None and resume_is_full:
        try:
            if 'optimizer' in resume_ckpt_obj and resume_ckpt_obj['optimizer'] is not None:
                optimizer.load_state_dict(resume_ckpt_obj['optimizer'])
            if 'scheduler' in resume_ckpt_obj and resume_ckpt_obj['scheduler'] is not None:
                scheduler.load_state_dict(resume_ckpt_obj['scheduler'])
            if use_amp and ('scaler' in resume_ckpt_obj) and (resume_ckpt_obj['scaler'] is not None):
                scaler.load_state_dict(resume_ckpt_obj['scaler'])
            print('[RESUME] optimizer/scheduler/scaler restored.')
        except Exception as e:
            print(f'[RESUME][WARN] Failed to restore optimizer/scheduler/scaler: {e}')
            print('             → continue with fresh optimizer/scheduler (weights are loaded).')

    # start epoch 계산
    start_ep_idx = int(resume_last_completed_epoch)
    if start_ep_idx >= args.epochs:
        print(f'[INFO] Already reached epochs={args.epochs}. Nothing to do. (start_ep_idx={start_ep_idx})')
        return

    # 9-8) Train loop
    for ep in range(start_ep_idx, args.epochs):
        epoch_num = ep + 1

        # ---- train
        net.train()
        running = 0.0

        for x, y in tr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = net(x)
                loss = mse(out, y)
                if args.wav_loss_weight > 0:
                    loss = loss + args.wav_loss_weight * wavelet_detail_loss(out, y)

            scaler.scale(loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item() * x.size(0)

        tr_loss = running / len(ds_tr)

        # ---- validation (PSNR)
        net.eval()
        psnr_sum = 0.0
        with torch.no_grad():
            for x, y in va_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                out = net(x)

                out_np = out[:, 0].detach().cpu().numpy()
                y_np   = y[:, 0].detach().cpu().numpy()
                for i in range(out_np.shape[0]):
                    psnr_sum += PSNR(out_np[i], y_np[i])

        psnr = psnr_sum / len(ds_va)

        print(f'[{angle:4d}] Epoch {epoch_num:03d}/{args.epochs:03d} | '
              f'train_loss={tr_loss:.5f} | val_PSNR={psnr:.2f} dB')

        # -------- checkpoint dict 생성
        train_state = {
            'epoch': epoch_num,  # last completed epoch (1-based)
            'angle': int(angle),
            'steps_per_epoch': int(steps_per_epoch),
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict() if (use_amp and scaler is not None) else None,
            'best_psnr': float(best_psnr),
            'args': vars(args),
        }

        #  best model 저장 (PSNR 갱신 시마다)
        if psnr > best_psnr:
            best_psnr = psnr
            train_state['best_psnr'] = float(best_psnr)
            atomic_torch_save(train_state, log_dir / 'best_model.pth')
            print(f'    [+] New best model saved ({best_psnr:.2f} dB)')

        #  last checkpoint: 매 epoch 끝에 저장(덮어쓰기)
        train_state['best_psnr'] = float(best_psnr)
        atomic_torch_save(train_state, log_dir / 'last.pth')

        #  매 save_every epoch마다 모델 저장 (+ 마지막 epoch)
        if (epoch_num % args.save_every == 0) or (epoch_num == args.epochs):
            ckpt_path = log_dir / f'epoch_{epoch_num:03d}.pth'
            atomic_torch_save(train_state, ckpt_path)
            print(f'    [CKPT] Saved checkpoint: {ckpt_path}')

        #  매 viz_every epoch마다 PNG 저장 (+ 마지막 epoch에도 저장)
        if (epoch_num % args.viz_every == 0) or (epoch_num == args.epochs):
            save_val_comparison_png(
                net=net,
                ds_va=ds_va,
                device=device,
                log_dir=log_dir,
                epoch=epoch_num,
                indices=viz_indices,
                angle=angle
            )


# ------------------------------------------------------------------
# 10) main
# ------------------------------------------------------------------
def main():
    args = get_args()

    device = get_device()
    seed_everything(args.seed)

    print(f'[INFO] Using device: {device}')
    if device.type == 'cuda':
        # default '--cudnn benchmark' == the published W-TransUNet regime
        apply_cudnn_regime(args.cudnn)

    for angle in args.angles:
        train_one_angle(args, device, angle)

    print('\n[Done] All angle experiments finished.')


if __name__ == '__main__':
    main()
