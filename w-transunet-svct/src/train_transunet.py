#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoDoPaB‑CT FBP → TransUNet 학습 스크립트
author : Sunghoon Choi (Electronics and Telecommunications Research Institute, KR)
date   : 2025‑07‑15
--------------------------------------------------------------------
필요 폴더 구조
├── ./cache/⟨angle⟩angle/cache_lodopab_train_fbp.npy
├── ./cache/⟨angle⟩angle/cache_lodopab_validation_fbp.npy
└── ./vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
"""

# ------------------------------------------------------------------
# 0) 공통 IMPORT 및 환경 설정
# ------------------------------------------------------------------
from pathlib import Path
import os, sys, math, random, argparse, logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from tqdm import tqdm

from dival.config import set_config                    # 데이터 경로 지정
set_config('lodopab_dataset/data_path',
           '/home/schoi/15_DIVAL/dival/dival/lodopab1') # <- **수정 필요 시 여기**

from dival import get_standard_dataset
from dival.measure import PSNR
from odl.tomo import fbp_op
from dival.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from dival.networks.vit_seg_modeling import CONFIGS as CFG_ViT

# ------------------------------------------------------------------
# 1) Argument parser (전체 학습에 공통)
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_npz', type=str,
                    default='/home/schoi/15_DIVAL/dival/dival/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz',
                    help='absolute path to R50+ViT-B_16.npz')
parser.add_argument('--img_size', type=int, default=352,
                    help='input resolution (16의 배수)')
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--batch',  type=int, default=64)
parser.add_argument('--base_lr', type=float, default=3e-4)
parser.add_argument('--angles', nargs='+', type=int,
                    # default=[1000, 500, 250, 125, 50],
                    default=[25, 10],
                    help='list of projection angle counts')
parser.add_argument('--cache_root', type=str, default='./cache',
                    help='root dir to store FBP caches')
parser.add_argument('--log_root',   type=str,
                    default='./logs/lodopab_transunet',
                    help='root dir for Tensor logs & checkpoints')
args = parser.parse_args()

# ------------------------------------------------------------------
# 2) Device & 재현성
# ------------------------------------------------------------------
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('mps')  if torch.backends.mps.is_available()
          else torch.device('cpu'))
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
print(f'[INFO] Using device: {device}')

# ------------------------------------------------------------------
# 3) 헬퍼 함수
# ------------------------------------------------------------------
def ensure_dir(p: Path):
    """mkdir ‑p"""
    p.mkdir(parents=True, exist_ok=True)

def load_or_create_fbp(dataset, ray_trafo, split: str,
                       cache_path: Path) -> np.ndarray:
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

class FBPCrop(Dataset):
    """Center‑crop to a multiple‑of‑16 patch (TransUNet 요구사항)"""
    def __init__(self, fbp, gt, size=352):
        self.fbp, self.gt, self.size = fbp, gt, size
    def __len__(self): return len(self.fbp)
    def _crop(self, img):
        h0, w0 = img.shape
        dh, dw = (h0-self.size)//2, (w0-self.size)//2
        return img[dh:dh+self.size, dw:dw+self.size]
    def __getitem__(self, idx):
        x = torch.from_numpy(self._crop(self.fbp[idx])).unsqueeze(0).float()
        y = torch.from_numpy(self._crop(self.gt[idx])).unsqueeze(0).float()
        return x, y

def build_transunet(img_size, pretrained_npz) -> nn.Module:
    cfg              = CFG_ViT['R50-ViT-B_16']
    cfg.pretrained_path = pretrained_npz
    cfg.n_classes    = 1
    cfg.n_skip       = 3
    grid             = img_size // cfg.patch_size
    cfg.patches.grid = (grid, grid)
    net = ViT_seg(cfg, img_size=img_size, num_classes=1)
    print('[INFO] Loading ImageNet‑21k weights …')
    net.load_from(np.load(cfg.pretrained_path))
    return net

# ------------------------------------------------------------------
# 4) 각 angle loop
# ------------------------------------------------------------------
for angle in args.angles:
    print(f'\n{"="*60}\n[ANGLE {angle}]')

    # 4‑1) Dataset & 기하학
    dataset   = get_standard_dataset('lodopab', impl='astra_cuda',
                                     num_angles=angle)
    ray_trafo = dataset.get_ray_trafo(impl='astra_cuda')

    # 4‑2) FBP 캐시: train / validation
    cache_dir = Path(args.cache_root) / f'{angle}angle'
    fbp_images = {
        'train':      load_or_create_fbp(
                        dataset, ray_trafo, 'train',
                        cache_dir/'cache_lodopab_train_fbp.npy'),
        'validation': load_or_create_fbp(
                        dataset, ray_trafo, 'validation',
                        cache_dir/'cache_lodopab_validation_fbp.npy')
    }

    # 4‑3) GT 이미지 (메모리에 필요한 만큼만)
    n_tr = fbp_images['train'].shape[0]
    n_va = fbp_images['validation'].shape[0]
    gt_train = np.stack([gt for _, gt in dataset.get_data_pairs('train', n_tr)],
                        axis=0)
    gt_val   = np.stack([gt for _, gt in dataset.get_data_pairs('validation',
                                                                n_va)], axis=0)

    # 4‑4) DataLoader
    ds_tr = FBPCrop(fbp_images['train'],      gt_train, size=args.img_size)
    ds_va = FBPCrop(fbp_images['validation'], gt_val,   size=args.img_size)
    tr_loader = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(ds_va, batch_size=args.batch, shuffle=False)

    # 4‑5) TransUNet
    net = build_transunet(args.img_size, args.pretrained_npz).to(device)

    # 4‑6) Optimizer / Scheduler
    optimizer = AdamW(net.parameters(), lr=args.base_lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(ds_tr) / args.batch)
    scheduler = OneCycleLR(optimizer,
                           max_lr=args.base_lr,
                           total_steps=args.epochs * steps_per_epoch,
                           pct_start=0.1, div_factor=10., final_div_factor=100.)
    criterion = nn.MSELoss()

    # 4‑7) 학습
    log_dir = Path(args.log_root) / f'{angle}angle'
    ensure_dir(log_dir)
    best_psnr = -1.0

    for ep in range(args.epochs):
        # ── Training ───────────────────────────────────────────────
        net.train(); running = 0.0
        for b, (x, y) in enumerate(tr_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            running += loss.item()*x.size(0)
        tr_loss = running / len(ds_tr)

        # ── Validation ────────────────────────────────────────────
        net.eval(); psnr = 0.0
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                out = net(x)
                for i in range(out.size(0)):
                    psnr += PSNR(out[i,0].cpu().numpy(), y[i,0].cpu().numpy())
        psnr /= len(ds_va)
        print(f'[{angle:4d}°] Epoch {ep+1:02d}: '
              f'train_loss={tr_loss:.4f}  val_PSNR={psnr:.2f} dB')

        # ── Checkpoint ────────────────────────────────────────────
        torch.save(net.state_dict(),
                   log_dir / f'epoch_{ep+1:03d}.pth')
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(net.state_dict(), log_dir/'best_model.pth')
            print(f'    [+] New best model saved ({best_psnr:.2f} dB)')

print('\n[Done] All angle experiments finished.')
