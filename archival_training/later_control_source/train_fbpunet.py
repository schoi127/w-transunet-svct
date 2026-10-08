# Archival source; complete checkpoint-generating run provenance is unavailable.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os, sys, math, random, argparse

# The released version hard-coded os.environ["CUDA_VISIBLE_DEVICES"] = "1",
# which silently moved the run to the second GPU of the author's machine and
# made the script unusable on a single-GPU host. Device selection now belongs to
# the caller: export CUDA_VISIBLE_DEVICES before launching, or use --single_gpu.
# This affects device placement only, never the numerics.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

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
from dival.reconstructors.fbpunet_reconstructor import FBPUNetReconstructor
from dival.reference_reconstructors import (
    check_for_params, download_params, get_hyper_params_path)
from dival.measure import PSNR
from odl.tomo import fbp_op

# Shared seeding / cuDNN controls (src/repro.py). The published U-Net run used
# cudnn deterministic=True, benchmark=False, which is '--cudnn deterministic'.
from repro import add_repro_args, apply_cudnn_regime, seed_everything

# ------------------ (1) CLI ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=352, help='입력 크기(16의 배수)')
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--batch',  type=int, default=64)  # ← 기본 64
parser.add_argument('--base_lr', type=float, default=3e-4)
parser.add_argument('--angles', nargs='+', type=int,
                    default=[1000, 500, 250, 125, 50],
                    help='투영 각도 개수 리스트')
# P6: keep best + every save_every epochs (+ the final epoch) instead of a
# checkpoint per epoch.
parser.add_argument('--save_every', type=int, default=10,
                    help='save checkpoint every N epochs')
parser.add_argument('--cache_root', type=str, default='./cache',
                    help='FBP 캐시 저장 루트')
parser.add_argument('--log_root',   type=str, default='./logs/lodopab_unet',
                    help='로그 & 체크포인트 루트')
parser.add_argument('--single_gpu', type=int, default=0,
                    help='단일 GPU id (DataParallel 해제 후 이 GPU만 사용)')
parser.add_argument('--legacy_amp_noclip_unscale', action='store_true',
                    help=('reproduce the published U-Net baseline exactly: clip '
                          'gradients WITHOUT scaler.unscale_() first, i.e. with '
                          'the AMP scaling bug that the published run had. '
                          'Off by default; see the comment in the training loop.'))
# 재현성: seed / cuDNN 레짐 (기본값 = 논문 학습 시 사용된 값)
add_repro_args(parser, default_cudnn='deterministic')
args = parser.parse_args()

# ------------------ (2) Device ------------------
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('mps')  if torch.backends.mps.is_available()
          else torch.device('cpu'))
seed_everything(args.seed)
# default '--cudnn deterministic' == the published U-Net regime
apply_cudnn_regime(args.cudnn)
print(f'[INFO] Using device: {device}')

# ------------------ (3) Helpers ------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_or_create_fbp(dataset, ray_trafo, split: str, cache_path: Path) -> np.ndarray:
    ensure_dir(cache_path.parent)
    if cache_path.is_file():
        print(f'  – load {split:10s} FBP from cache: {cache_path}')
        return np.load(cache_path)
    print(f'  – cache missing → computing {split:10s} FBP')
    sins = dataset.get_data_pairs(split)
    op_fbp = fbp_op(ray_trafo, filter_type='Ram-Lak', frequency_scaling=1.0)
    fbp_split = [np.asarray(op_fbp(sin)) for sin, _ in tqdm(sins)]
    arr = np.stack(fbp_split, axis=0)
    np.save(cache_path, arr)
    return arr

class FBPCrop(Dataset):
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

# ------------------ (4) AMP ------------------
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

# ------------------ (5) Main loop ------------------
for angle in args.angles:
    print(f'\n{"="*60}\n[ANGLE {angle}]')

    dataset   = get_standard_dataset('lodopab', impl='astra_cuda',
                                     num_angles=angle)
    ray_trafo = dataset.get_ray_trafo(impl='astra_cuda')

    cache_dir = Path(args.cache_root) / f'{angle}angle'
    fbp_images = {
        'train':      load_or_create_fbp(dataset, ray_trafo, 'train',
                                         cache_dir/'cache_lodopab_train_fbp.npy'),
        'validation': load_or_create_fbp(dataset, ray_trafo, 'validation',
                                         cache_dir/'cache_lodopab_validation_fbp.npy')
    }

    n_tr = fbp_images['train'].shape[0]
    n_va = fbp_images['validation'].shape[0]
    gt_train = np.stack([gt for _, gt in dataset.get_data_pairs('train', n_tr)], axis=0)
    gt_val   = np.stack([gt for _, gt in dataset.get_data_pairs('validation', n_va)], axis=0)

    ds_tr = FBPCrop(fbp_images['train'],      gt_train, size=args.img_size)
    ds_va = FBPCrop(fbp_images['validation'], gt_val,   size=args.img_size)
    tr_loader = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,  pin_memory=True)
    va_loader = DataLoader(ds_va, batch_size=args.batch, shuffle=False, pin_memory=True)

    LOG_DIR = str(Path(args.log_root) / f'{angle}angle'); ensure_dir(Path(LOG_DIR))
    SAVE_BEST_LEARNED_PARAMS_PATH = './params/lodopab_fbpunet'; ensure_dir(Path(SAVE_BEST_LEARNED_PARAMS_PATH))

    reconstructor = FBPUNetReconstructor(
        ray_trafo, log_dir=LOG_DIR,
        save_best_learned_params_path=SAVE_BEST_LEARNED_PARAMS_PATH)

    if not check_for_params('fbpunet', 'lodopab', include_learned=False):
        download_params('fbpunet', 'lodopab', include_learned=False)
    hyper_params_path = get_hyper_params_path('fbpunet', 'lodopab')
    reconstructor.load_hyper_params(hyper_params_path)

    # ── P13: keep DIVal's hyper parameters in step with the CLI values.
    #    A step-count scheduler built from DIVal's own epochs/batch_size
    #    raises 'Tried to step N times' once the CLI values differ.
    for _k, _v in [('epochs', args.epochs), ('batch_size', args.batch)]:
        try:
            _old = reconstructor.hyper_params.get(_k, None)
            reconstructor.hyper_params[_k] = _v
            print(f'[P13] hyper_params[{_k}]: {_old} -> {_v}')
        except Exception as _e:
            print(f'[P13][WARN] hyper_params[{_k}] sync failed: {_e!r}')
    print('[P13] DIVal hyper_params:', dict(reconstructor.hyper_params))

    reconstructor.init_model()

    # ----------  DataParallel 강제 해제 & 단일 GPU 고정 ----------
    # DIVal 내부에서 DataParallel을 감싸는 경우가 있어 OOM의 원인(작은 GPU에도 replica 생성)
    if isinstance(reconstructor.model, torch.nn.DataParallel):
        print('[INFO] Detected DataParallel → unwrap to single GPU.')
        reconstructor.model = reconstructor.model.module  # DP 해제
    device = torch.device(f'cuda:{args.single_gpu}') if device.type == 'cuda' else device
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    reconstructor.model.to(device)
    # -----------------------------------------------------------------------

    criterion = nn.MSELoss()
    reconstructor.init_optimizer(dataset_train=ds_tr)
    reconstructor.init_scheduler(dataset_train=ds_tr)

    # ── N3: record what the scheduler actually is, and whether the loop steps it.
    #    On FBPUNetReconstructor `reconstructor.scheduler` resolves to the
    #    hyper-parameter string, not the scheduler object (the metaclass builds a
    #    property for every HYPER_PARAMS key). The object lives in `_scheduler`.
    #    The loop's isinstance check below reads the former, so it never fires.
    from torch.optim.lr_scheduler import CyclicLR, OneCycleLR
    _sch_hp = getattr(reconstructor, 'scheduler', None)
    _sch_obj = getattr(reconstructor, '_scheduler', None)
    print('[N3] reconstructor.scheduler  =', repr(_sch_hp),
          '| type:', type(_sch_hp).__name__)
    print('[N3] reconstructor._scheduler =', type(_sch_obj).__name__)
    print('[N3] _scheduler attrs:', {k: getattr(_sch_obj, k, None)
                                     for k in ('total_steps', 'T_max', 'eta_min',
                                               'base_lrs', 'last_epoch')})
    print('[N3] loop steps it? (isinstance on .scheduler):',
          isinstance(_sch_hp, (CyclicLR, OneCycleLR)))
    print('[N3] would step if it used ._scheduler:',
          isinstance(_sch_obj, (CyclicLR, OneCycleLR)))

    best_psnr = -1.0
    print('\n② U-Net 학습/검증 시작 …')
    for epoch in range(args.epochs):
        # ---- Train ----
        reconstructor.model.train()
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(tr_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            reconstructor.optimizer.zero_grad(set_to_none=True)

            # ----------  AMP ----------
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out = reconstructor.model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            # ---------------------------------------------------------------
            # BUG FIX (AMP gradient clipping) — READ BEFORE COMPARING RESULTS
            #
            # The published U-Net baseline was trained WITHOUT the
            # `scaler.unscale_()` call below. Under AMP the gradients are still
            # multiplied by the GradScaler's loss scale at this point (order
            # 10^4), so clipping to max_norm=1.0 acted on scaled gradients: the
            # clip fired on essentially every step and rescaled each update by
            # 1/‖g_scaled‖. The U-Net was therefore trained under an effective
            # normalised-gradient rule that neither TransUNet (no AMP) nor
            # W-TransUNet (correct unscale_) was subject to.
            #
            # The line is added here so that anyone rerunning this script gets
            # the intended behaviour. It does NOT retroactively change the
            # published U-Net numbers, which were produced by the buggy path and
            # are reported as such in the paper's limitations. Any rerun of this
            # script is consequently NOT a bit-level reproduction of the
            # published U-Net baseline; pass --legacy_amp_noclip_unscale to
            # reproduce the original (incorrect) behaviour on purpose.
            # ---------------------------------------------------------------
            if not args.legacy_amp_noclip_unscale:
                scaler.unscale_(reconstructor.optimizer)
            torch.nn.utils.clip_grad_norm_(reconstructor.model.parameters(), 1.0)
            scaler.step(reconstructor.optimizer)
            scaler.update()
            # --------------------------------------

            if isinstance(reconstructor.scheduler,
                          (torch.optim.lr_scheduler.CyclicLR,
                           torch.optim.lr_scheduler.OneCycleLR)):
                reconstructor.scheduler.step()

            running_loss += loss.item() * x.size(0)

            if batch_idx % 10 == 0 or batch_idx == len(tr_loader):
                avg_loss = running_loss / (batch_idx * tr_loader.batch_size)
                print(f"Epoch [{epoch + 1}/{args.epochs}] "
                      f"Batch [{batch_idx}/{len(tr_loader)}] "
                      f"Loss: {loss.item():.4f} AvgLoss: {avg_loss:.4f}")

        train_loss = running_loss / len(ds_tr)

        # ---- Validation ----
        reconstructor.model.eval()
        val_psnr = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            for x, y in va_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = reconstructor.model(x)
                for i in range(out.size(0)):
                    val_psnr += PSNR(out[i, 0].float().cpu().numpy(),
                                     y[i, 0].float().cpu().numpy())
        val_psnr /= len(ds_va)

        print(f'[{angle:4d}°] Epoch {epoch+1:03d}/{args.epochs} | '
              f'train_loss={train_loss:.4f} | val_PSNR={val_psnr:.2f} dB')
        print(f'[N3] epoch {epoch+1} lr = {reconstructor.optimizer.param_groups[0]["lr"]:.8g}')

        # ---- Checkpoint ----
        if ((epoch + 1) % args.save_every == 0) or ((epoch + 1) == args.epochs):
            torch.save(reconstructor.model.state_dict(),
                       Path(LOG_DIR) / f'epoch_{epoch + 1:03d}.pth')
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(reconstructor.model.state_dict(), Path(LOG_DIR) / 'best_model.pth')
            print(f'    [+] New best model saved ({best_psnr:.2f} dB)')

        # ---- 10 epoch마다 시각화 (옵션) ----
        if (epoch + 1) % 10 == 0:
            save_dir = Path(LOG_DIR) / f'epoch_{epoch + 1:04d}'
            ensure_dir(save_dir)
            test_pairs = dataset.get_data_pairs('test', 10)
            my_fbp_op = fbp_op(ray_trafo, filter_type='Ram-Lak', frequency_scaling=1.0)

            import matplotlib.pyplot as plt
            def normalize(img):
                img = np.asarray(img)
                return (img - img.min()) / (img.max() - img.min() + 1e-8)

            for i, (sinogram, gt) in enumerate(test_pairs):
                fbp_np = np.asarray(my_fbp_op(sinogram))
                inp = torch.from_numpy(fbp_np).unsqueeze(0).unsqueeze(0).float().to(device)
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    reco_np = reconstructor.model(inp).float().cpu().numpy()[0, 0]
                gt_np = np.asarray(gt)

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(normalize(gt_np), cmap='gray');  axes[0].set_title('Ground-truth'); axes[0].axis('off')
                axes[1].imshow(normalize(fbp_np), cmap='gray'); axes[1].set_title('FBP input');    axes[1].axis('off')
                axes[2].imshow(normalize(reco_np), cmap='gray');axes[2].set_title('U-Net output'); axes[2].axis('off')
                fig.tight_layout()
                fig.savefig(save_dir / f'sample_{i:02d}.png', dpi=150)
                plt.close(fig)

print('\n[Done] All angle experiments finished.')
