
# W-TransUNet for Sparse-View CT Reconstruction (LoDoPaB-CT)

Official PyTorch implementation of **W-TransUNet**, a cascaded sparse-view CT reconstruction framework that combines:

1. Haar wavelet-domain residual learning (WavResNet)
2. Transformer–U-Net refinement (TransUNet)

The model is evaluated on the LoDoPaB-CT benchmark dataset under multiple sparse-view conditions
(1000, 500, 250, 125, and 50 views).

---

## 📌 Associated Paper

W-TransUNet: End-to-end wavelet-domain residual learning combined with Transformer–U-Net architecture for sparse-view CT reconstruction

If you use this code, please cite the associated publication.

---

# 1. Repository Structure

w-transunet-svct/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── src/
│   ├── vit_seg_configs.py
│   ├── vit_seg_modeling.py
│   │
│   ├── train_fbpunet.py
│   ├── train_transunet.py
│   ├── train_wavres_transunet.py
│   │
│   ├── inference.py
│   └── compute_metrics_models.py
│
├── cache/            # FBP cache (not tracked by git)
├── logs/             # Training logs and checkpoints
├── outputs/          # Inference outputs and metrics
└── vit_checkpoint/   # Pretrained ViT weights

---

# 2. Environment Setup

## Recommended Environment

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA-enabled GPU
- ODL + DIVal
- ASTRA toolbox (CUDA backend recommended)

## Installation Example

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

If using conda:

conda create -n wtransunet python=3.9
conda activate wtransunet
pip install -r requirements.txt

---

# 3. Dataset: LoDoPaB-CT

This project uses the LoDoPaB-CT dataset via DIVal.

Set dataset path:

export LODOPAB_PATH="/path/to/lodopab"

Or modify the dataset path inside the training scripts.

---

# 4. Pretrained Weights (Required for TransUNet)

Download:

R50+ViT-B_16.npz

Place it in:

vit_checkpoint/imagenet21k/R50+ViT-B_16.npz

This file is NOT distributed with this repository.

---

# 5. FBP Cache Layout

The scripts expect FBP cache arrays:

cache/
  {angle}angle/
    cache_lodopab_train_fbp.npy
    cache_lodopab_validation_fbp.npy
    cache_lodopab_test_fbp.npy

Train/validation cache is created automatically if missing.
Test cache must exist before inference.

---

# 6. Training

## 6.1 U-Net Baseline

python src/train_fbpunet.py   --img_size 352   --epochs 250   --batch 64   --angles 1000 500 250 125 50

---

## 6.2 TransUNet Baseline

python src/train_transunet.py   --pretrained_npz vit_checkpoint/imagenet21k/R50+ViT-B_16.npz   --img_size 352   --epochs 250   --batch 64   --angles 1000 500 250 125 50

---

## 6.3 Proposed W-TransUNet

python src/train_wavres_transunet.py   --pretrained_npz vit_checkpoint/imagenet21k/R50+ViT-B_16.npz   --img_size 352   --epochs 250   --batch 32   --angles 1000 500 250 125 50   --wav_base_ch 64   --wav_blocks 8   --wav_norm gn   --wav_loss_weight 0.05   --residual_out   --amp

---

# 7. Inference & Comparison

python src/inference.py   --angle 125   --cache_root ./cache   --ckpt_unet ./logs/unet/125angle/best_model.pth   --ckpt_transunet ./logs/transunet/125angle/best_model.pth   --ckpt_wavres ./logs/wtransunet/125angle/best_model.pth   --img_size 352   --batch 16   --residual_out   --save_all_png   --out_dir ./outputs/infer/125angle

Outputs include:
- Raw PNG and NPY reconstructions
- 5-panel visualization (GT / FBP / U-Net / TransUNet / W-TransUNet)
- Difference maps
- PSNR/RMSE metrics

---

# 8. Model Complexity & Training Cost

python src/compute_metrics_models.py   --img_size 352   --epochs 250   --angles 1000 500 250 125 50   --cache_root ./cache   --ray_impl astra_cuda   --device cuda   --profile_memory   --profile_time   --out_csv outputs/metrics.csv   --out_json outputs/metrics.json   --out_tex outputs/metrics_table.tex

---

# 9. Reproducibility Notes

- Random seeds are fixed inside training scripts.
- All models are trained using identical dataset splits.
- FBP preprocessing is consistent across models.
- FLOPs are measured for a single 352×352 slice.

---

# 10. Known Limitations

- Requires ASTRA CUDA backend.
- Pretrained ViT weights must be downloaded manually.
- LoDoPaB dataset must be installed separately.
- NPS/MTF evaluation scripts are not included in this repository.

---

# 11. License

Specify license (e.g., MIT or BSD-3-Clause).

---

# 12. Contact

Sunghoon Choi  
Electronics and Telecommunications Research Institute (ETRI)  
Email: schoi@etri.re.kr
