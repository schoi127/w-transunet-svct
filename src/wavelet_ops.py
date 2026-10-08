#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single source of truth for the wavelet front end of W-TransUNet.

Why this module exists
----------------------
`haar_dwt_hvd`, `_upsample_like`, `make_norm`, `ResBlock` and `WavMixResNet`
used to be copy-pasted into `train_wavres_transunet.py`, `inference.py` and
`compute_metrics_models.py` (and into eleven files of the original development
tree). Three independent copies of the numerical core mean an edit can silently
apply to training but not to evaluation, so an ablation switch can report the
wrong answer without raising an error. They are collected here instead.

Provenance
----------
The bodies below are copied verbatim from `train_wavres_transunet.py`, the
script that produced the published W-TransUNet checkpoints (identical in turn to
`train_wavres_transunet_lodopab.py` of the tree used for the paper, lines
358-388, 394-449 and 508-515). Nothing was re-derived, renamed or re-tuned: the
three previous copies were numerically identical and differed only in comments,
docstrings, quote style and type annotations. Behavioural equivalence before and
after the consolidation is checked numerically by the L4 regression test
(identical parameter counts, identical forward FLOPs and bit-identical outputs
for a fixed seed).

Note on the transform
---------------------
This is a one-level Haar analysis that returns LL/H/V/D. The model deliberately
discards LL and feeds only the three detail bands back at full resolution
alongside the FBP image; there is no synthesis (IDWT) stage anywhere in the
pipeline. That is the behaviour of the published model and is left untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "_upsample_like",
    "haar_dwt_hvd",
    "make_norm",
    "ResBlock",
    "WavMixResNet",
    "wavelet_detail_loss",
]


# ------------------------------------------------------------------
# Wavelet: Haar DWT -> H/V/D details
# ------------------------------------------------------------------
def _upsample_like(x: torch.Tensor, ref: torch.Tensor, mode: str) -> torch.Tensor:
    # x: (B,1,h,w), ref: (B,1,H,W)
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    if mode == 'nearest':
        return F.interpolate(x, size=ref.shape[-2:], mode='nearest')
    # bilinear
    return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)


def haar_dwt_hvd(x: torch.Tensor):
    """
    Haar 2D DWT (1-level) returning LL, H, V, D.
    x: (B, C, H, W) where H,W even
    return: (ll, h, v, d) each (B, C, H/2, W/2)
    """
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


# ------------------------------------------------------------------
# WavResNet: (FBP + H + V + D) mixture CNN
# ------------------------------------------------------------------
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
    """
    input : 4 channels (FBP, H, V, D) at full resolution
    output: 1 channel mixture (residual form for stability)
    """
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
        return y + self.skip_proj(x4)  # residual mixture


# ------------------------------------------------------------------
# Wavelet-detail loss term (the alpha * L_detail of the training objective)
# ------------------------------------------------------------------
def wavelet_detail_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B,1,H,W)
    Sum of MSEs over the Haar detail bands (H, V, D).
    """
    _, ph, pv, pd = haar_dwt_hvd(pred)
    _, th, tv, td = haar_dwt_hvd(target)
    return F.mse_loss(ph, th) + F.mse_loss(pv, tv) + F.mse_loss(pd, td)
