#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared reproducibility controls for the three training scripts.

Before this module each script seeded the RNGs its own way and made its own
undocumented choice about cuDNN: the U-Net script forced
`deterministic=True, benchmark=False`, the W-TransUNet script set
`benchmark=True`, and the TransUNet script set neither and inherited the PyTorch
defaults. The seed was hard-coded to 0 everywhere with no command-line control.

What changed and what did not
-----------------------------
The *mechanism* is now uniform and explicit: every script takes `--seed` and
`--cudnn`, and both go through this module. The *defaults* are deliberately left
at each script's published value, because the cuDNN regime affects which kernels
are selected and therefore the numerics of a training run:

    train_fbpunet.py         --cudnn deterministic   (published U-Net regime)
    train_transunet.py       --cudnn default         (published TransUNet regime)
    train_wavres_transunet.py --cudnn benchmark      (published W-TransUNet regime)

Running all three under one regime is the right thing to do for a controlled
comparison, but it requires retraining, so it is offered as a flag rather than
imposed as a default. The asymmetry itself is a documented limitation.

`seed_everything` is behaviour-preserving for all three scripts:
`torch.manual_seed` already seeds every CUDA device, so the explicit
`torch.cuda.manual_seed_all` merely makes that visible.
"""

from __future__ import annotations

import random

import numpy as np
import torch

CUDNN_CHOICES = ('default', 'benchmark', 'deterministic')


def seed_everything(seed: int = 0) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_cudnn_regime(mode: str = 'default', verbose: bool = True) -> None:
    """Select the cuDNN autotuning / determinism regime.

    'default'       leave torch.backends.cudnn untouched (benchmark=False,
                    deterministic=False unless something else changed them)
    'benchmark'     autotune convolution algorithms (fastest, non-deterministic)
    'deterministic' deterministic algorithms, no autotuning (slowest, repeatable)
    """
    if mode not in CUDNN_CHOICES:
        raise ValueError(f'--cudnn must be one of {CUDNN_CHOICES}, got {mode!r}')
    if mode == 'benchmark':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    elif mode == 'deterministic':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if verbose:
        print(f'[INFO] cuDNN regime: {mode} '
              f'(benchmark={torch.backends.cudnn.benchmark}, '
              f'deterministic={torch.backends.cudnn.deterministic})')


def add_repro_args(parser, default_cudnn: str = 'default') -> None:
    """Attach --seed / --cudnn to an argparse parser."""
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed for Python/NumPy/PyTorch '
                             '(0 = the value used for the published runs)')
    parser.add_argument('--cudnn', type=str, default=default_cudnn,
                        choices=list(CUDNN_CHOICES),
                        help='cuDNN regime; the default reproduces the '
                             'published run of this script')
