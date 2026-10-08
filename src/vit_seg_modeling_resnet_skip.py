# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# ResNetV2 (BiT) backbone with skip connections, used by the TransUNet encoder.
#
# ORIGIN / ATTRIBUTION
#   Original author : Jieneng Chen et al. ("Beckschen"), TransUNet
#                     https://github.com/Beckschen/TransUNet
#                     networks/vit_seg_modeling_resnet_skip.py
#   Original license: Apache License 2.0 (see the upstream repository for the
#                     full text). Copyright of the original code remains with
#                     its authors.
#   Redistributed here unmodified, as permitted by the Apache-2.0 license, so
#   that this repository imports without an external TransUNet checkout.
#
# WHY THIS FILE IS HERE
#   `vit_seg_modeling.py` imports `ResNetV2` from this module. The file was
#   present in the tree that produced the published results but was omitted
#   when the repository was prepared for release, which made
#   `import vit_seg_modeling` fail. It is restored byte-for-byte from that
#   tree (SHA-256 5badeb597c7cf40d0e444d6586c19b9484d2c98e15e107b969a7a6765a222c14);
#   the only differences against the upstream TransUNet file are an unused
#   debug helper and comments, so the network is the upstream network.
#
#   No line below this header has been altered: the model definition, and hence
#   every published number, is unchanged.
# ---------------------------------------------------------------------------

import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def _fmt_bwhc(self, x: torch.Tensor) -> str:
        b, c, h, w = x.shape
        return f"(B={b}, C={c}, W={w}, H={h})"

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        # print("[input]      ", self._fmt_bwhc(x))
        x = self.root(x) #(16, 64, 176, 176)
        # print("[after root] ", self._fmt_bwhc(x))
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x) #(16, 64, 87, 87)
        # print("[after pool] ", self._fmt_bwhc(x))

        for i in range(len(self.body)-1): # block1 ,block2만 순회
            # print(i)
            x = self.body[i](x) #i=0 : (16, 256, 87, 87) -> i=1 : (16, 512, 44, 44) (stride=2 downsample)
            # print(x.shape, x[:, :, :, -1])
            right_size = int(in_size / 4 / (i+1)) #i=0 : 88 -> i=1 : 44
            if x.size()[2] != right_size: #i=0 : 87 != 88 -> i=1 : 44 = 44
                pad = right_size - x.size()[2] # 88-87 = 1
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device) #i=0 : (16, 256, 88, 88) 0으로 채운 텐서 크기
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:] # 맨아래 행과 오른쪽 마지막 열은 0, 나머지는 기존 값 그대로 복사
                # print(f"  ↳ resized skip{i + 1} to", self._fmt_bwhc(feat))
            else: #i=1 : 44 = 44
                feat = x
                # print(f"  ↳ skip{i + 1}         ", self._fmt_bwhc(feat))
            # print(feat.shape, feat[:, :, :, -1])

            features.append(feat)

        x = self.body[-1](x) # block3 (16, 1024, 22, 22)
        # print(f"[after block{len(self.body)}]", self._fmt_bwhc(x))
        return x, features[::-1]

