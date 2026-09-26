"""OSNet / OSNet-AIN person Re-ID backbones (inference only), vendored from torchreid (deep-person-reid).

Source: https://github.com/KaiyangZhou/deep-person-reid  (torchreid/models/osnet.py, osnet_ain.py), trimmed to the
layers needed to run the published checkpoints: no classifier head, no training code, no download helpers.
References: Zhou et al., "Omni-Scale Feature Learning for Person Re-Identification", ICCV 2019;
"Learning Generalisable Omni-Scale Representations for Person Re-Identification", TPAMI 2021.

    MIT License
    Copyright (c) 2018 Kaiyang Zhou
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
    to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

The code licence above covers this file only. The pretrained WEIGHTS have their own terms (they were trained on
research datasets such as MSMT17); see README "Multi-camera Re-ID & global IDs".
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.InstanceNorm2d(cout, affine=True) if IN else nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, cin, cout, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, cin, cout, stride=1, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(cout) if bn else None

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x) if self.bn is not None else x


class LightConv3x3(nn.Module):
    """1x1 (linear) + depth-wise 3x3 (non-linear)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False, groups=cout)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class LightConvStream(nn.Module):
    def __init__(self, cin, cout, depth):
        super().__init__()
        self.layers = nn.Sequential(LightConv3x3(cin, cout), *[LightConv3x3(cout, cout) for _ in range(depth - 1)])

    def forward(self, x):
        return self.layers(x)


class ChannelGate(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c // reduction, kernel_size=1, bias=True, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(c // reduction, c, kernel_size=1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        g = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * g


# ---- OSNet (osnet.py): four fixed streams conv2a..conv2d, optional IN after the residual -------------------------
class OSBlockV1(nn.Module):
    def __init__(self, cin, cout, IN=False, bottleneck_reduction=4):
        super().__init__()
        mid = cout // bottleneck_reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(2)])
        self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
        self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None
        self.IN = nn.InstanceNorm2d(cout, affine=True) if IN else None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
        out = self.conv3(x2) + (self.downsample(x) if self.downsample is not None else x)
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


class OSNetV1(nn.Module):
    def __init__(self, channels, IN=False, feature_dim=512):
        super().__init__()
        c = channels
        self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._layer(c[0], c[1], True, IN)
        self.conv3 = self._layer(c[1], c[2], True)
        self.conv4 = self._layer(c[2], c[3], False)
        self.conv5 = Conv1x1(c[3], c[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU())
        self.feature_dim = feature_dim

    @staticmethod
    def _layer(cin, cout, reduce, IN=False):
        layers = [OSBlockV1(cin, cout, IN=IN), OSBlockV1(cout, cout, IN=IN)]
        if reduce:
            layers.append(nn.Sequential(Conv1x1(cout, cout), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.maxpool(self.conv1(x))))))
        return self.fc(self.global_avgpool(x).flatten(1))


# ---- OSNet-AIN (osnet_ain.py): T streams in a ModuleList; some blocks use instance norm inside the residual -------
class OSBlockT(nn.Module):
    def __init__(self, cin, cout, reduction=4, T=4):
        super().__init__()
        mid = cout // reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2 = nn.ModuleList([LightConvStream(mid, mid, t) for t in range(1, T + 1)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = sum(self.gate(s(x1)) for s in self.conv2)
        return F.relu(self.conv3(x2) + (self.downsample(x) if self.downsample is not None else x))


class OSBlockINin(nn.Module):
    def __init__(self, cin, cout, reduction=4, T=4):
        super().__init__()
        mid = cout // reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2 = nn.ModuleList([LightConvStream(mid, mid, t) for t in range(1, T + 1)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout, bn=False)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None
        self.IN = nn.InstanceNorm2d(cout, affine=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = sum(self.gate(s(x1)) for s in self.conv2)
        return F.relu(self.IN(self.conv3(x2)) + (self.downsample(x) if self.downsample is not None else x))


class OSNetAIN(nn.Module):
    BLOCKS = [[OSBlockINin, OSBlockINin], [OSBlockT, OSBlockINin], [OSBlockINin, OSBlockT]]

    def __init__(self, channels, feature_dim=512):
        super().__init__()
        c = channels
        self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3, IN=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Sequential(self.BLOCKS[0][0](c[0], c[1]), self.BLOCKS[0][1](c[1], c[1]))
        self.pool2 = nn.Sequential(Conv1x1(c[1], c[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = nn.Sequential(self.BLOCKS[1][0](c[1], c[2]), self.BLOCKS[1][1](c[2], c[2]))
        self.pool3 = nn.Sequential(Conv1x1(c[2], c[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = nn.Sequential(self.BLOCKS[2][0](c[2], c[3]), self.BLOCKS[2][1](c[3], c[3]))
        self.conv5 = Conv1x1(c[3], c[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(c[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU())
        self.feature_dim = feature_dim

    def forward(self, x):
        x = self.pool2(self.conv2(self.maxpool(self.conv1(x))))
        x = self.conv5(self.conv4(self.pool3(self.conv3(x))))
        return self.fc(self.global_avgpool(x).flatten(1))


CHANNELS = {"x1_0": [64, 256, 384, 512], "x0_75": [48, 192, 288, 384], "x0_5": [32, 128, 192, 256],
            "x0_25": [16, 64, 96, 128]}


def build(arch: str) -> nn.Module:
    """arch: osnet_x1_0 | osnet_x0_75 | osnet_x0_5 | osnet_x0_25 | osnet_ibn_x1_0 | osnet_ain_x1_0 | osnet_ain_x0_25 ..."""
    m = re.fullmatch(r"osnet_(ain_|ibn_)?(x1_0|x0_75|x0_5|x0_25)", arch)
    if not m:
        raise ValueError(f"unknown OSNet architecture {arch!r}")
    ch = CHANNELS[m.group(2)]
    if m.group(1) == "ain_":
        return OSNetAIN(ch)
    return OSNetV1(ch, IN=m.group(1) == "ibn_")


def arch_from_name(path: str) -> str:
    m = re.search(r"osnet_(ain_|ibn_)?(x1_0|x0_75|x0_5|x0_25)", Path(path).name)
    if not m:
        raise ValueError(f"cannot infer the OSNet architecture from {Path(path).name!r}; set reid.arch")
    return m.group(0)


def load(path: str, arch: str | None = None) -> nn.Module:
    """Build the backbone and load a torchreid checkpoint (classifier weights are dropped; everything else strict)."""
    arch = arch or arch_from_name(path)
    model = build(arch)
    d = torch.load(str(path), map_location="cpu", weights_only=True)
    sd = d.get("state_dict", d) if isinstance(d, dict) else d
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier.")}
    model.load_state_dict(sd, strict=True)
    return model.eval()
