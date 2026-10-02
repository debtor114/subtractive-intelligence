# -*- coding: utf-8 -*-
"""CIFAR 용 ResNet-18 (He 2016 의 CIFAR 변형: 3x3 스템, maxpool 없음, 단계 폭 64-128-256-512).

width 배율로 모든 단계 폭을 줄이면 같은 구조의 작은 dense 망 (가산 대조군). 가중치 수는 width 의 제곱에 비례.
프루닝 대상은 conv/linear 의 weight (BatchNorm, bias 제외). 기본 폭에서 프루닝 가능 가중치 약 11.2M.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    def __init__(self, input_shape: Sequence[int] = (3, 32, 32), num_classes: int = 10,
                 widths: Sequence[int] = (64, 128, 256, 512), blocks: Sequence[int] = (2, 2, 2, 2)):
        super().__init__()
        w = [max(1, int(c)) for c in widths]
        self.conv1 = nn.Conv2d(input_shape[0], w[0], 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w[0])
        layers = []
        cin = w[0]
        for i, (cout, n) in enumerate(zip(w, blocks)):
            for j in range(n):
                layers.append(BasicBlock(cin, cout, stride=2 if (i > 0 and j == 0) else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(w[-1], num_classes)
        self.widths = tuple(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def resnet18(width: float = 1.0, num_classes: int = 10) -> ResNetCIFAR:
    base = (64, 128, 256, 512)
    return ResNetCIFAR(widths=tuple(max(1, int(round(c * width))) for c in base), num_classes=num_classes)


def prunable_weights(model: nn.Module) -> int:
    return sum(m.weight.numel() for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))


def width_for_budget(budget: int) -> float:
    """가중치 수가 budget 에 가장 가까운 width 배율 (이분 탐색, 모델을 CPU 에 만들어 센다)."""
    lo, hi = 0.01, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if prunable_weights(resnet18(mid)) < budget:
            lo = mid
        else:
            hi = mid
    return hi
