# -*- coding: utf-8 -*-
"""CIFAR-10 소형 CNN (VGG 풍, BatchNorm). 핵심 실험 CIFAR 확장용.

구조: [conv-bn-relu] x2 -> pool -> [conv-bn-relu] x2 -> pool -> [conv-bn-relu] x2 -> pool -> fc -> fc(10)
channels=(64, 128, 256), fc=256 이면 가중치 약 2.2M (conv 1.15M + fc 1.05M). 과잉 초기화 출발점.
width 를 줄이면 같은 구조의 작은 dense 망 (가산 대조군). 가중치 수는 width 의 제곱에 비례.
프루닝 대상은 conv/linear 의 weight (BatchNorm 과 bias 는 제외).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    def __init__(self, input_shape: Sequence[int] = (3, 32, 32), num_classes: int = 10,
                 channels: Sequence[int] = (64, 128, 256), fc: int = 256, dropout: float = 0.0):
        super().__init__()
        c1, c2, c3 = [max(1, int(c)) for c in channels]
        fc = max(1, int(fc))
        cin = input_shape[0]
        self.features = nn.Sequential(
            _block(cin, c1), _block(c1, c1), nn.MaxPool2d(2),
            _block(c1, c2), _block(c2, c2), nn.MaxPool2d(2),
            _block(c2, c3), _block(c3, c3), nn.MaxPool2d(2),
        )
        side = input_shape[1] // 8
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(c3 * side * side, fc), nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(), nn.Linear(fc, num_classes),
        )
        self.channels = (c1, c2, c3)
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def prunable_weight_count(channels: Sequence[int], fc: int, cin: int = 3, side: int = 4, num_classes: int = 10) -> int:
    c1, c2, c3 = channels
    conv = 9 * (cin * c1 + c1 * c1 + c1 * c2 + c2 * c2 + c2 * c3 + c3 * c3)
    lin = c3 * side * side * fc + fc * num_classes
    return conv + lin


def scaled_config(width: float, base_channels=(64, 128, 256), base_fc=256):
    ch = tuple(max(1, int(round(c * width))) for c in base_channels)
    return ch, max(1, int(round(base_fc * width)))


class ShallowCNN(nn.Module):
    """SmallCNN 의 깊이 변형: conv 4 개 (2 단계) + 풀링 3 회 -> 4x4 -> fc -> fc(10). 같은 예산에서 층 수를 줄이고 폭을 늘린 가산 대조군."""
    def __init__(self, input_shape: Sequence[int] = (3, 32, 32), num_classes: int = 10,
                 channels: Sequence[int] = (128, 256), fc: int = 256):
        super().__init__()
        c1, c2 = [max(1, int(c)) for c in channels]
        fc = max(1, int(fc))
        cin = input_shape[0]
        self.features = nn.Sequential(
            _block(cin, c1), _block(c1, c1), nn.MaxPool2d(2),
            _block(c1, c2), _block(c2, c2), nn.MaxPool2d(2), nn.MaxPool2d(2),
        )
        side = input_shape[1] // 8
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c2 * side * side, fc), nn.ReLU(inplace=True),
                                        nn.Linear(fc, num_classes))
        self.channels = (c1, c2)
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def prunable_weight_count_shallow(channels: Sequence[int], fc: int, cin: int = 3, side: int = 4, num_classes: int = 10) -> int:
    c1, c2 = channels
    return 9 * (cin * c1 + c1 * c1 + c1 * c2 + c2 * c2) + c2 * side * side * fc + fc * num_classes


def scaled_config_shallow(width: float, base_channels=(128, 256), base_fc=256):
    return tuple(max(1, int(round(c * width))) for c in base_channels), max(1, int(round(base_fc * width)))


def width_for_budget_shallow(budget: int, base_channels=(128, 256), base_fc=256) -> float:
    lo, hi = 0.01, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        ch, fc = scaled_config_shallow(mid, base_channels, base_fc)
        if prunable_weight_count_shallow(ch, fc) < budget:
            lo = mid
        else:
            hi = mid
    return hi


def width_for_budget(budget: int, base_channels=(64, 128, 256), base_fc=256) -> float:
    """가중치 수가 budget 에 가장 가까운 width 배율 (이분 탐색)."""
    lo, hi = 0.01, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        ch, fc = scaled_config(mid, base_channels, base_fc)
        if prunable_weight_count(ch, fc) < budget:
            lo = mid
        else:
            hi = mid
    return hi
