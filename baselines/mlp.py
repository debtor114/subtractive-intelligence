# -*- coding: utf-8 -*-
"""MLP 기준선.

width_mult 로 폭을 키워 '과잉 초기화' 출발점을 만들 수 있다 (핵심 실험의 감산 쪽 출발점).
프루닝 실험에서 마스크를 씌울 대상은 self.net 안의 nn.Linear 들이다.
"""
from __future__ import annotations

from math import prod
from typing import Sequence

import torch
import torch.nn as nn

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class MLP(nn.Module):
    def __init__(self, input_shape: Sequence[int] = (1, 28, 28), num_classes: int = 10,
                 hidden: Sequence[int] = (256, 256), dropout: float = 0.0,
                 width_mult: float = 1.0, activation: str = "relu"):
        super().__init__()
        in_dim = int(prod(input_shape))
        dims = [in_dim] + [max(1, int(round(h * width_mult))) for h in hidden]
        act = _ACT[activation]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)
        self.hidden_dims = dims[1:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.flatten(x, 1))
