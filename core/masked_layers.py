# -*- coding: utf-8 -*-
"""weight_mask 버퍼를 가진 Linear / Conv2d.

- forward 는 weight * weight_mask 를 쓴다. 마스크가 0 인 원소는 출력에 기여하지 않고 gradient 도 0 이다.
- 옵티마이저(모멘텀, weight decay)가 마스크된 원소를 건드릴 수 있으므로, 매 스텝 뒤 apply_masks() 로
  weight.data 를 다시 0 으로 맞춘다 (utils.metrics 는 weight * mask 로 밀도를 재므로 결과엔 영향 없음).
- convert_to_masked(model) 은 기존 nn.Linear / nn.Conv2d 를 가중치를 복사해 마스크 버전으로 바꾼다.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    """dense_grad=True 면 마스크된 자리까지 포함한 '유효 가중치' 의 gradient 를 보관한다 (RigL 재성장용).
    weight 자체의 grad 는 마스크 곱 때문에 끊긴 자리에서 항상 0 이므로, w_eff = weight * mask 에 retain_grad 를 건다."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kw):
        super().__init__(in_features, out_features, bias=bias, **kw)
        self.register_buffer("weight_mask", torch.ones_like(self.weight))
        self.dense_grad = False
        self._w_eff = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.weight_mask
        if self.dense_grad and w.requires_grad:
            w.retain_grad()
            self._w_eff = w
        return F.linear(x, w, self.bias)

    def dense_weight_grad(self):
        return None if self._w_eff is None else self._w_eff.grad


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.register_buffer("weight_mask", torch.ones_like(self.weight))
        self.dense_grad = False
        self._w_eff = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.weight_mask
        if self.dense_grad and w.requires_grad:
            w.retain_grad()
            self._w_eff = w
        return self._conv_forward(x, w, self.bias)

    def dense_weight_grad(self):
        return None if self._w_eff is None else self._w_eff.grad


MASKED_TYPES = (MaskedLinear, MaskedConv2d)


def _replace(module: nn.Module, name: str, new: nn.Module) -> None:
    parent = module
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def convert_to_masked(model: nn.Module, skip: Iterable[str] = ()) -> nn.Module:
    """nn.Linear / nn.Conv2d 를 마스크 버전으로 교체 (제자리). skip 에 든 이름(부분 문자열 일치)은 건너뜀."""
    skip = tuple(skip)
    targets: List[Tuple[str, nn.Module]] = [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, (nn.Linear, nn.Conv2d)) and not isinstance(m, MASKED_TYPES)
        and not any(s in n for s in skip)
    ]
    for name, m in targets:
        if isinstance(m, nn.Linear):
            new = MaskedLinear(m.in_features, m.out_features, bias=m.bias is not None)
        else:
            new = MaskedConv2d(m.in_channels, m.out_channels, m.kernel_size, stride=m.stride,
                               padding=m.padding, dilation=m.dilation, groups=m.groups, bias=m.bias is not None)
        new = new.to(m.weight.device, m.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(m.weight)
            if m.bias is not None:
                new.bias.copy_(m.bias)
        _replace(model, name, new)
    return model


def masked_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, MASKED_TYPES)]


@torch.no_grad()
def apply_masks(model: nn.Module) -> None:
    """마스크된 원소의 weight 를 0 으로 고정."""
    for _, m in masked_modules(model):
        m.weight.mul_(m.weight_mask)


@torch.no_grad()
def mask_density(model: nn.Module) -> float:
    tot, on = 0, 0
    for _, m in masked_modules(model):
        tot += m.weight_mask.numel()
        on += int(m.weight_mask.sum().item())
    return on / max(tot, 1)


@torch.no_grad()
def active_connections(model: nn.Module) -> int:
    return int(sum(m.weight_mask.sum().item() for _, m in masked_modules(model)))
