# -*- coding: utf-8 -*-
"""가산적(additive) 기준선 모델.

핵심 실험에서 '작게 시작해 파라미터를 채워가는' 쪽의 대조군이다.
- MLP : MNIST / Split MNIST / Permuted MNIST
- ViT : CIFAR-10 / MNIST. 어텐션을 직접 구현해 실험 1(사전 라우팅)에서 교체 가능하게 했다.
"""
from __future__ import annotations

from typing import Sequence

import torch.nn as nn

from .mlp import MLP
from .vit import ViT, Attention, Block

_REGISTRY = {"mlp": MLP, "vit": ViT}


def build_model(cfg: dict, input_shape: Sequence[int], num_classes: int) -> nn.Module:
    """cfg 예: {"name": "vit", "dim": 192, "depth": 6, ...}. name 을 뺀 나머지는 생성자 인자."""
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}', choose from {sorted(_REGISTRY)}")
    return _REGISTRY[name](input_shape=tuple(input_shape), num_classes=num_classes, **cfg)


__all__ = ["MLP", "ViT", "Attention", "Block", "build_model"]
