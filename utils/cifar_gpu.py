# -*- coding: utf-8 -*-
"""CIFAR-10 을 GPU 에 올려 두고 증강(패딩 4 랜덤 크롭 + 좌우 반전)도 GPU 에서 하는 경로.

DataLoader 워커가 이 PC 에서 커밋 메모리를 많이 먹어 (워커당 1.3GB) 쓰지 않는다.
학습 데이터는 uint8 (B, 3, 32, 32) 로 보관 (150MB), 배치마다 float 변환 + 정규화.
"""
from __future__ import annotations

import torch
from torchvision import datasets

from .data_loader import CIFAR10_MEAN, CIFAR10_STD, DEFAULT_DATA_ROOT


def load_cifar10_gpu(root: str = DEFAULT_DATA_ROOT, device="cuda"):
    tr = datasets.CIFAR10(root, train=True, download=True)
    te = datasets.CIFAR10(root, train=False, download=True)
    x_tr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous().to(device)      # uint8 (N, 3, 32, 32)
    y_tr = torch.as_tensor(tr.targets, device=device)
    x_te = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous().to(device)
    y_te = torch.as_tensor(te.targets, device=device)
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    return (x_tr, y_tr), (x_te, y_te), (mean, std)


def normalize(x_u8: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x_u8.float().div_(255.0) - mean) / std


def augment(x_u8: torch.Tensor, pad: int = 4, generator=None) -> torch.Tensor:
    """표본별 랜덤 크롭 (제로 패딩 pad) + 50% 좌우 반전. 입력/출력 uint8 (B, 3, H, W)."""
    B, C, H, W = x_u8.shape
    xp = torch.nn.functional.pad(x_u8, (pad, pad, pad, pad))
    oy = torch.randint(0, 2 * pad + 1, (B,), device=x_u8.device, generator=generator)
    ox = torch.randint(0, 2 * pad + 1, (B,), device=x_u8.device, generator=generator)
    rows = (oy[:, None] + torch.arange(H, device=x_u8.device)[None, :])          # (B, H)
    cols = (ox[:, None] + torch.arange(W, device=x_u8.device)[None, :])          # (B, W)
    b_ix = torch.arange(B, device=x_u8.device)[:, None, None]
    out = xp[b_ix, :, rows[:, :, None], cols[:, None, :]]                       # (B, H, W, C)
    out = out.permute(0, 3, 1, 2)
    flip = torch.rand(B, device=x_u8.device, generator=generator) < 0.5
    out = torch.where(flip[:, None, None, None], out.flip(3), out)
    return out.contiguous()
