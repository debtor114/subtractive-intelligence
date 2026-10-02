# -*- coding: utf-8 -*-
"""MNIST 를 GPU 텐서로 통째로 올려 쓰는 빠른 경로.

DataLoader 는 MLP 급 모델에서 CPU 변환 비용이 병목이다 (60k 이미지 1 에폭에 수 초).
연속학습(실험 3), 핵심 실험(MLP), STDP(실험 4) 는 이 모듈로 배치를 GPU 인덱싱으로 뽑는다.

- load_mnist_tensors : ((x_tr, y_tr), (x_te, y_te)). x 는 (N, 1, 28, 28) float. normalize=False 면 0~1 원시 강도.
- TensorBatches      : 인덱스 텐서 위를 셔플하며 미니배치 인덱스를 낸다.
- make_split_tasks   : 클래스별 태스크 인덱스 (Split MNIST)
- make_permutations  : 태스크별 픽셀 순열 (Permuted MNIST). 태스크 0 은 항등.
- apply_perm         : 배치에 순열 적용
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torchvision import datasets

from .data_loader import DEFAULT_DATA_ROOT, MNIST_MEAN, MNIST_STD


def load_mnist_tensors(root: str = DEFAULT_DATA_ROOT, device="cpu", normalize: bool = True):
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)

    def prep(ds):
        x = ds.data.float().div_(255.0).unsqueeze(1)
        if normalize:
            x = (x - MNIST_MEAN[0]) / MNIST_STD[0]
        return x.to(device), ds.targets.to(device)

    return prep(tr), prep(te)


class TensorBatches:
    """인덱스 텐서 위의 미니배치 반복자. 매 에폭 셔플. drop_last=False."""

    def __init__(self, idx: torch.Tensor, batch_size: int, shuffle: bool = True,
                 generator: Optional[torch.Generator] = None):
        self.idx = idx
        self.bs = batch_size
        self.shuffle = shuffle
        self.gen = generator

    def __len__(self) -> int:
        return (self.idx.numel() + self.bs - 1) // self.bs

    def __iter__(self):
        n = self.idx.numel()
        if self.shuffle:
            order = torch.randperm(n, device=self.idx.device, generator=self.gen)
            idx = self.idx[order]
        else:
            idx = self.idx
        for s in range(0, n, self.bs):
            yield idx[s:s + self.bs]


def make_split_tasks(y_train: torch.Tensor, y_test: torch.Tensor, n_tasks: int = 5,
                     class_order: Optional[Sequence[int]] = None) -> List[Dict]:
    n_cls = int(y_train.max().item()) + 1
    assert n_cls % n_tasks == 0
    order = list(class_order) if class_order is not None else list(range(n_cls))
    per = n_cls // n_tasks
    tasks = []
    for t in range(n_tasks):
        cls = order[t * per:(t + 1) * per]
        cls_t = torch.as_tensor(cls, device=y_train.device)
        tr = torch.isin(y_train, cls_t).nonzero().squeeze(1)
        te = torch.isin(y_test, cls_t).nonzero().squeeze(1)
        tasks.append({"task_id": t, "classes": tuple(int(c) for c in cls), "train_idx": tr, "test_idx": te})
    return tasks


def make_permutations(n_tasks: int, seed: int, n_pixels: int = 28 * 28, device="cpu") -> List[torch.Tensor]:
    rng = np.random.RandomState(seed)
    perms = [torch.arange(n_pixels, device=device)]
    for _ in range(1, n_tasks):
        perms.append(torch.from_numpy(rng.permutation(n_pixels)).to(device))
    return perms


def apply_perm(x: torch.Tensor, perm: Optional[torch.Tensor]) -> torch.Tensor:
    if perm is None:
        return x
    shape = x.shape
    return x.reshape(shape[0], -1)[:, perm].reshape(shape)
