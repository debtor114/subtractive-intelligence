# -*- coding: utf-8 -*-
"""데이터셋 로더.

기본 분류용 MNIST / CIFAR-10 과, 연속학습(실험 3)용 Split MNIST / Permuted MNIST 를 제공한다.

- get_mnist / get_cifar10 : (train_loader, test_loader, DataInfo)
- get_split_mnist         : ([Task, ...], DataInfo)   태스크당 2개 클래스 (기본 5 태스크)
- get_permuted_mnist      : ([Task, ...], DataInfo)   태스크마다 고정 픽셀 순열, 첫 태스크는 원본

주의(Windows): num_workers > 0 이면 spawn 방식이라 transform 이 picklable 해야 한다.
그래서 픽셀 순열은 lambda 대신 Permute 클래스로 구현했다. 호출 스크립트는
`if __name__ == "__main__":` 가드가 필요하다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.path.join(REPO_ROOT, "data")

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class DataInfo:
    name: str
    input_shape: Tuple[int, int, int]  # (C, H, W)
    num_classes: int


@dataclass
class Task:
    task_id: int
    classes: Tuple[int, ...]
    train_loader: DataLoader
    test_loader: DataLoader


def _loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int,
            drop_last: bool = False) -> DataLoader:
    kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
              pin_memory=torch.cuda.is_available(), drop_last=drop_last)
    if num_workers > 0:
        kw["persistent_workers"] = True
    return DataLoader(ds, **kw)


def _mnist_tf(extra: Optional[list] = None) -> transforms.Compose:
    tf = [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    if extra:
        tf += extra
    return transforms.Compose(tf)


# ---------------------------------------------------------------------------
# 기본 분류
# ---------------------------------------------------------------------------

def get_mnist(root: str = DEFAULT_DATA_ROOT, batch_size: int = 128, num_workers: int = 0):
    tf = _mnist_tf()
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    info = DataInfo("mnist", (1, 28, 28), 10)
    return (_loader(train, batch_size, True, num_workers),
            _loader(test, batch_size * 2, False, num_workers), info)


def get_cifar10(root: str = DEFAULT_DATA_ROOT, batch_size: int = 128, num_workers: int = 0,
                augment: bool = True):
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_tf = [transforms.ToTensor(), norm]
    if augment:
        train_tf = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] + train_tf
    train = datasets.CIFAR10(root, train=True, download=True, transform=transforms.Compose(train_tf))
    test = datasets.CIFAR10(root, train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor(), norm]))
    info = DataInfo("cifar10", (3, 32, 32), 10)
    return (_loader(train, batch_size, True, num_workers),
            _loader(test, batch_size * 2, False, num_workers), info)


# ---------------------------------------------------------------------------
# 연속학습 (실험 3)
# ---------------------------------------------------------------------------

class RemapLabels(Dataset):
    """라벨을 태스크 내부 인덱스(0..k-1)로 바꾼다. Task-IL 평가용."""

    def __init__(self, base: Dataset, classes: Sequence[int]):
        self.base = base
        self.map = {int(c): i for i, c in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        return x, self.map[int(y)]


def _subset_by_classes(ds: Dataset, classes: Sequence[int]) -> Subset:
    targets = ds.targets if torch.is_tensor(ds.targets) else torch.as_tensor(ds.targets)
    mask = torch.zeros(len(targets), dtype=torch.bool)
    for c in classes:
        mask |= targets == int(c)
    return Subset(ds, torch.nonzero(mask).squeeze(1).tolist())


def get_split_mnist(root: str = DEFAULT_DATA_ROOT, n_tasks: int = 5, batch_size: int = 128,
                    num_workers: int = 0, task_il: bool = False, class_order_seed: Optional[int] = None):
    """Split MNIST. 10 개 클래스를 n_tasks 개로 나눈다 (기본 5 태스크 x 2 클래스).

    task_il=False : Class-IL. 라벨 0..9 그대로, 출력층 10 유닛 공유 (더 어려움, 기본).
    task_il=True  : Task-IL. 태스크 내부 라벨 0..k-1 로 재매핑 (태스크 id 를 안다고 가정).
    """
    assert 10 % n_tasks == 0, "n_tasks must divide 10"
    tf = _mnist_tf()
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)

    classes = list(range(10))
    if class_order_seed is not None:
        np.random.RandomState(class_order_seed).shuffle(classes)
    per = 10 // n_tasks

    tasks: List[Task] = []
    for t in range(n_tasks):
        cls = tuple(classes[t * per:(t + 1) * per])
        tr, te = _subset_by_classes(train, cls), _subset_by_classes(test, cls)
        if task_il:
            tr, te = RemapLabels(tr, cls), RemapLabels(te, cls)
        tasks.append(Task(t, cls, _loader(tr, batch_size, True, num_workers),
                          _loader(te, batch_size * 2, False, num_workers)))
    info = DataInfo("split_mnist", (1, 28, 28), per if task_il else 10)
    return tasks, info


class Permute:
    """픽셀 순열 transform. (1, 28, 28) -> 같은 모양. picklable."""

    def __init__(self, perm: torch.Tensor):
        self.perm = perm.clone()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        return x.reshape(-1)[self.perm].reshape(shape)


def get_permuted_mnist(root: str = DEFAULT_DATA_ROOT, n_tasks: int = 10, batch_size: int = 128,
                       num_workers: int = 0, seed: int = 0):
    """Permuted MNIST. 태스크 0 은 원본, 이후 태스크는 seed 로 고정된 픽셀 순열."""
    rng = np.random.RandomState(seed)
    tasks: List[Task] = []
    for t in range(n_tasks):
        perm = torch.arange(28 * 28) if t == 0 else torch.from_numpy(rng.permutation(28 * 28))
        tf = _mnist_tf([Permute(perm)])
        train = datasets.MNIST(root, train=True, download=True, transform=tf)
        test = datasets.MNIST(root, train=False, download=True, transform=tf)
        tasks.append(Task(t, tuple(range(10)), _loader(train, batch_size, True, num_workers),
                          _loader(test, batch_size * 2, False, num_workers)))
    info = DataInfo("permuted_mnist", (1, 28, 28), 10)
    return tasks, info


# ---------------------------------------------------------------------------
# 레지스트리
# ---------------------------------------------------------------------------

_BASIC = {"mnist": get_mnist, "cifar10": get_cifar10}
_CONTINUAL = {"split_mnist": get_split_mnist, "permuted_mnist": get_permuted_mnist}


def get_dataset(name: str, **kw):
    """기본 분류 데이터셋. (train_loader, test_loader, info) 를 돌려준다."""
    if name not in _BASIC:
        raise KeyError(f"unknown dataset '{name}', choose from {sorted(_BASIC)}")
    return _BASIC[name](**kw)


def get_continual_dataset(name: str, **kw):
    """연속학습 데이터셋. ([Task], info) 를 돌려준다."""
    if name not in _CONTINUAL:
        raise KeyError(f"unknown continual dataset '{name}', choose from {sorted(_CONTINUAL)}")
    return _CONTINUAL[name](**kw)
