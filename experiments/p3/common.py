# -*- coding: utf-8 -*-
"""논문 2 파일럿(p3) 공용: MNIST MLP '세 회사' 구성, 데이터, 평가, 활성 매개변수·살아 있는 뉴런 세기.

회사(company):
  dense        : 784-1024-1024-10 밀집망 (100%)
  pruned       : 같은 망에 논문 2 탐색(p2) 의 학습 중 전역 크기 가지치기 마스크를 고정 (results/p2/masks/e2_mnist_d{d}_global_s{seed}.pt),
                 초기 가중치는 그 런과 같은 시드의 초기화 (복권 티켓식 되감기)
  random_mask  : 같은 망에 층별 밀도만 같은 무작위 마스크
  dense_small  : 같은 가중치 예산의 작은 밀집망 (논문 1 dense small)
가중치는 전부 새로 초기화해 처음부터 학습한다 (마스크만 고정).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baselines.mlp import MLP                                                          # noqa: E402
from core.masked_layers import apply_masks, convert_to_masked, masked_modules         # noqa: E402
from experiments.core_prune_during_learning.run import dense_hidden_for_budget, n_weights  # noqa: E402
from utils.seed import set_seed                                                        # noqa: E402
from utils.tensor_data import load_mnist_tensors                                       # noqa: E402

RES_P3 = os.path.join(REPO_ROOT, "results", "p3")
MASK_DIR = os.path.join(REPO_ROOT, "results", "p2", "masks")
BIG = n_weights(1024)
COMPANIES = ("dense", "pruned", "random_mask", "dense_small")
_DATA: Dict[str, tuple] = {}


def get_mnist(device):
    if "mnist" not in _DATA:
        (x_tr, y_tr), (x_te, y_te) = load_mnist_tensors(device=device)
        _DATA["mnist"] = (x_tr, y_tr, x_te, y_te)
    return _DATA["mnist"]


def linears(model: nn.Module) -> List[nn.Linear]:
    return [m for m in model.net if isinstance(m, nn.Linear)]


def wmask(m: nn.Module):
    return m.weight_mask if hasattr(m, "weight_mask") else None


def eff_w(m: nn.Module) -> torch.Tensor:
    return m.weight * m.weight_mask if hasattr(m, "weight_mask") else m.weight


def mask_path(density: float, seed: int) -> str:
    return os.path.join(MASK_DIR, f"e2_mnist_d{density:g}_global_s{seed}.pt")


def build_company(company: str, density: float, seed: int, device) -> Tuple[nn.Module, dict]:
    """set_seed(seed) 직후 MLP 를 만들어 p2 'global' 런과 같은 초기화를 얻는다 (p2 도 set_seed 뒤 바로 build_model)."""
    set_seed(seed)
    info: dict = {"company": company, "density": density}
    if company == "dense_small":
        h = dense_hidden_for_budget(int(round(density * BIG)))
        info["hidden"] = h
        return MLP((1, 28, 28), 10, hidden=(h, h)).to(device), info
    m = MLP((1, 28, 28), 10, hidden=(1024, 1024))
    if company == "dense":
        return m.to(device), info
    m = convert_to_masked(m).to(device)
    p = mask_path(density, seed)
    masks = torch.load(p, map_location="cpu")
    info["mask_src"] = os.path.relpath(p, REPO_ROOT)
    g = torch.Generator(device="cpu").manual_seed(100_000 + seed)
    with torch.no_grad():
        for name, mod in masked_modules(m):
            mk = masks[name].bool()
            if company == "random_mask":
                k = int(mk.sum().item())
                flat = torch.zeros(mk.numel(), dtype=torch.bool)
                flat[torch.randperm(mk.numel(), generator=g)[:k]] = True
                mk = flat.view_as(mk)
            mod.weight_mask.copy_(mk.to(device=device, dtype=mod.weight_mask.dtype))
    apply_masks(m)
    return m, info


@torch.no_grad()
def evaluate(model: nn.Module, x_te: torch.Tensor, y_te: torch.Tensor, bs: int = 2000) -> float:
    was = model.training
    model.eval()
    correct = 0
    for s in range(0, x_te.shape[0], bs):
        correct += int((model(x_te[s:s + bs]).argmax(1) == y_te[s:s + bs]).sum().item())
    model.train(was)
    return correct / x_te.shape[0]


@torch.no_grad()
def count_active(model: nn.Module) -> int:
    tot = 0
    for m in linears(model):
        mk = wmask(m)
        tot += int(mk.sum().item()) if mk is not None else m.weight.numel()
    return tot


@torch.no_grad()
def alive_hidden_neurons(model: nn.Module) -> int:
    """은닉 뉴런 중 들어오는 연결과 나가는 연결이 모두 하나 이상 살아 있는 수."""
    L = linears(model)
    total = 0
    for i in range(len(L) - 1):
        w_in = eff_w(L[i]) if wmask(L[i]) is None else L[i].weight_mask
        w_out = eff_w(L[i + 1]) if wmask(L[i + 1]) is None else L[i + 1].weight_mask
        has_in = (w_in != 0).any(dim=1)
        has_out = (w_out != 0).any(dim=0)
        total += int((has_in & has_out).sum().item())
    return total


def hidden_neurons(model: nn.Module) -> int:
    return sum(m.out_features for m in linears(model)[:-1])
