# -*- coding: utf-8 -*-
"""성능 / 비용 지표.

논문 비교축 5 개 (사용자 확정, ADR-001 참고). 에너지 실측은 하지 않는다 (폰 노이만 GPU 위에서 돌기 때문).

1. 정확도            : accuracy / evaluate
2. 이론적 FLOPs       : count_flops (dense / effective), 누적 학습 FLOPs 는 학습 루프에서 합산
3. 망각 저항성        : continual_metrics (avg_acc, forgetting, bwt)
4. 데이터 효율        : data_efficiency (학습곡선 아래 넓이, 목표 정확도 도달 표본 수)
5. 확신도 보정        : calibration_metrics (ECE, MCE, NLL, Brier, AUROC, AURC, 선택적 정확도)

FLOPs 규약
- 1 샘플 순전파 기준. 곱셈과 덧셈을 각각 센다 (matmul = 2*M*N*K). MAC 이 아니다.
- dense     = torch.utils.flop_counter 가 센 전체 (어텐션 행렬곱 포함)
- effective = dense - sum_leaf( leaf_dense * (1 - density_leaf) )
  리프(Linear/Conv) 의 dense FLOPs 는 forward hook 으로 해석적으로 센다.
  프루닝 가능 모듈에 `weight_mask` 버퍼가 있으면 weight * mask 로 밀도를 구한다.
- 학습 FLOPs 근사 = 순전파 x 3 (역전파를 순전파의 2 배로 보는 통상 근사). 학습 루프에서 계산.
- 에너지는 이론 추정만: theoretical_energy_j(flops, pj_per_flop). 상수는 문헌값을 호출자가 넘긴다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

# numpy 2 는 trapz 를 trapezoid 로 바꿨다 (팟 numpy 2.x 에서 AttributeError)
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

PRUNABLE = (nn.Linear, nn.Conv1d, nn.Conv2d)


# ---------------------------------------------------------------------------
# 1. 파라미터 / 활성 연결
# ---------------------------------------------------------------------------

def effective_weight(m: nn.Module) -> torch.Tensor:
    mask = getattr(m, "weight_mask", None)
    return m.weight if mask is None else m.weight * mask


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


@torch.no_grad()
def count_active_params(model: nn.Module) -> int:
    """활성 파라미터 수. 프루닝 가능 가중치는 0 이 아닌 원소만, 나머지는 전부 센다."""
    prunable_w = {id(m.weight): m for m in model.modules() if isinstance(m, PRUNABLE)}
    total = 0
    for p in model.parameters():
        if id(p) in prunable_w:
            total += int((effective_weight(prunable_w[id(p)]) != 0).sum().item())
        else:
            total += p.numel()
    return total


@torch.no_grad()
def layer_densities(model: nn.Module) -> Dict[str, float]:
    return {name: float((effective_weight(m) != 0).float().mean().item())
            for name, m in model.named_modules() if isinstance(m, PRUNABLE)}


# ---------------------------------------------------------------------------
# 2. FLOPs
# ---------------------------------------------------------------------------

@dataclass
class FlopReport:
    dense: float                 # 1 샘플 순전파 dense FLOPs
    effective: float             # 리프 밀도를 반영한 실효 FLOPs
    leaf_dense: Dict[str, float] = field(default_factory=dict)
    leaf_density: Dict[str, float] = field(default_factory=dict)

    @property
    def leaf_total(self) -> float:
        return float(sum(self.leaf_dense.values()))

    def as_dict(self) -> dict:
        return {"dense": self.dense, "effective": self.effective,
                "leaf_total": self.leaf_total, "leaf_dense": self.leaf_dense,
                "leaf_density": self.leaf_density}


def _leaf_flops_via_hooks(model: nn.Module, x: torch.Tensor) -> Dict[str, float]:
    counts: Dict[str, float] = {}
    handles = []

    def make_hook(name: str):
        def hook(m, inp, out):
            if isinstance(m, nn.Linear):
                n_pos = inp[0].numel() // inp[0].shape[-1]
                f = 2.0 * n_pos * m.in_features * m.out_features
            else:  # Conv1d / Conv2d
                out_pos = out.numel() // out.shape[1]  # batch * spatial positions
                k = 1
                for s in m.kernel_size:
                    k *= s
                f = 2.0 * out_pos * m.out_channels * (m.in_channels // m.groups) * k
            counts[name] = counts.get(name, 0.0) + f
        return hook

    for name, m in model.named_modules():
        if isinstance(m, PRUNABLE):
            handles.append(m.register_forward_hook(make_hook(name)))
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
    return counts


@torch.no_grad()
def count_flops(model: nn.Module, input_shape: Sequence[int], device=None) -> FlopReport:
    """1 샘플 기준 dense / effective FLOPs. 호출 뒤 모델의 train/eval 상태를 원래대로 되돌린다."""
    was_training = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    x = torch.zeros((1,) + tuple(input_shape), device=device)

    with FlopCounterMode(display=False) as fc:
        model(x)
    dense = float(fc.get_total_flops())

    leaf = _leaf_flops_via_hooks(model, x)
    dens = layer_densities(model)
    effective = dense
    for name, f in leaf.items():
        effective -= f * (1.0 - dens.get(name, 1.0))

    model.train(was_training)
    return FlopReport(dense=dense, effective=effective, leaf_dense=leaf, leaf_density=dens)


def training_flops(forward_flops_per_sample: float, n_samples: int, backward_multiplier: float = 2.0) -> float:
    """학습 FLOPs 근사. 순전파 + 역전파(순전파의 backward_multiplier 배)."""
    return forward_flops_per_sample * n_samples * (1.0 + backward_multiplier)


def theoretical_energy_j(flops: float, pj_per_flop: float) -> float:
    """이론적 에너지 추정 (J). pj_per_flop 은 문헌값(예: Horowitz 2014 의 공정별 수치)을 호출자가 넘긴다.
    실측이 아니라는 점을 보고서에 반드시 명시할 것."""
    return flops * pj_per_flop * 1e-12


# ---------------------------------------------------------------------------
# 3. 망각 저항성
# ---------------------------------------------------------------------------

def continual_metrics(R) -> Dict[str, float]:
    """R: (T, T) 배열. R[i][j] = 태스크 i 까지 학습한 뒤 태스크 j 의 정확도 (j <= i 만 유효).

    avg_acc    : 마지막 시점의 전 태스크 평균 정확도
    forgetting : 각 과거 태스크의 (최고 정확도 - 마지막 정확도) 평균 (Chaudhry 2018). 클수록 나쁨
    bwt        : 각 과거 태스크의 (마지막 정확도 - 학습 직후 정확도) 평균 (Lopez-Paz 2017). 음수면 망각
    retention  : 각 과거 태스크의 (마지막 정확도 / 학습 직후 정확도) 평균. ARCHITECTURE.md 의 '유지율'
    """
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    out = {"avg_acc": float(R[T - 1, :T].mean())}
    if T > 1:
        out["forgetting"] = float(np.mean([R[:T - 1, j].max() - R[T - 1, j] for j in range(T - 1)]))
        out["bwt"] = float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)]))
        out["retention"] = float(np.mean([R[T - 1, j] / max(R[j, j], 1e-12) for j in range(T - 1)]))
    else:
        out.update({"forgetting": 0.0, "bwt": 0.0, "retention": 1.0})
    return out


# ---------------------------------------------------------------------------
# 4. 데이터 효율
# ---------------------------------------------------------------------------

def data_efficiency(samples_seen: Sequence[float], acc: Sequence[float],
                    target_acc: Optional[float] = None) -> Dict[str, Optional[float]]:
    """학습곡선 (표본 수 -> 정확도) 요약.

    aulc               : 학습곡선 아래 넓이를 전체 표본 수로 나눈 값 (0~1, 클수록 빨리 배움)
    samples_to_target  : 정확도가 target_acc 에 처음 도달하는 표본 수 (선형 보간). 미도달이면 None
    """
    s = np.asarray(samples_seen, dtype=float)
    a = np.asarray(acc, dtype=float)
    if s.size == 0:
        return {"aulc": None, "samples_to_target": None}
    order = np.argsort(s)
    s, a = s[order], a[order]
    if s[0] > 0:  # 시작점(0 표본, 우연 수준 미지)을 첫 관측으로 보정하지 않고 첫 관측부터 적분
        pass
    aulc = float(_trapz(a, s) / max(s[-1] - s[0], 1e-12)) if s.size > 1 else float(a[0])
    hit = None
    if target_acc is not None:
        idx = np.nonzero(a >= target_acc)[0]
        if idx.size:
            i = int(idx[0])
            if i == 0:
                hit = float(s[0])
            else:
                s0, s1, a0, a1 = s[i - 1], s[i], a[i - 1], a[i]
                hit = float(s0 + (target_acc - a0) / max(a1 - a0, 1e-12) * (s1 - s0))
    return {"aulc": aulc, "samples_to_target": hit}


# ---------------------------------------------------------------------------
# 5. 확신도 보정
# ---------------------------------------------------------------------------

def _auroc(score: np.ndarray, positive: np.ndarray) -> Optional[float]:
    """Mann-Whitney U 기반 AUROC (동점은 평균 순위). positive 가 한 종류뿐이면 None."""
    from scipy.stats import rankdata
    pos = positive.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(score)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def calibration_metrics(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15,
                        coverages: Sequence[float] = (0.9, 0.8, 0.5)) -> Dict[str, Optional[float]]:
    """probs: (N, C) 소프트맥스 확률, labels: (N,) 정답.

    ece / mce   : 신뢰도-정확도 갭 (등폭 15 구간). 낮을수록 좋음
    nll / brier : 확률 품질
    auroc       : 확신도가 정답/오답을 얼마나 가르는가 (0.5 = 못 가름)
    aurc        : 위험-커버리지 곡선 아래 넓이 (Geifman 2018). 낮을수록 좋음
    sel_acc@c   : 확신도 상위 c 비율만 답할 때의 정확도 ('모르겠다' 능력)
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels).astype(int)
    n = labels.shape[0]
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = (pred == labels).astype(float)

    out: Dict[str, Optional[float]] = {"acc": float(correct.mean())}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            mce = max(mce, gap)
    out["ece"] = float(ece)
    out["mce"] = float(mce)

    p_true = np.clip(probs[np.arange(n), labels], 1e-12, 1.0)
    out["nll"] = float(-np.log(p_true).mean())
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    out["brier"] = float(((probs - onehot) ** 2).sum(1).mean())

    out["auroc"] = _auroc(conf, correct)

    order = np.argsort(-conf, kind="stable")
    err_sorted = 1.0 - correct[order]
    risk = np.cumsum(err_sorted) / np.arange(1, n + 1)
    out["aurc"] = float(risk.mean())
    for c in coverages:
        k = max(1, int(round(c * n)))
        out[f"sel_acc@{c:g}"] = float(correct[order][:k].mean())
    return out


# ---------------------------------------------------------------------------
# 평가 헬퍼
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model: nn.Module, loader, device, amp: bool = False):
    """(probs (N, C), labels (N,)) 를 numpy 로 돌려준다. 확신도 보정 계산용."""
    was_training = model.training
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and x.is_cuda)):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        labels.append(y)
    model.train(was_training)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def accuracy(model: nn.Module, loader, device, amp: bool = False) -> float:
    was_training = model.training
    model.eval()
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and x.is_cuda)):
            logits = model(x)
        correct += int((logits.argmax(1) == y).sum().item())
        n += y.numel()
    model.train(was_training)
    return correct / max(n, 1)
