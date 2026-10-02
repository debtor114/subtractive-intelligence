# -*- coding: utf-8 -*-
"""학습 중 가지치기 규칙 (핵심 실험, 실험 3 의 수면 단계에서 사용).

점수(score) 가 낮은 연결을 끊는다. 점수 종류:
- magnitude : |w|                           (Han 2015. 가장 흔한 기준. 대조군)
- activity  : |c_ij|                        (헤비안 공활성 흔적. pre 입력 x_j 와 post 출력 y_i 의 곱의 EMA.
                                             gradient 를 쓰지 않는 지역 규칙. STDP 의 '인과 상관' 을 ANN 에 옮긴 것)
- activity_mag : |w_ij| * |c_ij|            (연결 강도 x 사용량. '쓰이는 강한 연결' 만 남김)
- random    : 무작위                         (하한 대조군)

ActivityTracker 는 forward hook 으로 각 마스크 층의 (post, pre) 공활성 행렬을 EMA 로 유지한다.
메모리는 가중치와 같은 크기. post 는 층의 출력(활성화 함수 적용 전) 의 절대값이 아니라 부호 있는 값을 쓰고,
pre 도 부호 있는 값을 써서 c_ij = E[y_i * x_j] (헤비안 상관) 로 둔다. 배치 평균은 batch 행렬곱 한 번이다.

스케줄:
- cubic (Zhu & Gupta 2017): 밀도 d(t) = d_final + (d_init - d_final) * (1 - t/T)^3, t 는 프루닝 구간 내 진행도.
- 프루닝 구간은 [begin_step, end_step], 간격 every 스텝.
- 전역(global) 또는 층별(layerwise) 임계값.
- regrow 옵션: 끊은 수만큼 무작위(SET) 로 다시 잇는다 -> 동적 희소 학습 대조군.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .masked_layers import MASKED_TYPES, MaskedConv2d, MaskedLinear, apply_masks, masked_modules


# ---------------------------------------------------------------------------
# 활동 추적
# ---------------------------------------------------------------------------

class ActivityTracker:
    """마스크 층마다 c = EMA[ y^T x / B ] (post x pre) 를 유지. Linear 만 정확히 지원, Conv 는 채널 평균 근사.

    post_nonlin=True 면 post 를 relu(y) 로 둔다. 발화율은 음수가 없다는 뜻에서 STDP 비유에 더 가깝다.
    (다음 층 입력 x 는 이미 ReLU 를 지난 값이라 음수가 없다. 첫 층 입력은 정규화 픽셀이라 부호가 있다.)"""

    def __init__(self, model: nn.Module, momentum: float = 0.99, post_nonlin: bool = True):
        self.momentum = momentum
        self.post_nonlin = post_nonlin
        self.traces: Dict[str, torch.Tensor] = {}      # 헤비안 상관 E[y_i x_j]
        self.drive: Dict[str, torch.Tensor] = {}       # 후뉴런 발화 시 시냅스 전 입력 크기 E[1(y_i>0) |x_j|]
        self.handles = []
        self.enabled = True
        for name, m in masked_modules(model):
            self.traces[name] = torch.zeros_like(m.weight)
            self.drive[name] = torch.zeros_like(m.weight)
            self.handles.append(m.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(m, inp, out):
            if not self.enabled or not torch.is_grad_enabled():
                return
            with torch.no_grad():
                x = inp[0].detach().float()
                y = out.detach().float()
                fire = (y > 0).float()
                if self.post_nonlin:
                    y = torch.relu(y)
                if isinstance(m, MaskedLinear):
                    x2 = x.reshape(-1, x.shape[-1])            # (B*, in)
                    y2 = y.reshape(-1, y.shape[-1])            # (B*, out)
                    f2 = fire.reshape(-1, fire.shape[-1])
                    c = y2.t() @ x2 / max(x2.shape[0], 1)      # (out, in)
                    d = f2.t() @ x2.abs() / max(x2.shape[0], 1)
                else:  # Conv2d: 시냅스(가중치 원소) 단위로 정확히 센다. unfold 로 각 출력 위치의 수용영역을 펼친다.
                    B = x.shape[0]
                    unf = torch.nn.functional.unfold(x, m.kernel_size, dilation=m.dilation, padding=m.padding,
                                                     stride=m.stride)                      # (B, in*k*k, L)
                    L = unf.shape[-1]
                    f2 = fire.reshape(B, fire.shape[1], -1)                                # (B, out, L)
                    y2 = y.reshape(B, y.shape[1], -1)
                    d = torch.einsum("bol,bil->oi", f2, unf.abs()) / max(B * L, 1)        # 후뉴런 발화 x 입력 크기
                    c = torch.einsum("bol,bil->oi", y2, unf) / max(B * L, 1)              # 헤비안 상관
                    d = d.reshape(m.weight.shape)
                    c = c.reshape(m.weight.shape)
                self.traces[name].mul_(self.momentum).add_(c * (1.0 - self.momentum))
                self.drive[name].mul_(self.momentum).add_(d * (1.0 - self.momentum))
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------------------
# 점수와 프루닝
# ---------------------------------------------------------------------------

@torch.no_grad()
def connection_scores(model: nn.Module, rule: str, tracker: Optional[ActivityTracker] = None,
                      eps: float = 1e-12) -> Dict[str, torch.Tensor]:
    scores: Dict[str, torch.Tensor] = {}
    for name, m in masked_modules(model):
        w = m.weight.abs()
        if rule == "magnitude":
            s = w
        elif rule == "random":
            s = torch.rand_like(w)
        elif rule in ("activity", "activity_mag"):
            assert tracker is not None, "activity rules need an ActivityTracker"
            c = tracker.traces[name].abs()
            # 층마다 스케일이 다르므로 층 내부에서 정규화 (전역 임계값 비교용)
            c = c / (c.mean() + eps)
            s = c if rule == "activity" else (w / (w.mean() + eps)) * c
        elif rule == "drive":
            # 시냅스 구동: |w_ij| * E[ 1(post_i 발화) * |x_j| ]. 후뉴런이 발화할 때 그 시냅스로 흐른 평균 입력량.
            # STDP 의 '전 발화 + 후 발화' 조건을 시냅스 단위로 옮긴 지역 규칙. gradient 불필요.
            assert tracker is not None, "drive rule needs an ActivityTracker"
            d = tracker.drive[name]
            s = w * (d / (d.mean() + eps))
        elif rule == "drive_norm":
            # 항상성 판: 구동을 후뉴런별 평균으로 나눠 '어느 뉴런이 많이 발화하나' 를 지우고
            # '각 뉴런에 어느 입력이 중요한가' 만 남긴다. 발화가 적은 채널의 입력이 통째로 잘려 죽는 붕괴를 막는다.
            assert tracker is not None, "drive_norm rule needs an ActivityTracker"
            d = tracker.drive[name]
            per_out = d.reshape(d.shape[0], -1).mean(dim=1).clamp_min(eps)
            d_norm = d / per_out.view(-1, *([1] * (d.dim() - 1)))
            s = w * d_norm
            s = s / (s.mean() + eps)
        else:
            raise KeyError(f"unknown pruning rule '{rule}'")
        scores[name] = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    return scores


@torch.no_grad()
def prune_to_density(model: nn.Module, target_density: float, rule: str = "magnitude",
                     tracker: Optional[ActivityTracker] = None, scope: str = "global",
                     per_layer: Optional[Dict[str, float]] = None) -> int:
    """마스크를 갱신해 전체(또는 층별) 밀도를 target_density 로 낮춘다. 이미 끊긴 연결은 다시 잇지 않는다.
    scope="erk" 면 층별 목표를 ERK 배분(erk_densities)으로 정한다. per_layer 를 주면 그 층별 목표를 쓴다.
    끊은 연결 수를 돌려준다."""
    mods = masked_modules(model)
    scores = connection_scores(model, rule, tracker)
    removed = 0
    if scope == "erk" and per_layer is None:
        per_layer = erk_densities(model, target_density)
    if per_layer is not None:
        for n, m in mods:
            k_keep = int(round(float(per_layer.get(n, target_density)) * m.weight_mask.numel()))
            removed += _keep_topk(m, scores[n], k_keep)
        apply_masks(model)
        return removed
    if scope == "global":
        # 전역: 살아있는 연결 전체에서 정확히 k_keep 개를 고른다 (동점이어도 수가 부풀지 않게 인덱스로 선택)
        flat_scores = torch.cat([scores[n].flatten() for n, m in mods])
        flat_alive = torch.cat([m.weight_mask.bool().flatten() for n, m in mods])
        total = flat_scores.numel()
        k_keep = int(round(target_density * total))
        n_alive = int(flat_alive.sum().item())
        if k_keep >= n_alive:
            return 0
        masked_scores = torch.where(flat_alive, flat_scores, torch.full_like(flat_scores, float("-inf")))
        keep_idx = torch.topk(masked_scores, k_keep, largest=True).indices
        new_flat = torch.zeros_like(flat_alive)
        new_flat[keep_idx] = True
        removed = n_alive - k_keep
        off = 0
        for n, m in mods:
            cnt = m.weight_mask.numel()
            m.weight_mask.copy_(new_flat[off:off + cnt].view_as(m.weight_mask).to(m.weight_mask.dtype))
            off += cnt
    else:
        for n, m in mods:
            k_keep = int(round(target_density * m.weight_mask.numel()))
            removed += _keep_topk(m, scores[n], k_keep)
    apply_masks(model)
    return removed


@torch.no_grad()
def _keep_topk(m: nn.Module, score: torch.Tensor, k_keep: int) -> int:
    """한 층에서 살아있는 연결 중 점수 상위 k_keep 개만 남긴다 (정확한 개수, 동점 무관). 끊은 수를 돌려준다."""
    alive = m.weight_mask.bool().flatten()
    n_alive = int(alive.sum().item())
    if k_keep >= n_alive:
        return 0
    flat = score.flatten()
    masked = torch.where(alive, flat, torch.full_like(flat, float("-inf")))
    keep_idx = torch.topk(masked, max(k_keep, 0), largest=True).indices if k_keep > 0 else torch.empty(0, dtype=torch.long, device=flat.device)
    new_flat = torch.zeros_like(alive)
    new_flat[keep_idx] = True
    m.weight_mask.copy_(new_flat.view_as(m.weight_mask).to(m.weight_mask.dtype))
    return n_alive - k_keep


@torch.no_grad()
def prune_below(model: nn.Module, threshold: float, rule: str = "magnitude",
                tracker: Optional[ActivityTracker] = None) -> int:
    """점수가 threshold 미만인 연결을 끊는다 (실험 3 수면 단계의 '약한 연결 제거')."""
    scores = connection_scores(model, rule, tracker)
    removed = 0
    for n, m in masked_modules(model):
        new_mask = (scores[n] >= threshold) & m.weight_mask.bool()
        removed += int((m.weight_mask.bool() & ~new_mask).sum().item())
        m.weight_mask.copy_(new_mask.to(m.weight_mask.dtype))
    apply_masks(model)
    return removed


@torch.no_grad()
def regrow_random(model: nn.Module, n_per_layer: Dict[str, int], init_std: float = 0.0) -> int:
    """끊긴 자리 중 무작위로 n 개를 다시 잇는다 (SET). 새 연결 가중치는 0 (또는 작은 난수)."""
    grown = 0
    for name, m in masked_modules(model):
        n = int(n_per_layer.get(name, 0))
        if n <= 0:
            continue
        dead = (~m.weight_mask.bool()).flatten().nonzero().squeeze(1)
        if dead.numel() == 0:
            continue
        pick = dead[torch.randperm(dead.numel(), device=dead.device)[:n]]
        flat_mask = m.weight_mask.flatten()
        flat_mask[pick] = 1.0
        m.weight_mask.copy_(flat_mask.view_as(m.weight_mask))
        flat_w = m.weight.flatten()
        flat_w[pick] = torch.randn(pick.numel(), device=flat_w.device) * init_std if init_std > 0 else 0.0
        m.weight.copy_(flat_w.view_as(m.weight))
        grown += pick.numel()
    return grown


@torch.no_grad()
def regrow_by_gradient(model: nn.Module, n_per_layer: Dict[str, int]) -> int:
    """끊긴 자리 중 |grad| 가 큰 n 개를 다시 잇는다 (RigL). 호출 전에 backward 로 grad 가 있어야 하고,
    마스크 층의 dense_grad=True 여야 끊긴 자리의 gradient 가 보인다."""
    grown = 0
    for name, m in masked_modules(model):
        n = int(n_per_layer.get(name, 0))
        g_src = m.dense_weight_grad() if hasattr(m, "dense_weight_grad") else None
        if g_src is None:
            g_src = m.weight.grad
        if n <= 0 or g_src is None:
            continue
        g = torch.nan_to_num(g_src.abs().flatten().clone(), nan=0.0, posinf=0.0)
        g[m.weight_mask.bool().flatten()] = -1.0  # 살아있는 연결은 후보에서 제외
        pick = torch.topk(g, min(n, int((g >= 0).sum().item())), largest=True).indices
        flat_mask = m.weight_mask.flatten()
        flat_mask[pick] = 1.0
        m.weight_mask.copy_(flat_mask.view_as(m.weight_mask))
        flat_w = m.weight.flatten()
        flat_w[pick] = 0.0
        m.weight.copy_(flat_w.view_as(m.weight))
        grown += pick.numel()
    return grown


@torch.no_grad()
def random_sparse_init(model: nn.Module, density: float, per_layer: Optional[Dict[str, float]] = None) -> None:
    """마스크 층에 무작위 마스크 (SET / RigL / static sparse 의 출발점). per_layer 가 있으면 층별 밀도를 쓴다."""
    for name, m in masked_modules(model):
        d = float(per_layer.get(name, density)) if per_layer else density
        n = m.weight_mask.numel()
        k = int(round(min(1.0, d) * n))
        flat = torch.zeros(n, device=m.weight_mask.device)
        flat[torch.randperm(n, device=flat.device)[:k]] = 1.0
        m.weight_mask.copy_(flat.view_as(m.weight_mask))
    apply_masks(model)


def erk_densities(model: nn.Module, global_density: float) -> Dict[str, float]:
    """Erdos-Renyi-Kernel 층별 밀도 (Evci 2020). 밀도_l ∝ (sum of dims)/(prod of dims), 전체 예산에 맞춰 스케일.
    1 을 넘는 층은 dense 로 고정하고 나머지에 예산을 재배분한다."""
    mods = masked_modules(model)
    numel = {n: m.weight.numel() for n, m in mods}
    raw = {}
    for n, m in mods:
        shape = tuple(m.weight.shape)
        raw[n] = sum(shape) / float(numel[n])
    total = sum(numel.values())
    budget = global_density * total
    dense = set()
    while True:
        rest_budget = budget - sum(numel[n] for n in dense)
        rest_raw = sum(raw[n] * numel[n] for n in raw if n not in dense)
        if rest_raw <= 0:
            break
        eps = rest_budget / rest_raw
        over = [n for n in raw if n not in dense and raw[n] * eps > 1.0]
        if not over:
            break
        dense.update(over)
    out = {}
    for n in raw:
        out[n] = 1.0 if n in dense else min(1.0, raw[n] * eps)
    return out


@torch.no_grad()
def dynamic_sparse_step(model: nn.Module, frac: float, regrow: str = "random") -> int:
    """동적 희소 학습 한 번: 층마다 살아있는 연결의 frac 만큼을 |w| 가 작은 순으로 끊고, 같은 수를 다시 잇는다.
    regrow: random (SET, Mocanu 2018) | gradient (RigL, Evci 2020). 밀도는 유지된다."""
    per_layer: Dict[str, int] = {}
    if regrow == "gradient":
        # AMP 에서 손실 스케일 오버플로 스텝은 grad 가 inf/NaN 이다. 그 스텝에 끊기만 하고 못 이으면 연결이 새어
        # 나가 망이 죽는다 (CIFAR 1% seed 2 에서 21,957 -> 1,459 로 붕괴). grad 가 유한하지 않으면 이번 갱신을 건너뛴다.
        for n, m in masked_modules(model):
            g = m.dense_weight_grad() if hasattr(m, "dense_weight_grad") else None
            if g is None:
                g = m.weight.grad
            if g is None or not torch.isfinite(g).all():
                return 0
    for n, m in masked_modules(model):
        alive_mask = m.weight_mask.bool().flatten()
        alive = int(alive_mask.sum().item())
        k = int(round(alive * frac))
        if k <= 0:
            continue
        s = m.weight.abs().flatten().clone()
        s[~alive_mask] = float("inf")
        drop = torch.topk(s, k, largest=False).indices
        fm = m.weight_mask.flatten()
        fm[drop] = 0.0
        m.weight_mask.copy_(fm.view_as(m.weight_mask))
        per_layer[n] = k
    if regrow == "random":
        grown = regrow_random(model, per_layer)
    elif regrow == "gradient":
        grown = regrow_by_gradient(model, per_layer)
    else:
        raise KeyError(regrow)
    apply_masks(model)
    return grown


# ---------------------------------------------------------------------------
# 스케줄
# ---------------------------------------------------------------------------

def cubic_density(step: int, begin: int, end: int, d_init: float, d_final: float) -> float:
    if step <= begin:
        return d_init
    if step >= end:
        return d_final
    p = (step - begin) / max(1, end - begin)
    return d_final + (d_init - d_final) * (1.0 - p) ** 3


class PruningScheduler:
    """학습 루프에서 매 스텝 step(global_step) 을 부른다. 프루닝이 일어나면 True."""

    def __init__(self, model: nn.Module, *, rule: str, d_final: float, begin_step: int, end_step: int,
                 every: int, scope: str = "global", tracker: Optional[ActivityTracker] = None,
                 regrow: Optional[str] = None, regrow_frac: float = 0.0, d_init: float = 1.0):
        self.model, self.rule, self.d_final = model, rule, d_final
        self.begin, self.end, self.every, self.scope = begin_step, end_step, every, scope
        self.tracker = tracker
        self.regrow, self.regrow_frac, self.d_init = regrow, regrow_frac, d_init
        self.log: List[Tuple[int, float, int]] = []  # (step, density, removed)

    def target(self, step: int) -> float:
        return cubic_density(step, self.begin, self.end, self.d_init, self.d_final)

    def step(self, global_step: int) -> bool:
        if global_step < self.begin or global_step > self.end:
            return False
        if (global_step - self.begin) % self.every != 0 and global_step != self.end:
            return False  # end 에서는 간격과 무관하게 한 번 더 깎아 최종 밀도를 정확히 맞춘다
        target = self.target(global_step)
        removed = prune_to_density(self.model, target, self.rule, self.tracker, self.scope)
        if self.regrow and self.regrow_frac > 0:
            # 동적 희소 학습: 살아있는 연결의 regrow_frac 만큼 추가로 끊고 같은 수를 다시 잇는다 (밀도 유지)
            mods = masked_modules(self.model)
            per_layer: Dict[str, int] = {}
            for n, m in mods:
                alive = int(m.weight_mask.sum().item())
                k = int(round(alive * self.regrow_frac))
                if k > 0:
                    scores = connection_scores(self.model, "magnitude")[n]
                    s = scores.flatten().clone()
                    s[~m.weight_mask.bool().flatten()] = float("inf")
                    drop = torch.topk(s, k, largest=False).indices
                    fm = m.weight_mask.flatten()
                    fm[drop] = 0.0
                    m.weight_mask.copy_(fm.view_as(m.weight_mask))
                    per_layer[n] = k
            if self.regrow == "random":
                regrow_random(self.model, per_layer)
            elif self.regrow == "gradient":
                regrow_by_gradient(self.model, per_layer)
            apply_masks(self.model)
        from .masked_layers import mask_density
        self.log.append((global_step, mask_density(self.model), removed))
        return True
