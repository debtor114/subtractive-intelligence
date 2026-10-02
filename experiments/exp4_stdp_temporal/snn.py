# -*- coding: utf-8 -*-
"""Diehl & Cook (2015) 형 비지도 STDP 스파이킹 망, PyTorch GPU 벡터화 + 배치 구현.

구조 (BindsNET 의 DiehlAndCook2015v2 단순화판을 따른다):
- 입력 784 포아송 뉴런. 발화율 = 픽셀 강도(0~1) x intensity Hz. dt = 1 ms.
- 흥분성 LIF 뉴런 n_e 개: v_rest -65, v_reset -60, v_thresh -52 (+ 적응 임계 theta), 불응기 5 ms, tc 100 ms.
- 측면 억제: 한 뉴런이 발화하면 다음 스텝에 나머지 전부의 막전위를 inh 만큼 낮춘다 (승자독식).
- 입력->흥분 시냅스: 흔적 기반 STDP. 시냅스 전 발화 시 LTD (-nu_pre * x_post), 후 발화 시 LTP (+nu_post * x_pre).
  가중치는 [0, wmax] 로 자르고, 이미지(배치) 마다 뉴런별 입력 가중치 합을 norm 으로 정규화한다.
- 적응 임계 theta: 발화마다 theta_plus 증가, 매우 느리게 감쇠 (항상성).

배치: B 개 이미지를 동시에 시뮬레이션하고 STDP 갱신은 배치 평균으로 적용한다 (BindsNET 과 같은 근사).
theta 갱신도 배치 평균 (이미지 1 장당 적응량이 배치 크기에 무관하도록).

이론 연산량 (사건 구동 기준, 에너지 실측 아님):
- 추론: 입력 스파이크 1 개당 n_e 번 누산 + 흥분 스파이크 1 개당 n_e 번 억제 누산 + 스텝마다 n_e 번 누출/임계 갱신
- 학습: 시냅스 전 스파이크 1 개당 n_e 번 (LTD), 후 스파이크 1 개당 784 번 (LTP) 갱신
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


@dataclass
class SNNConfig:
    n_input: int = 784
    n_e: int = 400
    dt: float = 1.0            # ms
    time: int = 250            # 제시 시간 (ms)
    intensity: float = 128.0   # Hz at pixel 1.0
    v_rest: float = -65.0
    v_reset: float = -60.0
    v_thresh: float = -52.0
    refrac: int = 5            # ms
    tc_decay: float = 100.0    # ms
    tc_trace: float = 20.0     # ms
    theta_plus: float = 0.05
    tc_theta: float = 1e7      # ms
    inh: float = 17.5
    nu_pre: float = 1e-4
    nu_post: float = 1e-2
    wmax: float = 1.0
    norm: float = 78.4         # = 0.1 * 784
    batch_size: int = 32
    reduction: str = "sum"     # 배치 STDP 합산(sum) / 평균(mean). sum 이 순차 제시에 가깝다.


@dataclass
class OpCounts:
    """사건 구동 이론 연산 수 (누적)."""
    input_spikes: float = 0.0
    exc_spikes: float = 0.0
    steps: float = 0.0
    images: int = 0

    def inference_ops_per_image(self, n_e: int) -> Dict[str, float]:
        n = max(self.images, 1)
        syn = self.input_spikes * n_e / n            # 입력 스파이크 -> 시냅스 누산
        inh = self.exc_spikes * n_e / n              # 억제 누산
        upd = self.steps * n_e / n                   # 뉴런 상태 갱신 (누출 + 임계 비교)
        return {"synaptic_acc": syn, "inhibition_acc": inh, "neuron_updates": upd, "total": syn + inh + upd,
                "input_spikes": self.input_spikes / n, "exc_spikes": self.exc_spikes / n}

    def learning_ops_per_image(self, n_input: int, n_e: int) -> float:
        n = max(self.images, 1)
        return (self.input_spikes * n_e + self.exc_spikes * n_input) / n


class DiehlCookSNN:
    def __init__(self, cfg: SNNConfig, device, seed: int = 0):
        self.cfg = cfg
        self.device = device
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        self.gen = g
        self.W = torch.rand(cfg.n_input, cfg.n_e, device=device, generator=g) * 0.3   # BindsNET 초기화
        self.theta = torch.zeros(cfg.n_e, device=device)
        self.decay_v = math.exp(-cfg.dt / cfg.tc_decay)
        self.decay_tr = math.exp(-cfg.dt / cfg.tc_trace)
        self.decay_theta = math.exp(-cfg.dt / cfg.tc_theta)
        self.normalize()
        self.ops_train = OpCounts()
        self.ops_eval = OpCounts()

    @torch.no_grad()
    def normalize(self) -> None:
        s = self.W.sum(0, keepdim=True)
        self.W.mul_(self.cfg.norm / s.clamp_min(1e-12))

    @torch.no_grad()
    def run_batch(self, images: torch.Tensor, learn: bool, ops: Optional[OpCounts] = None) -> torch.Tensor:
        """images: (B, 784) 0~1 강도. 반환: (B, n_e) 스파이크 수."""
        cfg = self.cfg
        B = images.shape[0]
        p = (images * cfg.intensity * cfg.dt / 1000.0).clamp_(0, 1)         # 스텝당 발화 확률
        v = torch.full((B, cfg.n_e), cfg.v_rest, device=self.device)
        refrac = torch.zeros((B, cfg.n_e), device=self.device)
        s_prev = torch.zeros((B, cfg.n_e), device=self.device)
        x_pre = torch.zeros((B, cfg.n_input), device=self.device)
        x_post = torch.zeros((B, cfg.n_e), device=self.device)
        counts = torch.zeros((B, cfg.n_e), device=self.device)
        steps = int(cfg.time / cfg.dt)
        for _ in range(steps):
            s_in = (torch.rand(p.shape, device=self.device, generator=self.gen) < p).float()
            I = s_in @ self.W                                                # (B, n_e)
            active = (refrac <= 0).float()
            v = self.decay_v * (v - cfg.v_rest) + cfg.v_rest + I * active
            # 측면 억제: 이전 스텝에 발화한 다른 뉴런 수 x inh
            others = s_prev.sum(1, keepdim=True) - s_prev
            v = v - cfg.inh * others * active
            s = ((v >= cfg.v_thresh + self.theta) & (refrac <= 0)).float()
            v = torch.where(s.bool(), torch.full_like(v, cfg.v_reset), v)
            refrac = torch.where(s.bool(), torch.full_like(refrac, float(cfg.refrac)), refrac - cfg.dt)
            # 흔적 (발화 시 1 로 설정)
            x_pre = torch.maximum(x_pre * self.decay_tr, s_in)
            x_post = torch.maximum(x_post * self.decay_tr, s)
            if learn:
                # LTD: pre 발화 x post 흔적 / LTP: post 발화 x pre 흔적, 배치 평균
                dW = cfg.nu_post * (x_pre.t() @ s) - cfg.nu_pre * (s_in.t() @ x_post)
                if cfg.reduction == "mean":
                    dW = dW / B
                self.W.add_(dW).clamp_(0.0, cfg.wmax)
                self.theta = self.theta * self.decay_theta + cfg.theta_plus * (s.mean(0) if cfg.reduction == "mean" else s.sum(0))
            counts += s
            s_prev = s
            # 스파이크 수는 텐서로 누적하고 마지막에 한 번만 .item()
            if _ == 0:
                in_acc = s_in.sum()
                ex_acc = s.sum()
            else:
                in_acc = in_acc + s_in.sum()
                ex_acc = ex_acc + s.sum()
        if learn:
            self.normalize()
        if ops is not None:
            ops.input_spikes += float(in_acc.item())
            ops.exc_spikes += float(ex_acc.item())
            ops.steps += steps * B
            ops.images += B
        return counts

    @torch.no_grad()
    def weight_stats(self) -> Dict[str, float]:
        w = self.W.flatten()
        wmax = self.cfg.wmax
        sorted_w, _ = torch.sort(w)
        n = w.numel()
        idx = torch.arange(1, n + 1, device=w.device, dtype=torch.float32)
        gini = float(((2 * idx - n - 1) * sorted_w).sum() / (n * sorted_w.sum() + 1e-12))
        return {"frac_below_1pct": float((w < 0.01 * wmax).float().mean()),
                "frac_below_5pct": float((w < 0.05 * wmax).float().mean()),
                "frac_below_10pct": float((w < 0.10 * wmax).float().mean()),
                "frac_above_50pct": float((w > 0.50 * wmax).float().mean()),
                "gini": gini, "mean": float(w.mean()), "theta_mean": float(self.theta.mean())}


@torch.no_grad()
def assign_labels(counts: torch.Tensor, labels: torch.Tensor, n_classes: int = 10) -> torch.Tensor:
    """뉴런별 라벨 = 클래스별 평균 발화가 가장 큰 클래스 (Diehl & Cook 방식)."""
    rates = torch.zeros(n_classes, counts.shape[1], device=counts.device)
    for c in range(n_classes):
        m = labels == c
        if m.any():
            rates[c] = counts[m].mean(0)
    return rates.argmax(0)                                                   # (n_e,)


@torch.no_grad()
def predict(counts: torch.Tensor, assignments: torch.Tensor, n_classes: int = 10) -> torch.Tensor:
    """클래스 = 그 클래스에 배정된 뉴런들의 평균 발화가 가장 큰 클래스."""
    B = counts.shape[0]
    scores = torch.zeros(B, n_classes, device=counts.device)
    for c in range(n_classes):
        m = assignments == c
        if m.any():
            scores[:, c] = counts[:, m].mean(1)
    return scores.argmax(1)
