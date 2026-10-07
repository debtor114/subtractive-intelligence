# -*- coding: utf-8 -*-
"""p3 학습기 사다리: 역전파(bp) > 직접 피드백 정렬(dfa) > 순방향 기울기(fg) ~ 가중치 섭동(wp) > 노드 섭동(np).

모든 학습기는 '추정한 기울기' 를 p.grad 에 넣고 같은 Adam 으로 한 걸음 간다 (추정기 크기 차이는 Adam 이 정규화).
마스크된 층은 섭동 방향과 기울기를 마스크 안으로 제한한다 — 죽은 연결은 흔들지도 갱신하지도 않는다.
  bp : 참 기울기 (감사팀)
  dfa: 출력 오차 벡터 e 를 고정 무작위 행렬 B_l 로 각 은닉층에 투영 (Nøkland 2016). 출력층은 정확한 국소 기울기.
  fg : 무작위 방향 v 의 방향 미분 (∇L·v) v (Baydin 등 2022, torch.func.jvp). 방향은 배치당 하나.
  wp : 양측 유한차분 (L(w+εv)-L(w-εv))/(2ε) · v (MeZO 식). 방송되는 숫자는 손실 차이 하나.
  np : 뉴런 단위 섭동. 표본마다 은닉·출력 선활성에 ξ~N(0,σ²) 를 더하고 ΔL·ξ/σ² 를 신용으로 (Fiete & Seung 2006).
SNR 탐침(probe): 같은 배치에서 참 기울기와 추정기 K 개를 비교 — 코사인·신호/잡음비.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp

from core.masked_layers import apply_masks
from experiments.p3.common import eff_w, linears, wmask


def manual_forward(model: nn.Module, x: torch.Tensor):
    """pres[i] = i 번째 Linear 의 선활성, posts[i] = 그 층의 입력 (posts[0] = 평탄화 입력, posts[-1] = 로짓)."""
    h = torch.flatten(x, 1)
    pres, posts = [], [h]
    L = linears(model)
    for i, m in enumerate(L):
        a = F.linear(h, eff_w(m), m.bias)
        pres.append(a)
        h = torch.relu(a) if i < len(L) - 1 else a
        posts.append(h)
    return pres, posts


def param_masks(model: nn.Module) -> Dict[int, torch.Tensor]:
    """id(param) -> mask. 가중치는 weight_mask, 편향은 '들어오는 연결이 하나라도 있는 뉴런' 만 (검토 반영 2026-10-07:
    죽은 뉴런의 편향까지 흔들고 갱신하면 pruned 의 섭동 차원이 dense_small 보다 10~17% 커진다)."""
    out = {}
    for m in linears(model):
        mk = wmask(m)
        if mk is not None:
            out[id(m.weight)] = mk
            if m.bias is not None:
                out[id(m.bias)] = (mk != 0).any(dim=1).to(mk.dtype)
    return out


def set_grads(model: nn.Module, grads: Dict[nn.Parameter, torch.Tensor], masks: Dict[int, torch.Tensor]) -> None:
    for p in model.parameters():
        p.grad = None
    for p, g in grads.items():
        mk = masks.get(id(p))
        p.grad = g * mk if mk is not None else g


def flat_grads(model: nn.Module) -> torch.Tensor:
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in model.parameters()])


class Learner:
    name = "base"
    stochastic = True

    def __init__(self, model: nn.Module, lr: float, cfg: dict):
        self.model, self.cfg = model, cfg
        self.masks = param_masks(model)
        # SGD+momentum: 잡음 추정기는 신호가 선형, 잡음이 제곱근으로 쌓이게 두어야 한다. Adam 은 잡음 방향도 lr 크기로
        # 밀어 초기화를 무작위 보행으로 지운다 (연기 시험 2026-10-07: fg/wp 가 우연 수준에 고정). 학습률은 학습기·회사마다 선택.
        if cfg.get("opt", "sgd") == "adam":
            self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            self.opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=float(cfg.get("momentum", 0.9)))

    def estimate(self, xb: torch.Tensor, yb: torch.Tensor) -> float:
        raise NotImplementedError

    def step(self, xb: torch.Tensor, yb: torch.Tensor) -> float:
        loss = self.estimate(xb, yb)
        self.opt.step()
        apply_masks(self.model)
        return loss


class Backprop(Learner):
    name = "bp"
    stochastic = False

    def estimate(self, xb, yb):
        self.model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.model(xb), yb)
        loss.backward()
        for m in linears(self.model):
            mk = wmask(m)
            if mk is not None and m.weight.grad is not None:
                m.weight.grad.mul_(mk)
        return float(loss.item())


class DFA(Learner):
    name = "dfa"
    stochastic = False

    def __init__(self, model, lr, cfg):
        super().__init__(model, lr, cfg)
        L = linears(model)
        n_out = L[-1].out_features
        g = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 0)) + 777)
        dev = L[-1].weight.device
        self.B = [(torch.randn(n_out, m.out_features, generator=g) / math.sqrt(n_out)).to(dev) for m in L[:-1]]

    @torch.no_grad()
    def estimate(self, xb, yb):
        pres, posts = manual_forward(self.model, xb)
        logits = posts[-1]
        B = xb.shape[0]
        e = (torch.softmax(logits, 1) - F.one_hot(yb, logits.shape[1]).float()) / B      # dL/dlogits (평균 CE)
        L = linears(self.model)
        grads = {L[-1].weight: e.t() @ posts[-2], L[-1].bias: e.sum(0)}
        for i, m in enumerate(L[:-1]):
            d = (e @ self.B[i]) * (pres[i] > 0).float()
            grads[m.weight] = d.t() @ posts[i]
            grads[m.bias] = d.sum(0)
        set_grads(self.model, grads, self.masks)
        return float(F.cross_entropy(logits, yb).item())


class ForwardGradient(Learner):
    name = "fg"

    def __init__(self, model, lr, cfg):
        super().__init__(model, lr, cfg)
        self.K = int(cfg.get("dirs", 8))
        self.names = [n for n, _ in model.named_parameters()]
        self.buffers = dict(model.named_buffers())
        self.name_mask = {}
        for mod_name, m in model.named_modules():
            if isinstance(m, nn.Linear) and wmask(m) is not None:
                self.name_mask[mod_name + ".weight"] = wmask(m)
                if m.bias is not None:
                    self.name_mask[mod_name + ".bias"] = self.masks[id(m.bias)]

    def estimate(self, xb, yb):
        params = {n: p.detach() for n, p in self.model.named_parameters()}

        def loss_fn(p):
            return F.cross_entropy(functional_call(self.model, (p, self.buffers), (xb,)), yb)

        acc = {n: torch.zeros_like(p) for n, p in params.items()}
        loss = None
        for _ in range(self.K):
            v = {n: torch.randn_like(p) for n, p in params.items()}
            for n, mk in self.name_mask.items():
                v[n].mul_(mk)
            loss, d = jvp(loss_fn, (params,), (v,))
            for n in acc:
                acc[n].add_(d * v[n])
        plist = dict(self.model.named_parameters())
        set_grads(self.model, {plist[n]: acc[n] / self.K for n in acc}, self.masks)
        return float(loss.item())


class WeightPerturbation(Learner):
    name = "wp"

    def __init__(self, model, lr, cfg):
        super().__init__(model, lr, cfg)
        self.eps = float(cfg.get("eps", 1e-3))
        self.K = int(cfg.get("dirs", 8))

    @torch.no_grad()
    def estimate(self, xb, yb):
        params = list(self.model.parameters())
        acc = [torch.zeros_like(p) for p in params]
        lp = lm = None
        for _ in range(self.K):
            v = [torch.randn_like(p) for p in params]
            for p, vi in zip(params, v):
                mk = self.masks.get(id(p))
                if mk is not None:
                    vi.mul_(mk)
            for p, vi in zip(params, v):
                p.add_(self.eps * vi)
            lp = F.cross_entropy(self.model(xb), yb)
            for p, vi in zip(params, v):
                p.sub_(2.0 * self.eps * vi)
            lm = F.cross_entropy(self.model(xb), yb)
            for p, vi in zip(params, v):
                p.add_(self.eps * vi)
            d = (lp - lm) / (2.0 * self.eps)
            for a, vi in zip(acc, v):
                a.add_(d * vi)
        set_grads(self.model, {p: a / self.K for p, a in zip(params, acc)}, self.masks)
        return float(0.5 * (lp + lm).item())


class NodePerturbation(Learner):
    name = "np"

    def __init__(self, model, lr, cfg):
        super().__init__(model, lr, cfg)
        self.sigma = float(cfg.get("sigma", 0.1))

    @torch.no_grad()
    def estimate(self, xb, yb):
        L = linears(self.model)
        B = xb.shape[0]
        _, posts0 = manual_forward(self.model, xb)
        l0 = F.cross_entropy(posts0[-1], yb, reduction="none")
        h = torch.flatten(xb, 1)
        posts, xis = [h], []
        for i, m in enumerate(L):
            a = F.linear(h, eff_w(m), m.bias)
            xi = torch.randn_like(a) * self.sigma
            bm = self.masks.get(id(m.bias)) if m.bias is not None else None
            if bm is not None:
                xi = xi * bm[None, :]                            # 들어오는 연결이 없는 뉴런은 흔들지 않는다
            a = a + xi
            xis.append(xi)
            h = torch.relu(a) if i < len(L) - 1 else a
            posts.append(h)
        l1 = F.cross_entropy(posts[-1], yb, reduction="none")
        dl = (l1 - l0) / (self.sigma ** 2)                       # (B,) 표본별 방송
        grads = {}
        for i, m in enumerate(L):
            s = dl[:, None] * xis[i]                             # (B, out)
            grads[m.weight] = s.t() @ posts[i] / B
            grads[m.bias] = s.sum(0) / B
        set_grads(self.model, grads, self.masks)
        return float(l0.mean().item())


LEARNERS = {c.name: c for c in (Backprop, DFA, ForwardGradient, WeightPerturbation, NodePerturbation)}


def make_learner(name: str, model: nn.Module, lr: float, cfg: dict) -> Learner:
    return LEARNERS[name](model, lr, cfg)


def probe(learner: Learner, xb: torch.Tensor, yb: torch.Tensor, K: int = 8) -> dict:
    """같은 배치에서 참 기울기 g 와 학습기 추정 K 개를 비교. 훈련 상태(p.grad)는 호출 뒤 비운다."""
    model = learner.model
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    for m in linears(model):
        mk = wmask(m)
        if mk is not None and m.weight.grad is not None:
            m.weight.grad.mul_(mk)
    g = flat_grads(model).clone()
    n = K if learner.stochastic else 1
    ests = []
    for _ in range(n):
        learner.estimate(xb, yb)
        ests.append(flat_grads(model).clone())
    E = torch.stack(ests)
    mean = E.mean(0)
    out = {
        "loss": float(loss.item()),
        "cos_each": float(F.cosine_similarity(E, g[None], dim=1).mean().item()),
        "cos_mean": float(F.cosine_similarity(mean[None], g[None], dim=1).item()),
        "norm_ratio": float((mean.norm() / (g.norm() + 1e-12)).item()),
        "K": n,
    }
    if n > 1:
        noise = float(((E - mean) ** 2).sum(1).mean().item())
        signal = float((mean ** 2).sum().item())
        out["snr"] = signal / (noise + 1e-30)
    model.zero_grad(set_to_none=True)
    return out
