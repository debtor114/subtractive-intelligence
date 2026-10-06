# -*- coding: utf-8 -*-
"""H1: 배포 뒤 기울기 없는 '결산 가지치기'.

밀집 MLP 를 역전파로 학습(한 번, 비싼 구조 만들기) → 배포: 라벨 없는 스트림(훈련 집합 1 에폭)을 흘리며
뉴런 단위 흔적을 적립 → 결산 때 기준별로 가지치기 → 시험 정확도. 미세조정 없음. 층별 배분은 모든 기준이 같다
(논문 1/p2 의 학습 중 가지치기 마스크가 남긴 층별 밀도) — 기준은 '어느 연결' 만 정한다.

기준(critera):
  magnitude     |w|
  rank1_pre     |w_ij| * t_in_j                 보내는 뉴런 흐름 (Wanda 식, 저장 O(뉴런))
  rank1_prepost |w_ij| * t_in_j * t_out_i       보내는 x 받는 흐름
  rank1_now     t_in_j * t_out_i                가중치 없는 제안 공식 (사실상 뉴런 가지치기)
  conn_drive    |w_ij| * E[1(y_i>0)|x_j|]       연결 단위 흔적 (저장 O(연결), core.pruning.ActivityTracker)
  rank1_reward  |w_ij| * t_in_j * f(r_i)        r_i = 받는 뉴런 활동과 방송 R=-손실 의 상관 흔적 (오라클, 라벨 사용)
  rank1_conf    같은 것, R = 최대 소프트맥스 확률 (라벨 없음)
  random
결산: oneshot (D 로 한 번) / gradual (4 회 세제곱 스케줄, 결산마다 스트림 재적립; 1%·0.5% 만).
결과 JSON: results/p3/h1/seed<k>.json
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.masked_layers import apply_masks, convert_to_masked, masked_modules   # noqa: E402
from core.pruning import ActivityTracker, _keep_topk                           # noqa: E402
from experiments.p3.common import (RES_P3, alive_hidden_neurons, build_company, count_active, evaluate,  # noqa: E402
                                   get_mnist, mask_path)
from utils.seed import set_seed                                                  # noqa: E402
from utils.tensor_data import TensorBatches                                      # noqa: E402

CRITERIA = ("magnitude", "rank1_pre", "rank1_prepost", "rank1_now", "conn_drive", "rank1_reward", "rank1_conf", "random")
DENSITIES = (0.02, 0.01, 0.005)
GRADUAL_DENSITIES = (0.01, 0.005)
STREAM_BS = 500
H1_DIR = os.path.join(RES_P3, "h1")


def train_dense(seed: int, device, log=print, epochs: int = 15, lr: float = 1e-3) -> nn.Module:
    """밀집 MLP 를 역전파로 학습 (캐시: results/p3/h1/dense_s<k>.pt)."""
    os.makedirs(H1_DIR, exist_ok=True)
    cache = os.path.join(H1_DIR, f"dense_s{seed}.pt")
    model, _ = build_company("dense", 1.0, seed, device)
    if os.path.exists(cache):
        model.load_state_dict(torch.load(cache, map_location=device))
        return model
    x_tr, y_tr, x_te, y_te = get_mnist(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_tr.shape[0]
    t0 = time.time()
    for ep in range(epochs):
        for idx in TensorBatches(torch.arange(n, device=device), 128):
            loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if (ep + 1) % 5 == 0:
            log(f"  [h1 dense s{seed}] epoch {ep + 1}/{epochs} acc {evaluate(model, x_te, y_te):.4f} ({time.time() - t0:.0f}s)")
    torch.save(model.state_dict(), cache)
    return model


def per_layer_targets(density: float, seed: int, names: List[str]) -> Dict[str, float]:
    """층별 배분: p2 학습 중 가지치기 마스크(0.01, 0.005)의 층별 밀도. 없으면 균일."""
    p = mask_path(density, seed)
    if os.path.exists(p):
        masks = torch.load(p, map_location="cpu")
        return {n: float(masks[n].float().mean().item()) for n in names if n in masks}
    return {n: density for n in names}


class NeuronTraces:
    """Linear 마다 입력 뉴런 흔적 t_in (EMA |x_j|), 출력 뉴런 흔적 t_out (EMA |post_i|),
    방송 상관 흔적 r (EMA cov(post_i, R)) 를 유지. 저장은 뉴런 수에 비례."""

    def __init__(self, model: nn.Module, momentum: float = 0.99):
        self.m = momentum
        self.t_in: Dict[str, torch.Tensor] = {}
        self.t_out: Dict[str, torch.Tensor] = {}
        self.r_loss: Dict[str, torch.Tensor] = {}
        self.r_conf: Dict[str, torch.Tensor] = {}
        self.R: Dict[str, torch.Tensor] = {}
        self.handles = []
        mods = masked_modules(model)
        for i, (name, mod) in enumerate(mods):
            dev = mod.weight.device
            self.t_in[name] = torch.zeros(mod.in_features, device=dev)
            self.t_out[name] = torch.zeros(mod.out_features, device=dev)
            self.r_loss[name] = torch.zeros(mod.out_features, device=dev)
            self.r_conf[name] = torch.zeros(mod.out_features, device=dev)
            self.handles.append(mod.register_forward_hook(self._hook(name, last=(i == len(mods) - 1))))
        self._pending: Dict[str, tuple] = {}

    def _hook(self, name, last):
        def hook(m, inp, out):
            with torch.no_grad():
                x = inp[0].detach().float()
                post = out.detach().float() if last else torch.relu(out.detach().float())
                self._pending[name] = (x.abs().mean(0), post.abs().mean(0), post)
        return hook

    @torch.no_grad()
    def commit(self, R_loss: torch.Tensor, R_conf: torch.Tensor) -> None:
        """배치 끝에 호출: 흔적 갱신. R_* 는 표본별 방송 (B,)."""
        rl = R_loss - R_loss.mean()
        rc = R_conf - R_conf.mean()
        for name, (tin, tout, post) in self._pending.items():
            pc = post - post.mean(0, keepdim=True)
            self.t_in[name].mul_(self.m).add_(tin * (1 - self.m))
            self.t_out[name].mul_(self.m).add_(tout * (1 - self.m))
            self.r_loss[name].mul_(self.m).add_((pc * rl[:, None]).mean(0) * (1 - self.m))
            self.r_conf[name].mul_(self.m).add_((pc * rc[:, None]).mean(0) * (1 - self.m))
        self._pending = {}

    def remove(self):
        for h in self.handles:
            h.remove()


def stream(model: nn.Module, x: torch.Tensor, y: torch.Tensor, traces: NeuronTraces, tracker: ActivityTracker,
           device, bs: int = STREAM_BS) -> None:
    """라벨 없는 배포 스트림 1 에폭 (정답은 오라클 방송 R=-손실 에만 쓴다). 역전파 없음.
    ActivityTracker 훅은 grad 가 켜져 있어야 기록하므로 enable_grad 안에서 forward 만 한다."""
    model.eval()
    with torch.enable_grad():
        for idx in TensorBatches(torch.arange(x.shape[0], device=device), bs):
            logits = model(x[idx])
            with torch.no_grad():
                R_loss = -F.cross_entropy(logits, y[idx], reduction="none")
                R_conf = torch.softmax(logits, 1).max(1).values
            traces.commit(R_loss, R_conf)
    model.train()


def _rank_factor(r: torch.Tensor) -> torch.Tensor:
    """상관 흔적을 순위(0,1] 로 바꿔 0.5~1.5 배율로."""
    order = r.argsort()
    rank = torch.empty_like(r)
    rank[order] = torch.arange(1, r.numel() + 1, device=r.device, dtype=r.dtype)
    return 0.5 + rank / r.numel()


@torch.no_grad()
def scores_for(model: nn.Module, crit: str, traces: NeuronTraces, tracker: ActivityTracker, gen: torch.Generator) -> Dict[str, torch.Tensor]:
    out = {}
    for name, m in masked_modules(model):
        w = m.weight.abs()
        tin, tout = traces.t_in[name], traces.t_out[name]
        if crit == "magnitude":
            s = w
        elif crit == "rank1_pre":
            s = w * tin[None, :]
        elif crit == "rank1_prepost":
            s = w * tin[None, :] * tout[:, None]
        elif crit == "rank1_now":
            s = (tin[None, :] * tout[:, None]).expand_as(w).clone()
        elif crit == "conn_drive":
            s = w * tracker.drive[name]
        elif crit == "rank1_reward":
            s = w * tin[None, :] * _rank_factor(traces.r_loss[name])[:, None]
        elif crit == "rank1_conf":
            s = w * tin[None, :] * _rank_factor(traces.r_conf[name])[:, None]
        elif crit == "random":
            s = torch.rand(w.shape, generator=gen, device="cpu").to(w.device)
        else:
            raise KeyError(crit)
        out[name] = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    return out


@torch.no_grad()
def prune_layerwise(model: nn.Module, scores: Dict[str, torch.Tensor], targets: Dict[str, float]) -> None:
    for name, m in masked_modules(model):
        k = int(round(targets[name] * m.weight_mask.numel()))
        _keep_topk(m, scores[name], k)
    apply_masks(model)


def cubic_schedule(final: float, n: int = 4) -> List[float]:
    return [final + (1.0 - final) * (1.0 - k / n) ** 3 for k in range(1, n + 1)]


def fresh_masked(dense_state: dict, seed: int, device) -> nn.Module:
    model, _ = build_company("dense", 1.0, seed, device)
    model.load_state_dict(dense_state)
    model = convert_to_masked(model).to(device)
    return model


def run_h1(cfg: dict, log=print) -> dict:
    seed = int(cfg["seed"])
    device = torch.device("cuda")
    x_tr, y_tr, x_te, y_te = get_mnist(device)
    dense = train_dense(seed, device, log)
    dense_acc = evaluate(dense, x_te, y_te)
    dense_state = copy.deepcopy(dense.state_dict())
    names = [n for n, _ in masked_modules(convert_to_masked(copy.deepcopy(dense)))]
    log(f"  [h1 s{seed}] dense acc {dense_acc:.4f}")
    t0 = time.time()
    res = {"seed": seed, "dense_acc": dense_acc, "oneshot": {}, "gradual": {}, "allocation": {}}
    gen = torch.Generator(device="cpu").manual_seed(seed + 4242)

    # one-shot: 스트림 1 회(밀집망) 의 흔적을 모든 기준이 공유
    model = fresh_masked(dense_state, seed, device)
    traces, tracker = NeuronTraces(model), ActivityTracker(model, momentum=0.99)
    set_seed(seed + 11)
    stream(model, x_tr, y_tr, traces, tracker, device)
    base_state = copy.deepcopy(model.state_dict())
    for D in DENSITIES:
        targets = per_layer_targets(D, seed, names)
        res["allocation"][f"{D:g}"] = targets
        for crit in CRITERIA:
            model.load_state_dict(base_state)
            for _, m in masked_modules(model):
                m.weight_mask.fill_(1.0)
            prune_layerwise(model, scores_for(model, crit, traces, tracker, gen), targets)
            acc = evaluate(model, x_te, y_te)
            res["oneshot"].setdefault(crit, {})[f"{D:g}"] = {"acc": acc, "active": count_active(model),
                                                              "alive_hidden": alive_hidden_neurons(model)}
        log(f"  [h1 s{seed}] one-shot d={D:g}: " + " ".join(f"{c}={res['oneshot'][c][f'{D:g}']['acc']:.3f}" for c in CRITERIA)
            + f" ({time.time() - t0:.0f}s)")
    traces.remove()
    tracker.remove()

    # gradual: 결산 4 회, 결산마다 가지치기된 망으로 스트림 재적립
    for D in GRADUAL_DENSITIES:
        final_targets = per_layer_targets(D, seed, names)
        for crit in CRITERIA:
            model = fresh_masked(dense_state, seed, device)
            traj = []
            for d_k in cubic_schedule(D):
                frac = (d_k - D) / (1.0 - D)                      # 1 -> 0
                targets = {n: final_targets[n] + (1.0 - final_targets[n]) * frac for n in names}
                traces, tracker = NeuronTraces(model), ActivityTracker(model, momentum=0.99)
                set_seed(seed + 11)
                stream(model, x_tr, y_tr, traces, tracker, device)
                prune_layerwise(model, scores_for(model, crit, traces, tracker, gen), targets)
                traces.remove()
                tracker.remove()
                traj.append({"density": sum(targets[n] * dict(masked_modules(model))[n].weight_mask.numel() for n in names)
                             / sum(dict(masked_modules(model))[n].weight_mask.numel() for n in names),
                             "acc": evaluate(model, x_te, y_te)})
            res["gradual"].setdefault(crit, {})[f"{D:g}"] = {"acc": traj[-1]["acc"], "traj": traj,
                                                              "active": count_active(model), "alive_hidden": alive_hidden_neurons(model)}
        log(f"  [h1 s{seed}] gradual d={D:g}: " + " ".join(f"{c}={res['gradual'][c][f'{D:g}']['acc']:.3f}" for c in CRITERIA)
            + f" ({time.time() - t0:.0f}s)")
    res["elapsed_s"] = time.time() - t0
    out = cfg.get("out") or os.path.join(H1_DIR, f"seed{seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


if __name__ == "__main__":
    run_h1({"seed": int(sys.argv[1]) if len(sys.argv) > 1 else 0})
