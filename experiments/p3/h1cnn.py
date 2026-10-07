# -*- coding: utf-8 -*-
"""H1 CNN: 배포 결산 가지치기의 허브 배분이 합성곱 망에서도 성립하는가 (CIFAR-10 SmallCNN 64-128-256, 논문 1 과 같은 망).

왜: 논문 1 에서 활동 기반 규칙(구동)은 CNN 에서 뉴런별 정규화(허브의 반대) 없이 채널을 죽여 무너졌다. MLP H1c 의 결론
("극단 희소에는 뉴런별 비균등 예산 = 허브 배분이 필요하고, 받는 뉴런 활동이 그 배분을 알려 준다")이 CNN 에서도 맞는지가
논문 2 1 장의 일반성을 정한다.

설정: 논문 1 레시피로 밀집 CNN 학습(20 에폭, 시드 3, 캐시) → 배포 스트림(훈련 집합 1 에폭, 증강·라벨 없음)에서 흔적 →
한 번에 자름 → BN 재보정(스트림 1 만 장, 라벨·기울기 없음, 모든 기준 동일; 재보정 전 정확도도 기록) → 시험 정확도.
'행' = 출력 채널(합성곱)·출력 뉴런(완전연결). 입력 통계는 입력 채널 단위(커널 위치 공통).
배분: U = 첫 합성곱·출력 머리 유지, 나머지 6 층 균일 D ∈ {50..1%} / L = 논문 1 학습 중 가지치기 마스크의 층별 밀도(10·3·1%, 전 층).
기준: magnitude(_row), wanda_row/_layer (abs(w)·‖X_c‖₂), prepost_layer/_row (abs(w)·평균|x_c|·평균|y_o|, y 는 BN·ReLU 뒤),
conn_drive·coincidence_w (core.pruning.ActivityTracker, 합성곱 출력은 BN 앞), ria_row/_layer (Zhang 등 2024: 행합·열합 정규화 + ‖X‖₂^0.5,
허브의 반대 방향 대조), rates_only, random.
결과: results/p3/h1cnn/seed<k>.json, summarize() -> summary.json, tables_h1cnn.md, fig_h1cnn_sweep.png
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baselines.cnn import SmallCNN                                             # noqa: E402
from core.masked_layers import apply_masks, convert_to_masked, masked_modules  # noqa: E402
from core.pruning import ActivityTracker, _keep_topk                           # noqa: E402
from experiments.p3.common import RES_P3                                       # noqa: E402
from experiments.p3.h1c import gini, spearman                                  # noqa: E402
from experiments.p3.h2 import cosine_lr                                        # noqa: E402
from utils.cifar_gpu import augment, load_cifar10_gpu, normalize               # noqa: E402
from utils.seed import set_seed                                                # noqa: E402
from utils.tensor_data import TensorBatches                                    # noqa: E402

OUT_DIR = os.path.join(RES_P3, "h1cnn")
DENSITIES = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
TRANSPLANT_D = [0.1, 0.05, 0.02, 0.01]
LEARNED_D = [0.1, 0.03, 0.01]
ORDER = ["features.0.0", "features.1.0", "features.3.0", "features.4.0", "features.6.0", "features.7.0", "classifier.1", "classifier.4"]
POST = {"features.0.0": "features.0.2", "features.1.0": "features.1.2", "features.3.0": "features.3.2", "features.4.0": "features.4.2",
        "features.6.0": "features.6.2", "features.7.0": "features.7.2", "classifier.1": "classifier.2", "classifier.4": None}
FIRST, HEAD = "features.0.0", "classifier.4"
CRITS = {  # 이름: (점수, 입도)
    "magnitude": ("magnitude", "layer"), "magnitude_row": ("magnitude", "row"),
    "wanda_row": ("wanda", "row"), "wanda_layer": ("wanda", "layer"),
    "prepost_layer": ("prepost", "layer"), "prepost_row": ("prepost", "row"),
    "conn_drive": ("conn_drive", "layer"), "coincidence_w": ("coincidence_w", "layer"),
    "ria_row": ("ria", "row"), "ria_layer": ("ria", "layer"),
    "rates_only": ("rates_only", "layer"), "random": ("random", "layer"),
}
TRANSPLANTS = {
    "deg(prepost)+sel(wanda)": ("prepost_layer", "wanda"),
    "deg(prepost)+sel(magnitude)": ("prepost_layer", "magnitude"),
    "deg(magnitude)+sel(prepost)": ("magnitude", "prepost"),
    "deg(prepost)+sel(prepost) [항등 점검]": ("prepost_layer", "prepost"),
}
_DATA: Dict[str, tuple] = {}


def get_cifar(device):
    if "c" not in _DATA:
        _DATA["c"] = load_cifar10_gpu(device=device)
    return _DATA["c"]


def build(device) -> nn.Module:
    return SmallCNN(channels=(64, 128, 256), fc=256).to(device)


@torch.no_grad()
def evaluate(model, x_te, y_te, mean, std, bs: int = 1000) -> float:
    was = model.training
    model.eval()
    correct = 0
    for s in range(0, x_te.shape[0], bs):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(normalize(x_te[s:s + bs], mean, std))
        correct += int((logits.argmax(1) == y_te[s:s + bs]).sum().item())
    model.train(was)
    return correct / x_te.shape[0]


def train_dense_cnn(seed: int, device, log=print, epochs: int = 20, lr: float = 0.05, wd: float = 5e-4, bs: int = 128) -> nn.Module:
    """논문 1 CNN 레시피 (SGD 네스테로프 0.9, lr 0.05, wd 5e-4 (BN·편향 제외), 1 에폭 워밍업 코사인, 증강, AMP). 캐시."""
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = os.path.join(OUT_DIR, f"dense_s{seed}.pt")
    set_seed(seed)
    model = build(device)
    if os.path.exists(cache):
        model.load_state_dict(torch.load(cache, map_location=device))
        return model
    (x_tr, y_tr), (x_te, y_te), (mean, std) = get_cifar(device)
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                          lr=lr, momentum=0.9, nesterov=True)
    scaler = torch.amp.GradScaler("cuda")
    n = x_tr.shape[0]
    spe = math.ceil(n / bs)
    total = epochs * spe
    step, t0 = 0, time.time()
    model.train()
    for ep in range(epochs):
        for idx in TensorBatches(torch.arange(n, device=device), bs):
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, spe, lr)
            xb, yb = normalize(augment(x_tr[idx]), mean, std), y_tr[idx]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
        if (ep + 1) % 5 == 0:
            log(f"  [h1cnn dense s{seed}] epoch {ep + 1}/{epochs} acc {evaluate(model, x_te, y_te, mean, std):.4f} ({time.time() - t0:.0f}s)")
    torch.save(model.state_dict(), cache)
    return model


def fresh_masked(dense_state, device) -> nn.Module:
    m = build(device)
    m.load_state_dict(dense_state)
    return convert_to_masked(m).to(device)


class CNNTraces:
    """층마다 입력 채널(특징) 단위 평균|x|·제곱합, 출력 채널(뉴런) 단위 평균|y| (BN·ReLU 뒤; 마지막 층은 |로짓|). 스트림 전체 평균."""

    def __init__(self, model: nn.Module):
        self.sum_in, self.sq_in, self.sum_out, self.n_in, self.n_out = {}, {}, {}, {}, {}
        self.handles = []
        mods = dict(model.named_modules())
        for name, m in masked_modules(model):
            cin = m.in_channels if isinstance(m, nn.Conv2d) else m.in_features
            cout = m.out_channels if isinstance(m, nn.Conv2d) else m.out_features
            dev = m.weight.device
            self.sum_in[name] = torch.zeros(cin, device=dev)
            self.sq_in[name] = torch.zeros(cin, device=dev)
            self.sum_out[name] = torch.zeros(cout, device=dev)
            self.n_in[name] = 0
            self.n_out[name] = 0
            self.handles.append(m.register_forward_hook(self._in_hook(name)))
            target = mods[POST[name]] if POST[name] else m
            self.handles.append(target.register_forward_hook(self._out_hook(name)))

    def _in_hook(self, name):
        def hook(mod, inp, out):
            with torch.no_grad():
                x = inp[0].detach().float()
                if x.dim() == 4:
                    self.sum_in[name].add_(x.abs().mean((2, 3)).sum(0))
                    self.sq_in[name].add_((x * x).sum((0, 2, 3)))
                else:
                    self.sum_in[name].add_(x.abs().sum(0))
                    self.sq_in[name].add_((x * x).sum(0))
                self.n_in[name] += x.shape[0]
        return hook

    def _out_hook(self, name):
        def hook(mod, inp, out):
            with torch.no_grad():
                y = out.detach().float()
                self.sum_out[name].add_(y.abs().mean((2, 3)).sum(0) if y.dim() == 4 else y.abs().sum(0))
                self.n_out[name] += y.shape[0]
        return hook

    def t_in(self, name):
        return self.sum_in[name] / max(self.n_in[name], 1)

    def t_out(self, name):
        return self.sum_out[name] / max(self.n_out[name], 1)

    def remove(self):
        for h in self.handles:
            h.remove()


def stream(model, x_tr, mean, std, traces, tracker, device, bs: int = 200) -> None:
    """배포 스트림: 훈련 집합 1 에폭, 증강·라벨 없음, eval 모드. ActivityTracker 는 grad 가 켜져야 기록 — backward 는 하지 않는다."""
    model.eval()
    with torch.enable_grad():
        for idx in TensorBatches(torch.arange(x_tr.shape[0], device=device), bs):
            model(normalize(x_tr[idx], mean, std))
    model.train()


def col_stat(m, v: torch.Tensor) -> torch.Tensor:
    """입력 채널 통계 (cin,) 를 2 차원 가중치 (out, cin*kh*kw) 의 열로 펼친다."""
    if isinstance(m, nn.Conv2d):
        return v.repeat_interleave(m.kernel_size[0] * m.kernel_size[1])
    return v


@torch.no_grad()
def scores_2d(model, kind: str, traces: CNNTraces, tracker: ActivityTracker, gen) -> Dict[str, torch.Tensor]:
    out = {}
    for name, m in masked_modules(model):
        w = m.weight.abs().reshape(m.weight.shape[0], -1)
        tin, tout, sq = col_stat(m, traces.t_in(name)), traces.t_out(name), col_stat(m, traces.sq_in[name])
        if kind == "magnitude":
            s = w
        elif kind == "wanda":
            s = w * sq.sqrt()[None, :]
        elif kind == "prepost":
            s = w * tin[None, :] * tout[:, None]
        elif kind == "rates_only":
            s = (tin[None, :] * tout[:, None]).expand_as(w).clone()
        elif kind == "conn_drive":
            s = w * tracker.drive[name].reshape(w.shape)
        elif kind == "coincidence_w":
            s = w * tracker.traces[name].abs().reshape(w.shape)
        elif kind == "ria":
            s = (w / (w.sum(1, keepdim=True) + 1e-12) + w / (w.sum(0, keepdim=True) + 1e-12)) * sq.sqrt().pow(0.5)[None, :]
        elif kind == "random":
            s = torch.rand(w.shape, generator=gen).to(w.device)
        else:
            raise KeyError(kind)
        out[name] = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    return out


@torch.no_grad()
def prune(model, scores: Dict[str, torch.Tensor], targets: Dict[str, float], gran: str) -> None:
    for name, m in masked_modules(model):
        d = float(targets.get(name, 1.0))
        if d >= 1.0:
            continue
        s2 = scores[name]
        if gran == "layer":
            _keep_topk(m, s2.reshape(m.weight.shape), int(round(d * m.weight_mask.numel())))
        else:
            k = max(1, int(round(d * s2.shape[1])))
            idx = torch.topk(s2, k, dim=1).indices
            keep = torch.zeros_like(s2, dtype=torch.bool)
            keep.scatter_(1, idx, True)
            m.weight_mask.copy_(keep.reshape(m.weight.shape).to(m.weight_mask.dtype))
    apply_masks(model)


@torch.no_grad()
def prune_transplant(model, src_masks: Dict[str, torch.Tensor], scores: Dict[str, torch.Tensor]) -> None:
    for name, m in masked_modules(model):
        src = src_masks[name].reshape(m.weight.shape[0], -1)
        k = src.sum(1).long()
        order = torch.argsort(scores[name], dim=1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(order.shape[1], device=order.device).expand_as(order))
        m.weight_mask.copy_((rank < k[:, None]).reshape(m.weight.shape).to(m.weight_mask.dtype))
    apply_masks(model)


@torch.no_grad()
def recalibrate_bn(model, x_recal, mean, std, bs: int = 500) -> None:
    """BN 재보정: 라벨·기울기 없이 배포 스트림 1 만 장으로 이동 평균을 다시 잰다 (누적 평균)."""
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats()
            mod.momentum = None
    model.train()
    for s in range(0, x_recal.shape[0], bs):
        model(normalize(x_recal[s:s + bs], mean, std))
    model.eval()


@torch.no_grad()
def chain_stats(model, traces: CNNTraces) -> dict:
    """살아 있는 은닉 단위(채널·뉴런), 막다른 길 낭비 비율, 층별 입력 수 지니·받는 활동 상관."""
    M = {n: m.weight_mask.bool() for n, m in masked_modules(model)}
    mods = dict(masked_modules(model))
    in_deg = {n: M[n].reshape(M[n].shape[0], -1).sum(1) for n in ORDER}
    out_deg = {}
    for k, name in enumerate(ORDER[:-1]):
        nxt = M[ORDER[k + 1]]
        if name == "features.7.0":
            out_deg[name] = nxt.reshape(nxt.shape[0], 256, -1).sum((0, 2))
        elif nxt.dim() == 4:
            out_deg[name] = nxt.sum((0, 2, 3))
        else:
            out_deg[name] = nxt.sum(0)
    alive = int(sum(((in_deg[n] > 0) & (out_deg[n] > 0)).sum().item() for n in ORDER[:-1]))
    units = int(sum(in_deg[n].numel() for n in ORDER[:-1]))
    wasted, kept = 0, 0
    for k, name in enumerate(ORDER):
        M2 = M[name].reshape(M[name].shape[0], -1)
        kept += int(M2.sum().item())
        bad_rows = (out_deg[name] == 0) if name in out_deg else torch.zeros(M2.shape[0], dtype=torch.bool, device=M2.device)
        if k > 0:
            m = mods[name]
            src_dead = in_deg[ORDER[k - 1]] == 0
            if isinstance(m, nn.Conv2d):
                col_unit = torch.arange(m.in_channels, device=M2.device).repeat_interleave(m.kernel_size[0] * m.kernel_size[1])
            elif name == "classifier.1":
                col_unit = torch.arange(256, device=M2.device).repeat_interleave(M2.shape[1] // 256)
            else:
                col_unit = torch.arange(M2.shape[1], device=M2.device)
            bad_cols = src_dead[col_unit]
        else:
            bad_cols = torch.zeros(M2.shape[1], dtype=torch.bool, device=M2.device)
        wasted += int((M2 & (bad_rows[:, None] | bad_cols[None, :])).sum().item())
    per_layer = {}
    for name in ORDER[1:-1]:
        deg = in_deg[name].float().cpu().numpy()
        per_layer[name] = {"gini_in": gini(deg), "spearman": spearman(deg, traces.t_out(name).cpu().numpy())}
    return {"alive_units": alive, "units": units, "wasted_frac": wasted / max(kept, 1),
            "gini_mean": float(np.mean([v["gini_in"] for v in per_layer.values()])),
            "spearman_mean": float(np.nanmean([v["spearman"] for v in per_layer.values()])), "per_layer": per_layer}


def learned_targets(D: float, seed: int) -> Dict[str, float]:
    p = os.path.join(REPO_ROOT, "results", "core_cifar", f"d{D:g}", "pd_mag_global", f"seed{seed}.json")
    return json.load(open(p, encoding="utf-8"))["layer_densities"]


def run_cnn(seed: int, log=print) -> dict:
    device = torch.device("cuda")
    (x_tr, y_tr), (x_te, y_te), (mean, std) = get_cifar(device)
    dense = train_dense_cnn(seed, device, log)
    dense_state = copy.deepcopy(dense.state_dict())
    dense_acc = evaluate(dense, x_te, y_te, mean, std)
    log(f"  [h1cnn s{seed}] dense acc {dense_acc:.4f}")
    t0 = time.time()
    model = fresh_masked(dense_state, device)
    traces, tracker = CNNTraces(model), ActivityTracker(model, momentum=0.99)
    set_seed(seed + 11)
    stream(model, x_tr, mean, std, traces, tracker, device)
    traces.remove()
    tracker.remove()
    g = torch.Generator(device="cpu").manual_seed(seed + 777)
    recal_idx = torch.randperm(x_tr.shape[0], generator=g)[:10000].to(device)
    x_recal = x_tr[recal_idx]
    gen = torch.Generator(device="cpu").manual_seed(seed + 4242)
    total_w = sum(m.weight_mask.numel() for _, m in masked_modules(model))
    res = {"seed": seed, "dense_acc": dense_acc, "uniform": {}, "learned": {}, "transplant": {}}
    src_masks = {}

    def one(crit_name, targets):
        kind, gran = CRITS[crit_name]
        m = fresh_masked(dense_state, device)
        prune(m, scores_2d(m, kind, traces, tracker, gen), targets, gran)
        stale = evaluate(m, x_te, y_te, mean, std)
        recalibrate_bn(m, x_recal, mean, std)
        acc = evaluate(m, x_te, y_te, mean, std)
        st = chain_stats(m, traces)
        dens = sum(int(mm.weight_mask.sum().item()) for _, mm in masked_modules(m)) / total_w
        return m, {"acc": acc, "acc_stale_bn": stale, "total_density": dens, **{k: v for k, v in st.items() if k != "per_layer"},
                   "per_layer": st["per_layer"]}

    for D in DENSITIES:
        tg = {n: (1.0 if n in (FIRST, HEAD) else D) for n in ORDER}
        for c in CRITS:
            m, r = one(c, tg)
            res["uniform"].setdefault(c, {})[f"{D:g}"] = r
            if D in TRANSPLANT_D and c in ("prepost_layer", "magnitude"):
                src_masks[(c, D)] = {n: mm.weight_mask.clone() for n, mm in masked_modules(m)}
        log(f"  [h1cnn s{seed}] U D={D:g}: " + " ".join(f"{c}={res['uniform'][c][f'{D:g}']['acc']:.3f}" for c in CRITS)
            + f" ({time.time() - t0:.0f}s)")
    for D in TRANSPLANT_D:
        for t, (src, kind) in TRANSPLANTS.items():
            m = fresh_masked(dense_state, device)
            prune_transplant(m, src_masks[(src, D)], scores_2d(m, kind, traces, tracker, gen))
            recalibrate_bn(m, x_recal, mean, std)
            st = chain_stats(m, traces)
            res["transplant"].setdefault(t, {})[f"{D:g}"] = {"acc": evaluate(m, x_te, y_te, mean, std), "alive_units": st["alive_units"],
                                                              "wasted_frac": st["wasted_frac"]}
        log(f"  [h1cnn s{seed}] transplant D={D:g}: " + " ".join(f"{t.split(' ')[0]}={res['transplant'][t][f'{D:g}']['acc']:.3f}" for t in TRANSPLANTS))
    for D in LEARNED_D:
        tg = learned_targets(D, seed)
        for c in CRITS:
            _, r = one(c, tg)
            res["learned"].setdefault(c, {})[f"{D:g}"] = r
        log(f"  [h1cnn s{seed}] L D={D:g}: " + " ".join(f"{c}={res['learned'][c][f'{D:g}']['acc']:.3f}" for c in CRITS))
    res["elapsed_s"] = time.time() - t0
    with open(os.path.join(OUT_DIR, f"seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


LABEL = {"wanda_row": "Wanda (행별, 원본)", "wanda_layer": "Wanda 점수 (층 순위)", "prepost_layer": "보내는×받는 흐름 (층 순위)",
         "prepost_row": "보내는×받는 흐름 (행별)", "magnitude": "크기 (층 순위)", "magnitude_row": "크기 (행별)",
         "conn_drive": "연결 단위 구동 흔적", "coincidence_w": "동시 발화×abs(w)", "ria_row": "RIA (행별)", "ria_layer": "RIA (층 순위)",
         "rates_only": "흐름만 (가중치 없음)", "random": "무작위"}
COLORS = {"wanda_row": "#1f5fbf", "wanda_layer": "#7fa7e6", "prepost_layer": "#d9480f", "prepost_row": "#f4a261",
          "magnitude": "#495057", "magnitude_row": "#adb5bd", "conn_drive": "#2b8a3e", "coincidence_w": "#7048e8",
          "ria_row": "#0c8599", "ria_layer": "#66d9e8", "rates_only": "#e599f7", "random": "#ced4da"}
STYLE = {c: ("--" if c.endswith("_row") else (":" if c in ("rates_only", "random") else "-")) for c in COLORS}


def summarize() -> dict:
    runs = [json.load(open(os.path.join(OUT_DIR, f"seed{k}.json"), encoding="utf-8"))
            for k in range(3) if os.path.exists(os.path.join(OUT_DIR, f"seed{k}.json"))]
    if not runs:
        return {}

    def agg(vals):
        v = [x for x in vals if x is not None]
        return (float(np.mean(v)), float(np.std(v))) if v else (None, None)

    S = {"n": len(runs), "dense_acc": agg([r["dense_acc"] for r in runs])[0], "uniform": {}, "learned": {}, "transplant": {}}
    for sch, Ds in (("uniform", DENSITIES), ("learned", LEARNED_D)):
        for c in CRITS:
            for D in Ds:
                k = f"{D:g}"
                rs = [r[sch][c][k] for r in runs]
                S[sch].setdefault(c, {})[k] = {
                    "acc": agg([x["acc"] for x in rs]), "acc_stale_bn": agg([x["acc_stale_bn"] for x in rs])[0],
                    "alive_units": agg([x["alive_units"] for x in rs])[0], "units": rs[0]["units"],
                    "wasted_frac": agg([x["wasted_frac"] for x in rs])[0], "gini_mean": agg([x["gini_mean"] for x in rs])[0],
                    "spearman_mean": agg([x["spearman_mean"] for x in rs])[0], "total_density": agg([x["total_density"] for x in rs])[0]}
    for t in TRANSPLANTS:
        for D in TRANSPLANT_D:
            k = f"{D:g}"
            S["transplant"].setdefault(t, {})[k] = {"acc": agg([r["transplant"][t][k]["acc"] for r in runs])}
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, ensure_ascii=False, indent=1)

    def pm(a):
        return "-" if a[0] is None else f"{100 * a[0]:.1f}±{100 * a[1]:.1f}"

    L = [f"## H1 CNN 밀도 스윕 (CIFAR-10 SmallCNN, 첫 합성곱·출력 머리 유지, 나머지 6 층 균일 D, BN 재보정, 미세조정 없음; 시험 정확도 %, 시드 {len(runs)}; 밀집 {100 * S['dense_acc']:.1f})", "",
         "| 기준 | " + " | ".join(f"{100 * D:g}%" for D in DENSITIES) + " |", "|---|" + "---|" * len(DENSITIES)]
    for c in CRITS:
        L.append(f"| {LABEL[c]} | " + " | ".join(pm(S['uniform'][c][f'{D:g}']['acc']) for D in DENSITIES) + " |")
    L += ["", "## H1 CNN 허브·경로 지표 (균일 배분; 살아 있는 은닉 단위 / 층 평균 입력 수 지니 / 층 평균 입력 수-받는 활동 상관 / 낭비 연결 %)", "",
          "| 기준 | " + " | ".join(f"{100 * D:g}%" for D in DENSITIES[3:]) + " |", "|---|" + "---|" * len(DENSITIES[3:])]
    for c in ("wanda_row", "wanda_layer", "prepost_layer", "magnitude", "conn_drive", "ria_row", "ria_layer"):
        cells = []
        for D in DENSITIES[3:]:
            h = S["uniform"][c][f"{D:g}"]
            cells.append(f"{h['alive_units']:.0f}/{h['units']} / {h['gini_mean']:.2f} / {h['spearman_mean']:.2f} / {100 * h['wasted_frac']:.1f}")
        L.append(f"| {LABEL[c]} | " + " | ".join(cells) + " |")
    L += ["", "## H1 CNN 차수 이식 (균일 배분, BN 재보정)", "", "| 이식 | " + " | ".join(f"{100 * D:g}%" for D in TRANSPLANT_D) + " |",
          "|---|" + "---|" * len(TRANSPLANT_D)]
    for t in TRANSPLANTS:
        L.append(f"| {t} | " + " | ".join(pm(S['transplant'][t][f'{D:g}']['acc']) for D in TRANSPLANT_D) + " |")
    L += ["", "## H1 CNN 학습 배분 (논문 1 학습 중 가지치기 마스크의 층별 밀도, 전 층; 정확도 % / 낭비 연결 % / 살아 있는 은닉 단위)", "",
          "| 기준 | " + " | ".join(f"{100 * D:g}%" for D in LEARNED_D) + " |", "|---|" + "---|" * len(LEARNED_D)]
    for c in CRITS:
        cells = []
        for D in LEARNED_D:
            h = S["learned"][c][f"{D:g}"]
            cells.append(f"{pm(h['acc'])} / {100 * h['wasted_frac']:.1f} / {h['alive_units']:.0f}")
        L.append(f"| {LABEL[c]} | " + " | ".join(cells) + " |")
    L += ["", "## BN 재보정 전 정확도 (균일 배분, 참고)", "", "| 기준 | " + " | ".join(f"{100 * D:g}%" for D in DENSITIES) + " |",
          "|---|" + "---|" * len(DENSITIES)]
    for c in ("wanda_row", "prepost_layer", "magnitude"):
        L.append(f"| {LABEL[c]} | " + " | ".join(f"{100 * S['uniform'][c][f'{D:g}']['acc_stale_bn']:.1f}" for D in DENSITIES) + " |")
    md = "\n".join(L)
    with open(os.path.join(RES_P3, "tables_h1cnn.md"), "w", encoding="utf-8") as f:
        f.write(md)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fname in ("Malgun Gothic", "NanumGothic"):
        if any(fname == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fname
            break
    plt.rcParams["axes.unicode_minus"] = False
    xs = [100 * D for D in DENSITIES]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for c in ("wanda_row", "wanda_layer", "prepost_layer", "prepost_row", "magnitude", "magnitude_row", "conn_drive", "ria_row", "ria_layer", "random"):
        a = [S["uniform"][c][f"{D:g}"]["acc"] for D in DENSITIES]
        lw = 2.4 if c in ("wanda_row", "prepost_layer") else 1.4
        ax[0].errorbar(xs, [100 * x[0] for x in a], yerr=[100 * x[1] for x in a], color=COLORS[c], linestyle=STYLE[c], marker="o",
                       ms=4, lw=lw, capsize=2, label=LABEL[c])
        ax[1].plot(xs, [S["uniform"][c][f"{D:g}"]["alive_units"] for D in DENSITIES], color=COLORS[c], linestyle=STYLE[c], marker="o",
                   ms=4, lw=lw, label=LABEL[c])
    ax[0].axhline(100 * S["dense_acc"], color="#212529", lw=0.8, ls=":")
    ax[0].text(100 * 0.03, 100 * S["dense_acc"] + 1.2, f"밀집 {100 * S['dense_acc']:.1f}% (깎기 전)", ha="center", fontsize=8, color="#495057")
    ax[0].set_ylim(0, 100)
    for a_ in ax:
        a_.set_xscale("log")
        a_.invert_xaxis()
        a_.set_xticks(xs)
        a_.set_xticklabels([f"{x:g}%" for x in xs])
        a_.grid(alpha=0.3, which="both")
        a_.set_xlabel("깎는 층의 밀도 (왼쪽이 덜 깎음)")
    ax[0].set_ylabel("시험 정확도 (%)")
    ax[0].set_title("CNN 배포 결산 가지치기 (BN 재보정, 미세조정 없음)")
    ax[1].set_ylabel(f"살아 있는 은닉 단위 수 (최대 {S['uniform']['magnitude'][f'{DENSITIES[0]:g}']['units']})")
    ax[1].set_title("허브 배분: 살아 남은 채널·뉴런 수")
    ax[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(RES_P3, "fig_h1cnn_sweep.png"), dpi=140)
    plt.close(fig)
    print(md)
    return S


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "summarize":
        summarize()
    elif args and args[0] == "smoke":
        # 연기 시험: 1 에폭짜리 밀집 CNN 으로 경로만 확인 (캐시 오염 방지 위해 OUT_DIR 을 바꾼다)
        OUT_DIR = os.path.join(RES_P3, "_smoke_h1cnn")
        globals()["OUT_DIR"] = OUT_DIR
        DENSITIES[:] = [0.1, 0.01]
        TRANSPLANT_D[:] = [0.01]
        LEARNED_D[:] = [0.01]
        device = torch.device("cuda")
        os.makedirs(OUT_DIR, exist_ok=True)
        m = train_dense_cnn(0, device, epochs=1)
        r = run_cnn(0)
        print("SMOKE dense", r["dense_acc"], "U1%", {c: round(r["uniform"][c]["0.01"]["acc"], 3) for c in CRITS})
    else:
        for s in [int(a) for a in args] or [0, 1, 2]:
            run_cnn(s)
        summarize()
