# -*- coding: utf-8 -*-
"""H1c: 배포 결산 가지치기의 밀도 스윕 (Wanda 본래 영역 50% 부터 0.5% 까지) + 허브 배분 측정 + 차수 이식 분해.

외부 검토 (2026-10-07): "Wanda 는 원래 50% 용도 — 1% 에서만 이기면 '안 쓰는 영역에서 이긴 것뿐'. 50/10/1% 를 다 보이고 어디서
갈리는지 그려라. 그리고 이유가 허브 배분이라는 걸 보여라."

설정 (Wanda 본래 프로토콜): 은닉 두 층(net.0, net.2)을 같은 밀도 D 로 균일하게 깎고 출력 머리(net.4)는 깎지 않는다.
미세조정 없음. 흔적은 H1 과 같은 배포 스트림(훈련 집합 1 에폭, 라벨 없음)에서 한 번 쌓아 모든 기준이 공유한다.

기준 (층 = 층 안 전역 순위, 행 = 출력 뉴런마다 같은 수, Wanda 식):
  magnitude / magnitude_row       abs(w)
  wanda_row / wanda_layer         abs(w) x ||X_j||_2  (wanda_row 가 Wanda 원본)
  prepost_layer / prepost_row     abs(w) x 평균|x_j| x 평균|y_i|  (H1 의 rank1_prepost; 행 판은 행 안 순위가 abs(w)x평균|x_j| 와 같다)
  conn_drive                      abs(w) x E[1(y_i>0)|x_j|]       (연결 단위 흔적, O(연결))
  coincidence / coincidence_w     |E[relu(y_i) x_j]| (Scholl 등 2021 의 '전·후 동시 발화' 추정의 ReLU 판, 가중치 없음 / abs(w) 곱)
  rates_only                      평균|x_j| x 평균|y_i| (가중치 없음)
  random
허브 지표: 살아 있는 은닉 뉴런, 들어오는 연결 수(행 차수)의 지니계수·상위 10% 뉴런 몫, 행 차수와 받는 뉴런 활동의 스피어만 상관.
차수 이식: '어느 뉴런이 연결을 몇 개 갖나' 는 기준 A 에서, '그 뉴런 안에서 어느 연결' 은 기준 B 에서 — 허브 배분만으로 격차가 설명되는지.
결과: results/p3/h1c/seed<k>.json, summarize() -> summary.json, tables_h1c.md, fig_h1c_sweep.png, fig_h1c_transplant.png
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.masked_layers import apply_masks, masked_modules                    # noqa: E402
from core.pruning import ActivityTracker                                      # noqa: E402
from experiments.p3.common import RES_P3, alive_hidden_neurons, count_active, evaluate, get_mnist  # noqa: E402
from experiments.p3.h1 import (NeuronTraces, fresh_masked, prune_layerwise, prune_rowwise, scores_for,  # noqa: E402
                               stream, train_dense)
from utils.seed import set_seed                                               # noqa: E402

DENSITIES = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
TRANSPLANT_D = [0.1, 0.05, 0.02, 0.01, 0.005]
HIDDEN = ("net.0", "net.2")
HEAD = "net.4"
OUT_DIR = os.path.join(RES_P3, "h1c")
CRIT = {  # 이름: (h1.scores_for 의 기준 이름 또는 None=여기서 계산, 입도)
    "magnitude": ("magnitude", "layer"), "magnitude_row": ("magnitude", "row"),
    "wanda_row": ("wanda_row", "row"), "wanda_layer": ("wanda_global", "layer"),
    "prepost_layer": ("rank1_prepost", "layer"), "prepost_row": ("rank1_prepost", "row"),
    "conn_drive": ("conn_drive", "layer"),
    "coincidence": (None, "layer"), "coincidence_w": (None, "layer"),
    "rates_only": ("rank1_now", "layer"), "random": ("random", "layer"),
}
TRANSPLANTS = {  # 이름: (차수 원천 기준, 선택 기준)
    "deg(prepost)+sel(magnitude)": ("prepost_layer", "magnitude"),
    "deg(prepost)+sel(wanda)": ("prepost_layer", "wanda_layer"),
    "deg(magnitude)+sel(prepost)": ("magnitude", "prepost_layer"),
    "deg(wanda_layer)+sel(prepost)": ("wanda_layer", "prepost_layer"),
    "deg(prepost)+sel(prepost) [항등 점검]": ("prepost_layer", "prepost_layer"),
}


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n, s = x.size, x.sum()
    if n == 0 or s <= 0:
        return 0.0
    return float((2.0 * np.sum(np.arange(1, n + 1) * x)) / (n * s) - (n + 1.0) / n)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def hub_stats(model, traces: NeuronTraces) -> dict:
    out = {"alive_hidden": alive_hidden_neurons(model)}
    mods = dict(masked_modules(model))
    for name in HIDDEN:
        deg = mods[name].weight_mask.sum(1).cpu().numpy()
        k = max(1, int(round(0.1 * deg.size)))
        top = np.sort(deg)[::-1][:k].sum() / max(deg.sum(), 1)
        out[name] = {"gini_in": gini(deg), "top10_share": float(top), "zero_rows": int((deg == 0).sum()),
                     "spearman_deg_tout": spearman(deg, traces.t_out[name].cpu().numpy())}
    return out


EXTRA_CRIT = {"ria_row": (None, "row"), "ria_layer": (None, "layer")}   # Zhang 등 2024 RIA (허브의 반대 방향 대조, 2026-10-07 추가)


@torch.no_grad()
def compute_scores(model, crit: str, traces, tracker, gen) -> Dict[str, torch.Tensor]:
    if crit in EXTRA_CRIT:
        out = {}
        for name, m in masked_modules(model):
            w = m.weight.abs()
            s = (w / (w.sum(1, keepdim=True) + 1e-12) + w / (w.sum(0, keepdim=True) + 1e-12)) * traces.sq_in[name].sqrt().pow(0.5)[None, :]
            out[name] = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        return out
    base, _ = CRIT[crit]
    if base is not None:
        return scores_for(model, base, traces, tracker, gen)
    out = {}
    for name, m in masked_modules(model):
        c = tracker.traces[name].abs()
        out[name] = torch.nan_to_num(c if crit == "coincidence" else m.weight.abs() * c, nan=0.0, posinf=0.0, neginf=0.0)
    return out


@torch.no_grad()
def prune_transplant(model, deg_masks: Dict[str, torch.Tensor], scores: Dict[str, torch.Tensor]) -> None:
    """행 i 마다 k_i = 원천 마스크의 행 i 연결 수, 선택은 scores 행 i 의 상위 k_i."""
    for name, m in masked_modules(model):
        k = deg_masks[name].sum(1).long()                                   # (out,)
        order = torch.argsort(scores[name], dim=1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(order.shape[1], device=order.device).expand_as(order))
        keep = rank < k[:, None]
        m.weight_mask.copy_(keep.to(m.weight_mask.dtype))
    apply_masks(model)


def targets_for(D: float) -> Dict[str, float]:
    return {"net.0": D, "net.2": D, HEAD: 1.0}


def run_h1c(seed: int, log=print) -> dict:
    device = torch.device("cuda")
    x_tr, y_tr, x_te, y_te = get_mnist(device)
    dense = train_dense(seed, device, log)
    dense_state = copy.deepcopy(dense.state_dict())
    dense_acc = evaluate(dense, x_te, y_te)
    t0 = time.time()
    model = fresh_masked(dense_state, seed, device)
    traces, tracker = NeuronTraces(model), ActivityTracker(model, momentum=0.99)
    set_seed(seed + 11)
    stream(model, x_tr, y_tr, traces, tracker, device)
    traces.remove()
    tracker.remove()
    total_w = sum(m.weight_mask.numel() for _, m in masked_modules(model))
    gen = torch.Generator(device="cpu").manual_seed(seed + 4242)
    res = {"seed": seed, "dense_acc": dense_acc, "sweep": {}, "transplant": {}, "hub": {}}
    masks_at: Dict[tuple, Dict[str, torch.Tensor]] = {}
    for D in DENSITIES:
        tg = targets_for(D)
        for crit, (_, gran) in CRIT.items():
            m = fresh_masked(dense_state, seed, device)
            sc = compute_scores(m, crit, traces, tracker, gen)
            (prune_rowwise if gran == "row" else prune_layerwise)(m, sc, tg)
            acc = evaluate(m, x_te, y_te)
            res["sweep"].setdefault(crit, {})[f"{D:g}"] = {"acc": acc, "total_density": count_active(m) / total_w}
            res["hub"].setdefault(crit, {})[f"{D:g}"] = hub_stats(m, traces)
            if D in TRANSPLANT_D and crit in ("prepost_layer", "magnitude", "wanda_layer"):
                masks_at[(crit, D)] = {n: mm.weight_mask.clone() for n, mm in masked_modules(m)}
        log(f"  [h1c s{seed}] D={D:g}: " + " ".join(f"{c}={res['sweep'][c][f'{D:g}']['acc']:.3f}" for c in CRIT)
            + f" ({time.time() - t0:.0f}s)")
    for D in TRANSPLANT_D:
        for tname, (src, sel) in TRANSPLANTS.items():
            m = fresh_masked(dense_state, seed, device)
            sc = compute_scores(m, sel, traces, tracker, gen)
            prune_transplant(m, masks_at[(src, D)], sc)
            res["transplant"].setdefault(tname, {})[f"{D:g}"] = {"acc": evaluate(m, x_te, y_te), "hub": hub_stats(m, traces)}
        log(f"  [h1c s{seed}] transplant D={D:g}: " + " ".join(f"{t.split(' ')[0]}={res['transplant'][t][f'{D:g}']['acc']:.3f}"
                                                          for t in TRANSPLANTS))
    res["elapsed_s"] = time.time() - t0
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


# ---------------------------------------------------------------------------
# 집계·그림
# ---------------------------------------------------------------------------
COLORS = {  # 기준마다 고정 색 (계열: Wanda 파랑, 우리 주황, 크기 회색, 연결 단위 초록, 동시 발화 보라)
    "wanda_row": "#1f5fbf", "wanda_layer": "#7fa7e6", "prepost_layer": "#d9480f", "prepost_row": "#f4a261",
    "magnitude": "#495057", "magnitude_row": "#adb5bd", "conn_drive": "#2b8a3e", "coincidence_w": "#7048e8",
    "coincidence": "#b197fc", "rates_only": "#e599f7", "random": "#ced4da",
}
STYLE = {"wanda_row": "-", "wanda_layer": "--", "prepost_layer": "-", "prepost_row": "--", "magnitude": "-",
         "magnitude_row": "--", "conn_drive": "-", "coincidence_w": "-", "coincidence": ":", "rates_only": ":", "random": ":"}
LABEL = {"wanda_row": "Wanda (행별, 원본)", "wanda_layer": "Wanda 점수 (층 순위)", "prepost_layer": "보내는×받는 흐름 (층 순위)",
         "prepost_row": "보내는×받는 흐름 (행별)", "magnitude": "크기 (층 순위)", "magnitude_row": "크기 (행별)",
         "conn_drive": "연결 단위 구동 흔적", "coincidence_w": "동시 발화×abs(w) (Scholl 식)", "coincidence": "동시 발화 (가중치 없음)",
         "rates_only": "흐름만 (가중치 없음)", "random": "무작위"}


def summarize() -> dict:
    runs = []
    for k in range(3):
        p = os.path.join(OUT_DIR, f"seed{k}.json")
        if os.path.exists(p):
            runs.append(json.load(open(p, encoding="utf-8")))
    if not runs:
        return {}

    def agg(get):
        v = [get(r) for r in runs]
        v = [x for x in v if x is not None]
        return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (None, None, 0)

    S = {"n": len(runs), "dense_acc": agg(lambda r: r["dense_acc"])[0], "sweep": {}, "hub": {}, "transplant": {}}
    for c in CRIT:
        for D in DENSITIES:
            k = f"{D:g}"
            m, s, n = agg(lambda r: r["sweep"][c][k]["acc"])
            S["sweep"].setdefault(c, {})[k] = {"mean": m, "std": s, "n": n,
                                                "total_density": agg(lambda r: r["sweep"][c][k]["total_density"])[0]}
            S["hub"].setdefault(c, {})[k] = {
                "alive_hidden": agg(lambda r: r["hub"][c][k]["alive_hidden"])[0],
                "gini_in_net0": agg(lambda r: r["hub"][c][k]["net.0"]["gini_in"])[0],
                "gini_in_net2": agg(lambda r: r["hub"][c][k]["net.2"]["gini_in"])[0],
                "top10_net2": agg(lambda r: r["hub"][c][k]["net.2"]["top10_share"])[0],
                "spearman_net2": agg(lambda r: r["hub"][c][k]["net.2"]["spearman_deg_tout"])[0],
            }
    for t in TRANSPLANTS:
        for D in TRANSPLANT_D:
            k = f"{D:g}"
            m, s, n = agg(lambda r: r["transplant"][t][k]["acc"])
            S["transplant"].setdefault(t, {})[k] = {"mean": m, "std": s, "n": n,
                                                     "alive_hidden": agg(lambda r: r["transplant"][t][k]["hub"]["alive_hidden"])[0]}
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, ensure_ascii=False, indent=1)

    def pct(x):
        return "-" if x is None else f"{100 * x:.1f}"

    L = [f"## H1c 밀도 스윕 (은닉 두 층 균일 밀도 D, 출력 머리 유지, 미세조정 없음; 시험 정확도 %, 시드 {len(runs)}개 평균±표준편차; 밀집 {pct(S['dense_acc'])})", "",
         "| 기준 | " + " | ".join(f"{100 * D:g}%" for D in DENSITIES) + " |", "|---|" + "---|" * len(DENSITIES)]
    for c in CRIT:
        L.append(f"| {LABEL[c]} | " + " | ".join(f"{pct(S['sweep'][c][f'{D:g}']['mean'])}±{pct(S['sweep'][c][f'{D:g}']['std'])}"
                                                for D in DENSITIES) + " |")
    L += ["", "## H1c 허브 지표 (살아 있는 은닉 뉴런 / 2층 행 차수 지니 / 2층 상위 10% 뉴런 몫 / 2층 차수-받는 활동 스피어만)", "",
          "| 기준 | " + " | ".join(f"{100 * D:g}%" for D in DENSITIES) + " |", "|---|" + "---|" * len(DENSITIES)]
    for c in ("wanda_row", "wanda_layer", "prepost_layer", "prepost_row", "magnitude", "conn_drive", "coincidence_w"):
        cells = []
        for D in DENSITIES:
            h = S["hub"][c][f"{D:g}"]
            cells.append(f"{h['alive_hidden']:.0f} / {h['gini_in_net2']:.2f} / {h['top10_net2']:.2f} / {h['spearman_net2']:.2f}")
        L.append(f"| {LABEL[c]} | " + " | ".join(cells) + " |")
    L += ["", "## H1c 차수 이식 (행마다 연결 수는 원천 기준에서, 행 안 선택은 선택 기준에서; 시험 정확도 %)", "",
          "| 이식 | " + " | ".join(f"{100 * D:g}%" for D in TRANSPLANT_D) + " |", "|---|" + "---|" * len(TRANSPLANT_D)]
    for t in TRANSPLANTS:
        L.append(f"| {t} | " + " | ".join(f"{pct(S['transplant'][t][f'{D:g}']['mean'])}±{pct(S['transplant'][t][f'{D:g}']['std'])}"
                                          for D in TRANSPLANT_D) + " |")
    md = "\n".join(L)
    with open(os.path.join(RES_P3, "tables_h1c.md"), "w", encoding="utf-8") as f:
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
    for c in ("wanda_row", "wanda_layer", "prepost_layer", "prepost_row", "magnitude", "magnitude_row", "conn_drive", "coincidence_w", "random"):
        ys = [100 * S["sweep"][c][f"{D:g}"]["mean"] for D in DENSITIES]
        es = [100 * S["sweep"][c][f"{D:g}"]["std"] for D in DENSITIES]
        lw = 2.4 if c in ("wanda_row", "prepost_layer") else 1.4
        ax[0].errorbar(xs, ys, yerr=es, color=COLORS[c], linestyle=STYLE[c], marker="o", ms=4, lw=lw, capsize=2, label=LABEL[c])
        al = [S["hub"][c][f"{D:g}"]["alive_hidden"] for D in DENSITIES]
        ax[1].plot(xs, al, color=COLORS[c], linestyle=STYLE[c], marker="o", ms=4, lw=lw, label=LABEL[c])
    ax[0].axhline(100 * S["dense_acc"], color="#212529", lw=0.8, ls=":")
    ax[0].text(100 * 0.02, 100 * S["dense_acc"] + 1.2, f"밀집 {100 * S['dense_acc']:.1f}% (깎기 전)", ha="center", fontsize=8, color="#495057")
    ax[0].set_ylim(0, 103)
    for a_ in ax:
        a_.set_xscale("log")
        a_.invert_xaxis()
        a_.set_xticks(xs)
        a_.set_xticklabels([f"{x:g}%" for x in xs])
        a_.grid(alpha=0.3, which="both")
        a_.set_xlabel("은닉층 밀도 (남긴 연결 비율, 왼쪽이 덜 깎음)")
    ax[0].set_ylabel("시험 정확도 (%)")
    ax[0].set_title("배포 결산 가지치기: 밀도별 정확도 (미세조정 없음)")
    ax[1].set_ylabel("살아 있는 은닉 뉴런 수 (최대 2,048)")
    ax[1].set_title("허브 배분: 살아 남은 뉴런 수")
    ax[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(RES_P3, "fig_h1c_sweep.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    xs2 = [100 * D for D in TRANSPLANT_D]
    ref = {"prepost_layer": "보내는×받는 흐름 (원본)", "magnitude": "크기 (원본)", "wanda_row": "Wanda (원본)"}
    for c, lab in ref.items():
        ax.plot(xs2, [100 * S["sweep"][c][f"{D:g}"]["mean"] for D in TRANSPLANT_D], color=COLORS[c], lw=2.2, marker="o", label=lab)
    tcol = {"deg(prepost)+sel(magnitude)": "#e8590c", "deg(magnitude)+sel(prepost)": "#868e96", "deg(prepost)+sel(wanda)": "#f08c00",
            "deg(wanda_layer)+sel(prepost)": "#74c0fc"}
    for t, col in tcol.items():
        ax.plot(xs2, [100 * S["transplant"][t][f"{D:g}"]["mean"] for D in TRANSPLANT_D], color=col, lw=1.4, ls="--", marker="s", ms=4, label=t)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(xs2)
    ax.set_xticklabels([f"{x:g}%" for x in xs2])
    ax.grid(alpha=0.3, which="both")
    ax.set_xlabel("은닉층 밀도")
    ax.set_ylabel("시험 정확도 (%)")
    ax.set_title("차수 이식: 뉴런별 연결 수(허브 배분)만 옮겨도 되는가")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES_P3, "fig_h1c_transplant.png"), dpi=140)
    plt.close(fig)
    print(md)
    return S


@torch.no_grad()
def dead_end_stats(model) -> dict:
    """층 사이 끊긴 경로: 남긴 연결 중 '입력이 없는 뉴런에서 나가거나(출처 없음) 출력이 없는 뉴런으로 들어가는(막다른 길)' 비율."""
    M = {n: m.weight_mask.bool() for n, m in masked_modules(model)}
    in1, out1 = M["net.0"].sum(1), M["net.2"].sum(0)
    in2, out2 = M["net.2"].sum(1), M["net.4"].sum(0)
    w0 = int(M["net.0"][out1 == 0].sum())                                         # 출력 없는 1 층 뉴런으로 들어감
    w2 = int((M["net.2"] & ((out2 == 0)[:, None] | (in1 == 0)[None, :])).sum())   # 막다른 2 층 뉴런 또는 출처 없는 1 층 뉴런
    w4 = int(M["net.4"][:, in2 == 0].sum())                                        # 출처 없는 2 층 뉴런에서 나감
    kept = int(sum(v.sum() for v in M.values()))
    return {"wasted_frac": (w0 + w2 + w4) / max(kept, 1), "kept": kept, "w0": w0, "w2": w2, "w4": w4}


def run_deadend(seeds=(0, 1, 2), log=print) -> dict:
    """배분 방식 3 가지(은닉 균일+머리 유지 / 학습 배분(p2 마스크, 머리 깎음) / 머리까지 균일)에서 기준별 정확도·낭비 연결·살아 있는 뉴런."""
    from experiments.p3.h1 import per_layer_targets
    device = torch.device("cuda")
    x_tr, y_tr, x_te, y_te = get_mnist(device)
    crits = ["magnitude", "prepost_layer", "wanda_row", "wanda_layer", "conn_drive", "prepost_row"]
    out = {}
    for seed in seeds:
        dense = train_dense(seed, device, log)
        dense_state = copy.deepcopy(dense.state_dict())
        model = fresh_masked(dense_state, seed, device)
        traces, tracker = NeuronTraces(model), ActivityTracker(model, momentum=0.99)
        set_seed(seed + 11)
        stream(model, x_tr, y_tr, traces, tracker, device)
        traces.remove()
        tracker.remove()
        names = [n for n, _ in masked_modules(model)]
        gen = torch.Generator(device="cpu").manual_seed(seed + 4242)
        for D in (0.02, 0.01, 0.005):
            schemes = {"hidden_uniform_head_dense": targets_for(D), "all_uniform": {n: D for n in names}}
            if D in (0.01, 0.005):
                schemes["learned_alloc"] = per_layer_targets(D, seed, names)
            for sch, tg in schemes.items():
                for c in crits:
                    m = fresh_masked(dense_state, seed, device)
                    sc = compute_scores(m, c, traces, tracker, gen)
                    (prune_rowwise if CRIT[c][1] == "row" else prune_layerwise)(m, sc, tg)
                    r = {"acc": evaluate(m, x_te, y_te), "alive_hidden": alive_hidden_neurons(m), **dead_end_stats(m)}
                    out.setdefault(sch, {}).setdefault(f"{D:g}", {}).setdefault(c, []).append(r)
        log(f"  [deadend s{seed}] done")
    S = {}
    for sch, dd in out.items():
        for D, cc in dd.items():
            for c, rs in cc.items():
                S.setdefault(sch, {}).setdefault(D, {})[c] = {k: float(np.mean([r[k] for r in rs])) for k in ("acc", "alive_hidden", "wasted_frac")}
                S[sch][D][c]["acc_std"] = float(np.std([r["acc"] for r in rs]))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "deadend.json"), "w", encoding="utf-8") as f:
        json.dump({"per_seed": out, "summary": S}, f, ensure_ascii=False, indent=1)
    L = ["", "## H1c 층 사이 끊긴 경로 (배분 방식별; 정확도 % ± 표준편차 / 낭비 연결 비율 % / 살아 있는 은닉 뉴런; 시드 3 평균)", ""]
    for sch in ("hidden_uniform_head_dense", "learned_alloc", "all_uniform"):
        if sch not in S:
            continue
        Ds = sorted(S[sch].keys(), key=float, reverse=True)
        L += [f"### {sch}", "", "| 기준 | " + " | ".join(f"{100 * float(D):g}%" for D in Ds) + " |", "|---|" + "---|" * len(Ds)]
        for c in crits:
            L.append(f"| {LABEL[c]} | " + " | ".join(
                f"{100 * S[sch][D][c]['acc']:.1f}±{100 * S[sch][D][c]['acc_std']:.1f} / {100 * S[sch][D][c]['wasted_frac']:.1f} / {S[sch][D][c]['alive_hidden']:.0f}"
                for D in Ds) + " |")
        L.append("")
    md = "\n".join(L)
    with open(os.path.join(RES_P3, "tables_h1c.md"), "a", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return S


def run_extra(seeds=(0, 1, 2), log=print) -> dict:
    """RIA(행별·층 순위)를 같은 스윕(은닉 균일+머리 유지)과 학습 배분 1%·0.5% 에서. 결과 h1c/extra.json, 표는 tables_h1c.md 에 덧붙임."""
    from experiments.p3.h1 import per_layer_targets
    device = torch.device("cuda")
    x_tr, y_tr, x_te, y_te = get_mnist(device)
    out = {}
    for seed in seeds:
        dense = train_dense(seed, device, log)
        dense_state = copy.deepcopy(dense.state_dict())
        model = fresh_masked(dense_state, seed, device)
        traces, tracker = NeuronTraces(model), ActivityTracker(model, momentum=0.99)
        set_seed(seed + 11)
        stream(model, x_tr, y_tr, traces, tracker, device)
        traces.remove()
        tracker.remove()
        names = [n for n, _ in masked_modules(model)]
        gen = torch.Generator(device="cpu").manual_seed(seed + 4242)
        for crit, (_, gran) in EXTRA_CRIT.items():
            for D in DENSITIES:
                m = fresh_masked(dense_state, seed, device)
                (prune_rowwise if gran == "row" else prune_layerwise)(m, compute_scores(m, crit, traces, tracker, gen), targets_for(D))
                out.setdefault("uniform", {}).setdefault(crit, {}).setdefault(f"{D:g}", []).append(
                    {"acc": evaluate(m, x_te, y_te), "alive_hidden": alive_hidden_neurons(m), **dead_end_stats(m)})
            for D in (0.01, 0.005):
                m = fresh_masked(dense_state, seed, device)
                (prune_rowwise if gran == "row" else prune_layerwise)(m, compute_scores(m, crit, traces, tracker, gen),
                                                                     per_layer_targets(D, seed, names))
                out.setdefault("learned", {}).setdefault(crit, {}).setdefault(f"{D:g}", []).append(
                    {"acc": evaluate(m, x_te, y_te), "alive_hidden": alive_hidden_neurons(m), **dead_end_stats(m)})
        log(f"  [h1c extra s{seed}] done")
    S = {sch: {c: {D: {"acc": float(np.mean([r["acc"] for r in rs])), "acc_std": float(np.std([r["acc"] for r in rs])),
                       "alive_hidden": float(np.mean([r["alive_hidden"] for r in rs])),
                       "wasted_frac": float(np.mean([r["wasted_frac"] for r in rs]))} for D, rs in dd.items()}
               for c, dd in cc.items()} for sch, cc in out.items()}
    with open(os.path.join(OUT_DIR, "extra.json"), "w", encoding="utf-8") as f:
        json.dump({"per_seed": out, "summary": S}, f, ensure_ascii=False, indent=1)
    L = ["", "## H1c 추가: RIA (Zhang 등 2024, 행합·열합 정규화 — 허브 반대 방향) — 정확도 % ± 표준편차 / 살아 있는 은닉 뉴런", "",
         "| 기준 | " + " | ".join(f"{100 * D:g}%" for D in DENSITIES) + " | 학습 배분 1% | 학습 배분 0.5% |",
         "|---|" + "---|" * (len(DENSITIES) + 2)]
    for c in EXTRA_CRIT:
        cells = [f"{100 * S['uniform'][c][f'{D:g}']['acc']:.1f}±{100 * S['uniform'][c][f'{D:g}']['acc_std']:.1f} / {S['uniform'][c][f'{D:g}']['alive_hidden']:.0f}"
                 for D in DENSITIES]
        cells += [f"{100 * S['learned'][c][D]['acc']:.1f}±{100 * S['learned'][c][D]['acc_std']:.1f} / {S['learned'][c][D]['alive_hidden']:.0f}"
                  for D in ("0.01", "0.005")]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    md = "\n".join(L)
    with open(os.path.join(RES_P3, "tables_h1c.md"), "a", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    return S


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "extra":
        run_extra()
    elif args and args[0] == "deadend":
        run_deadend()
    elif args and args[0] == "summarize":
        summarize()
    else:
        for s in [int(a) for a in args] or [0, 1, 2]:
            run_h1c(s)
        summarize()
