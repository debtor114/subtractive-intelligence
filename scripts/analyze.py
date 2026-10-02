# -*- coding: utf-8 -*-
"""결과 집계 + 표 + 그림 생성.

  python scripts/analyze.py            # 전부
  python scripts/analyze.py core exp3  # 일부

출력: docs/report/tables/<name>.md, docs/report/figures/<name>.png
표는 시드 평균 +- 표준편차. 그림 색은 dataviz 기본 팔레트 (범주형 고정 순서).
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO_ROOT, "results")
TAB = os.path.join(REPO_ROOT, "docs", "report", "tables")
FIG = os.path.join(REPO_ROOT, "docs", "report", "figures")
# 논문용 그림 모드 (PAPER_FIGS=1): 한국어 제목 생략, 기본 글꼴, paper/figures 에 저장. 보고서용 그림은 그대로.
PAPER = os.environ.get("PAPER_FIGS") == "1"
if PAPER:
    FIG = os.path.join(REPO_ROOT, "paper", "figures")

os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# dataviz 기본 팔레트 (light). 범주형은 이 순서를 고정해서 쓴다.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0, "lines.markersize": 7,
    "font.size": 10, "legend.frameon": False, "axes.titleweight": "semibold", "axes.titlesize": 11,
    "font.family": ["Malgun Gothic", "Segoe UI", "sans-serif"], "axes.unicode_minus": False,
})


if PAPER:
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    plt.Axes.set_title = lambda self, *a, **k: None   # 그림 제목은 캡션이 대신한다

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ms(vals: List[float], fmt: str = "{:.4f}") -> str:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return "-"
    m = float(np.mean(vals))
    if len(vals) == 1:
        return fmt.format(m)
    return (fmt + " +- " + fmt).format(m, float(np.std(vals)))


def sci(v: float) -> str:
    return "-" if v is None else f"{v:.2e}"


def write_table(name: str, header: List[str], rows: List[List[str]], note: str = "") -> None:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if note:
        lines += ["", note]
    with open(os.path.join(TAB, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[table] {name}: {len(rows)} rows")


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=150)
    plt.close(fig)
    print(f"[figure] {name}")


# ---------------------------------------------------------------------------
# 기준선
# ---------------------------------------------------------------------------

def analyze_baselines():
    groups = defaultdict(list)
    for f in glob.glob(os.path.join(RES, "baseline_*", "seed*", "results.json")):
        r = load_json(f)
        groups[r["run_name"]].append(r)
    if not groups:
        return
    rows = []
    for name, rs in sorted(groups.items()):
        rows.append([name, len(rs), f"{rs[0]['params_total']:,}", sci(rs[0]["flops_dense_per_sample"]),
                     ms([r["final_test_acc"] for r in rs]), ms([r["calibration"]["ece"] for r in rs], "{:.3f}"),
                     ms([r["calibration"]["auroc"] for r in rs], "{:.3f}"),
                     ms([r["total_train_time_s"] for r in rs], "{:.0f}")])
    write_table("baselines", ["run", "seeds", "params", "FLOPs/sample", "test acc", "ECE", "AUROC(correct)", "time(s)"], rows)


# ---------------------------------------------------------------------------
# 핵심 실험
# ---------------------------------------------------------------------------

# 용어는 본문과 같게: prune-after (학습 후 가지치기), Hebbian (activity 규칙), uniform per layer (층별 균등 배분)
ARM_LABEL = {
    "pd_drive_layer": "prune-during: synaptic drive (uniform per layer)",
    "pd_drive_erk": "prune-during: synaptic drive (ERK per layer)", "pd_drivenorm_erk": "prune-during: synaptic drive, per-neuron normalised (ERK per layer)",
    "pd_mag_erk": "prune-during: magnitude (ERK per layer)",
    "dense_small": "dense small (additive)", "dense_small_shallow": "dense small, shallow and wide (additive)",
    "dense_big": "dense big (no pruning)", "static_sparse": "static random sparse",
    "set": "SET (dynamic sparse training, random regrowth)", "rigl": "RigL (dynamic sparse training, gradient regrowth)",
    "rigl_x3": "RigL, 3x training length",
    "pd_mag_layer": "prune-during: magnitude (uniform per layer)", "pd_mag_global": "prune-during: magnitude (global)",
    "pd_act_layer": "prune-during: activity, Hebbian (uniform per layer)", "pd_actmag_layer": "prune-during: abs(w) x activity, Hebbian (uniform per layer)",
    "pd_random_layer": "prune-during: random (uniform per layer)",
    "ttp": "prune-after: one-shot magnitude (global) + fine-tune", "ttp_gradual": "prune-after: gradual magnitude (global) during fine-tune",
}
ARM_ORDER = ["dense_small", "dense_small_shallow", "ttp", "ttp_gradual", "pd_mag_layer", "pd_mag_global", "pd_mag_erk", "pd_act_layer", "pd_actmag_layer",
             "pd_drive_layer", "pd_drive_erk", "pd_drivenorm_erk", "pd_random_layer", "set", "rigl", "rigl_x3", "static_sparse"]


# 팔별 고정 색과 선 모양 (팔을 추가해도 기존 색이 바뀌지 않도록 이름으로 고정; 본문 캡션의 색 이름이 여기에 맞춰져 있다)
ARM_STYLE = {
    "dense_small": ("#2a78d6", "-"), "dense_small_shallow": ("#2a78d6", "--"),
    "ttp": ("#eb6834", "-"), "ttp_gradual": ("#eb6834", "--"),
    "pd_mag_global": ("#eda100", "-"), "pd_mag_erk": ("#eda100", "--"), "pd_mag_layer": ("#1baf7a", "-"),
    "pd_act_layer": ("#4a3aa7", "-"), "pd_actmag_layer": ("#4a3aa7", "--"),
    "pd_drive_layer": ("#e87ba4", "-"), "pd_drive_erk": ("#e87ba4", "--"), "pd_drivenorm_erk": ("#e87ba4", "-"),
    "pd_random_layer": ("#e34948", "-"), "set": ("#008300", "--"), "rigl": ("#008300", "-"), "rigl_x3": ("#008300", ":"),
    "static_sparse": ("#8a8a8a", ":"), "dense_big": ("#303030", "-"),
}
# 범례용 짧은 라벨 (표는 ARM_LABEL 의 긴 이름을 쓴다)
ARM_SHORT = {
    "dense_small": "dense small (additive)", "dense_small_shallow": "dense small, shallow and wide",
    "ttp": "prune-after, one-shot", "ttp_gradual": "prune-after, gradual",
    "pd_mag_global": "prune-during, magnitude (global)", "pd_mag_erk": "prune-during, magnitude (ERK)",
    "pd_mag_layer": "prune-during, magnitude (uniform per layer)",
    "pd_act_layer": "prune-during, activity (Hebbian)", "pd_actmag_layer": "prune-during, |w| x activity (Hebbian)",
    "pd_drive_layer": "prune-during, synaptic drive", "pd_drive_erk": "prune-during, drive (ERK)",
    "pd_drivenorm_erk": "prune-during, drive, per-neuron normalised (ERK)",
    "pd_random_layer": "prune-during, random", "set": "SET", "rigl": "RigL", "rigl_x3": "RigL, 3x training",
    "static_sparse": "static random sparse", "dense_big": "dense big",
}


def style(arm):
    return ARM_STYLE.get(arm, ("#303030", "-"))


def converged(rs):
    """발산한 시드 (우연 수준 정확도) 를 뺀 run. 그림은 이 평균을 쓰고, 표는 전 시드를 쓴다 (본문 표 2 와 같은 규칙)."""
    ok = [r for r in rs if r["final_acc"] > 0.15]
    return ok or rs


def analyze_core(root: str = "core", suffix: str = "", ylim=(0.85, 1.0), dataset_label: str = "MNIST", big_weights: str = "1.86M"):
    data = defaultdict(lambda: defaultdict(list))  # density -> arm -> [runs]
    for f in glob.glob(os.path.join(RES, root, "d*", "*", "seed*.json")):
        r = load_json(f)
        data[r["density"]][r["arm"]].append(r)
    if not data:
        return
    big = data.get(1.0, {}).get("dense_big", [])
    present = [a for a in ARM_ORDER if any(a in data[d] for d in data if d < 1.0)]
    cidx = {a: i for i, a in enumerate(present)}
    rows = []
    for d in sorted(data, reverse=True):
        for arm in ARM_ORDER + ["dense_big"]:
            rs = data[d].get(arm)
            if not rs:
                continue
            rows.append([f"{d:g}", ARM_LABEL.get(arm, arm), len(rs), f"{int(np.mean([r['final_active'] for r in rs])):,}",
                         ms([r["final_acc"] for r in rs]), ms([r["calibration"]["ece"] for r in rs], "{:.3f}"),
                         sci(np.mean([r["infer_flops"] for r in rs])), sci(np.mean([r["cum_train_flops"] for r in rs])),
                         ms([(r.get("data_efficiency_97") or r.get("data_efficiency_85"))["samples_to_target"] for r in rs], "{:.0f}")])
    write_table(f"core_summary{suffix}", ["density", "arm", "seeds", "active weights", "test acc", "ECE", "infer FLOPs",
                                 "cum. train FLOPs", "samples to target (97% MNIST / 85% CIFAR)"], rows,
                "density 는 과잉 초기화 망(784-1024-1024-10, 1,861,632 가중치) 대비 최종 활성 비율. "
                "dense small 은 같은 예산의 작은 망. 학습 FLOPs 는 매 스텝 실제 활성 연결 기준 누적 (순전파 x3).")

    # 그림 1: 정확도 대 활성 연결 수
    densities = sorted(d for d in data if d < 1.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    diverged_pts = []
    for arm in present:
        i = cidx[arm]
        xs, ys, es = [], [], []
        for d in densities:
            rs = data[d].get(arm)
            if rs:
                xs.append(np.mean([r["final_active"] for r in rs]))
                ys.append(np.mean([r["final_acc"] for r in converged(rs)]))
                es.append(np.std([r["final_acc"] for r in converged(rs)]))
                if len(converged(rs)) < len(rs):
                    diverged_pts.append((xs[-1], np.mean([r["final_acc"] for r in rs]), style(arm)[0]))
        if xs:
            ax.errorbar(xs, ys, yerr=es, label=ARM_SHORT.get(arm, arm), color=style(arm)[0], linestyle=style(arm)[1], marker="o",
                        capsize=2)
    ax.set_xscale("log")
    if big:
        big_acc = np.mean([r["final_acc"] for r in big])
        ax.axhline(big_acc, color=MUTED, linewidth=1, linestyle=":")
        ax.text(0.01, big_acc + 0.001, f"dense big ({big_weights} weights, no pruning)", color=INK2, fontsize=8,
                transform=ax.get_yaxis_transform())
    ax.set_xlabel("final active weights")
    ax.set_ylabel(f"{dataset_label} test accuracy")
    ax.set_title(f"{dataset_label}: 정확도 대 최종 활성 연결 수 (시드 평균)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    save(fig, f"core_acc_vs_active{suffix}")

    # 그림 2: 가지치기 궤적 + 정확도 (가장 희소한 밀도, seed 0)
    d_show = min(densities) if densities else None
    if d_show is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for arm in present:
            i = cidx[arm]
            rs = data[d_show].get(arm)
            if not rs or "_x" in arm:   # k 배 긴 학습 팔은 축이 늘어나 궤적 그림에서는 뺀다 (표에는 있음)
                continue
            r = sorted(rs, key=lambda x: x["seed"])[0]
            c = [x for x in r["curve"] if x.get("phase") in ("main", "finetune", "post_prune", "pre_prune", "final")]
            axes[0].plot([x["step"] for x in c], [x["active"] for x in c], color=style(arm)[0], label=ARM_SHORT.get(arm, arm),
                         linestyle=style(arm)[1])
            axes[1].plot([x["step"] for x in c], [x["test_acc"] for x in c], color=style(arm)[0],
                         linestyle=style(arm)[1])
        axes[0].set_yscale("log")
        axes[0].set_xlabel("training step")
        axes[0].set_ylabel("active weights")
        axes[0].set_title(f"연결 수 궤적 (density {d_show:g}, seed 0)")
        axes[1].set_xlabel("training step")
        axes[1].set_ylabel("test accuracy")
        axes[1].set_ylim(ylim[0] - 0.25, ylim[1])
        axes[1].set_title("정확도 궤적 (학습 후 가지치기는 본 학습 끝에 한 번에 깎임)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
        fig.tight_layout(rect=[0, 0.2, 1, 1])
        fig.savefig(os.path.join(FIG, f"core_trajectory{suffix}.png"), dpi=150)
        plt.close(fig)
        print(f"[figure] core_trajectory{suffix}")

    # 그림 3: 누적 학습 FLOPs 대 정확도
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in present:
        i = cidx[arm]
        xs, ys = [], []
        for d in densities:
            rs = data[d].get(arm)
            if rs:
                xs.append(np.mean([r["cum_train_flops"] for r in rs]))
                ys.append(np.mean([r["final_acc"] for r in converged(rs)]))
        if xs:
            ax.plot(xs, ys, marker="o", color=style(arm)[0], label=ARM_SHORT.get(arm, arm), linestyle=style(arm)[1])
    ax.set_xscale("log")
    ax.set_xlabel("cumulative training FLOPs (forward x3, actual density per step)")
    ax.set_ylabel(f"{dataset_label} test accuracy")
    ax.set_title(f"{dataset_label}: 학습 비용 대 정확도 (점 = 밀도 단계)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    save(fig, f"core_trainflops_vs_acc{suffix}")

    # 그림 3b: 추론 FLOPs 대 정확도 (CNN 은 같은 가중치 수라도 conv 를 남기면 FLOPs 가 커진다)
    fig, ax = plt.subplots(figsize=(8, 5))
    diverged_pts = []
    for arm in present:
        i = cidx[arm]
        xs, ys = [], []
        for d in densities:
            rs = data[d].get(arm)
            if rs:
                xs.append(np.mean([r["infer_flops"] for r in rs]))
                ys.append(np.mean([r["final_acc"] for r in converged(rs)]))
                if len(converged(rs)) < len(rs):
                    diverged_pts.append((xs[-1], np.mean([r["final_acc"] for r in rs]), style(arm)[0]))
        if xs:
            ax.plot(xs, ys, marker="o", color=style(arm)[0], label=ARM_SHORT.get(arm, arm), linestyle=style(arm)[1])
    for j, (x, y, c) in enumerate(diverged_pts):
        ax.plot([x], [y], marker="o", markerfacecolor="none", color=c, linestyle="none",
                label="all-seed mean incl. diverged seed" if j == 0 else None)
    ax.set_xscale("log")
    ax.set_xlabel("inference FLOPs per sample (effective)")
    ax.set_ylabel(f"{dataset_label} test accuracy")
    ax.set_title(f"{dataset_label}: 추론 연산량 대 정확도 (점 = 밀도 단계)")
    ax.legend(fontsize=8, ncol=1, loc="lower right")
    save(fig, f"core_inferflops_vs_acc{suffix}")

    # 그림 4: 데이터 효율 (초기 학습곡선, 가장 희소한 밀도)
    if d_show is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for arm in present:
            i = cidx[arm]
            rs = data[d_show].get(arm)
            if not rs:
                continue
            curves = []
            for r in rs:
                c = [x for x in r["curve"] if x.get("phase") == "main"]
                curves.append(([x["samples_seen"] for x in c], [x["test_acc"] for x in c]))
            n = min(len(c[0]) for c in curves)
            xs = np.array(curves[0][0][:n])
            ys = np.mean([c[1][:n] for c in curves], axis=0)
            m = xs <= (240000 if root == "core" else 400000)
            ax.plot(xs[m], ys[m], color=style(arm)[0], label=ARM_SHORT.get(arm, arm), linestyle=style(arm)[1])
        ax.set_xlabel("training samples seen")
        ax.set_ylabel("test accuracy")
        ax.set_ylim(*ylim)
        ax.set_title(f"{dataset_label} 데이터 효율: 초기 학습곡선 (density {d_show:g}). 가지치기 전까지 과잉 망 곡선은 겹침")
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        save(fig, f"core_data_efficiency{suffix}")

    # 표: 규칙별 정확도 (밀도 x 규칙), 간결판
    rows = []
    for d in densities:
        row = [f"{d:g}"]
        for arm in ARM_ORDER:
            rs = data[d].get(arm)
            row.append(ms([r["final_acc"] for r in rs], "{:.4f}") if rs else "-")
        rows.append(row)
    write_table(f"core_acc_matrix{suffix}", ["density"] + [ARM_LABEL[a] for a in ARM_ORDER], rows)


# ---------------------------------------------------------------------------
# 실험 1
# ---------------------------------------------------------------------------

def analyze_exp1():
    for ds in ("mnist", "cifar10"):
        groups = defaultdict(list)
        for f in glob.glob(os.path.join(RES, "exp1", ds, "*", "seed*.json")):
            r = load_json(f)
            groups[(r["mode"], r["keep"])].append(r)
        if not groups:
            continue
        rows = []
        order = sorted(groups, key=lambda k: ({"full": 0, "post": 1, "pre": 2}[k[0]], -k[1]))
        # 총 FLOPs = full 모델 측정값 - depth x dense 어텐션 + depth x 모드별 어텐션. 모든 모드에 같은 기준(full 측정값)을 쓴다.
        # (run.py 의 flops_total_exploited 는 pre 모드에서 라우팅된 모델의 측정값을 기준으로 삼아 절감을 두 번 세었다.)
        full_meas = groups[("full", 1.0)][0]["flops_total_measured"] if ("full", 1.0) in groups else None

        def total_flops(r):
            if full_meas is None:
                return r["flops_total_exploited"]
            return full_meas - r["depth"] * r["attn_flops_full_per_block"] + r["depth"] * r["attn_flops_per_block_exploited"]
        for k in order:
            rs = groups[k]
            r0 = rs[0]
            rows.append([k[0], f"{k[1]:g}", r0["k"], len(rs), ms([r["final_acc"] for r in rs]),
                         ms([r["calibration"]["ece"] for r in rs], "{:.3f}"),
                         sci(r0["attn_flops_per_block_dense"]), sci(r0["attn_flops_per_block_exploited"]),
                         f"{total_flops(r0) / full_meas:.3f}" if full_meas else "-"])
        if full_meas:
            share = order and groups[order[0]][0]["depth"] * groups[order[0]][0]["attn_flops_full_per_block"] / full_meas
            print(f"[exp1 {ds}] full model FLOPs {full_meas:.3e}, attention share of total {share:.3f}")
        write_table(f"exp1_{ds}", ["mode", "keep ratio", "k (keys/query)", "seeds", "test acc", "ECE",
                                   "attn FLOPs/block (dense)", "attn FLOPs/block (sparsity exploited)", "total FLOPs ratio"],
                    rows, f"토큰 수 {rows and groups[order[0]][0]['n_tokens']}. post = 사후 top-k 마스킹, pre = 저차원 라우터 사전 선택 (r=8). "
                          "총 FLOPs 비율은 full 대비, 어텐션 부분만 모드별 해석값으로 치환.")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for i, mode in enumerate(("post", "pre")):
            pts = sorted([(k[1], groups[k]) for k in groups if k[0] == mode], key=lambda t: t[0])
            if not pts:
                continue
            xs = [np.mean([r["attn_flops_per_block_exploited"] for r in rs]) for _, rs in pts]
            ys = [np.mean([r["final_acc"] for r in rs]) for _, rs in pts]
            es = [np.std([r["final_acc"] for r in rs]) for _, rs in pts]
            axes[0].errorbar(xs, ys, yerr=es, marker="o", color=SERIES[i + 1], label=mode, capsize=2)
            xt = [np.mean([total_flops(r) for r in rs]) for _, rs in pts]
            axes[1].errorbar(xt, ys, yerr=es, marker="o", color=SERIES[i + 1], label=mode, capsize=2)
            for (kr, rs), x, y in zip(pts, xs, ys):
                axes[0].annotate(f"k={kr:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, color=INK2)
        full = groups.get(("full", 1.0))
        if full:
            fa = np.mean([r["final_acc"] for r in full])
            axes[0].errorbar([full[0]["attn_flops_per_block_dense"]], [fa], yerr=[np.std([r["final_acc"] for r in full])],
                             marker="s", color=SERIES[0], label="full", capsize=2)
            axes[1].errorbar([full[0]["flops_total_measured"]], [fa], marker="s", color=SERIES[0], label="full")
        axes[0].set_xlabel("attention FLOPs per block (sparsity exploited)")
        axes[0].set_ylabel("test accuracy")
        axes[0].set_title(f"{ds}: 어텐션 연산량 대 정확도")
        axes[1].set_xlabel("total FLOPs per sample")
        axes[1].set_title(f"{ds}: 총 연산량 대 정확도")
        axes[0].legend()
        save(fig, f"exp1_{ds}")


# ---------------------------------------------------------------------------
# 실험 2
# ---------------------------------------------------------------------------

def analyze_exp2():
    for ds in ("mnist", "cifar10"):
        files = glob.glob(os.path.join(RES, "exp2", ds, "seed*.json"))
        if not files:
            continue
        rs = [load_json(f) for f in files]
        r0 = rs[0]
        base = np.mean([r["baseline"]["acc"] for r in rs])
        rows = [["baseline", "-", "0", ms([r["baseline"]["acc"] for r in rs]), ms([r["baseline"]["ece"] for r in rs], "{:.3f}"), "1.000"]]
        keys = defaultdict(list)
        for r in rs:
            for row in r["rows"]:
                keys[(row["score"], row["substitute"], row["skip_frac"])].append(row)
        for k in sorted(keys, key=lambda t: ({"predicted": 0, "random": 1, "oracle": 2}[t[0]], t[1], t[2])):
            v = keys[k]
            rows.append([k[0], k[1], f"{k[2]:.1f}", ms([x["acc"] for x in v]), ms([x["ece"] for x in v], "{:.3f}"),
                         f"{np.mean([x['flops_ratio'] for x in v]):.3f}"])
        write_table(f"exp2_{ds}", ["score", "substitute", "skip frac", "test acc", "ECE", "FLOPs ratio"], rows,
                    f"예측기 상대 MSE (블록별): {[round(x, 3) for x in (r0.get('predictor_rel_mse') or [])]}. "
                    "oracle 은 실제 잔차 크기로 고른 상한 (실제 절감 없음).")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for i, (score, sub) in enumerate([("predicted", "identity"), ("predicted", "predicted"), ("random", "identity"),
                                          ("oracle", "identity"), ("predicted", "identity_late_only")]):
            pts = sorted([(k[2], keys[k]) for k in keys if k[0] == score and k[1] == sub])
            if not pts:
                continue
            xs = [np.mean([x["flops_ratio"] for x in v]) for _, v in pts]
            ys = [np.mean([x["acc"] for x in v]) for _, v in pts]
            ax.plot(xs, ys, marker="o", color=SERIES[i], label=f"{score} / {sub}")
        ax.axhline(base, color=MUTED, linewidth=1, linestyle=":")
        ax.set_xlabel("FLOPs ratio vs full model")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"{ds}: 토큰 스킵 비율에 따른 정확도 (스킵 비율 0.1~0.8)")
        ax.legend(fontsize=8)
        save(fig, f"exp2_{ds}")


# ---------------------------------------------------------------------------
# 실험 3
# ---------------------------------------------------------------------------

METHOD_LABEL = {
    "finetune": "fine-tune (no protection)", "joint": "joint retraining (upper bound)", "er": "experience replay (buffer 500)",
    "er_x2ep": "experience replay, 2x epochs", "cls_plain": "CLS + sleep (replay + distill)", "cls_nokd": "CLS + sleep (no distill)",
    "cls_down": "CLS + sleep + downscale 0.1", "cls_prune": "CLS + sleep + prune 5%/sleep (magnitude)",
    "cls_pruneact": "CLS + sleep + prune 5%/sleep (activity)", "cls_full": "CLS + sleep + downscale + prune",
    "er_balanced": "ER, class-balanced batches (buffer 500)",
    "cls2_dense": "CLS2: balanced sleep replay, no distill, dense fast",
    "cls2_kwta10": "CLS2 + sparse fast (k-WTA 10%)", "cls2_kwta5": "CLS2 + sparse fast (k-WTA 5%)",
    "cls2_kwta10_prune": "CLS2 + sparse fast 10% + magnitude prune 5%/sleep",
    "cls2_kwta10_lr": "CLS2 + sparse fast 10%, sleep 300 steps lr 1e-3",
    "cls2_decay_boundary": "CLS2 + sleep decay: one-shot 10% at boundary",
    "cls2_decay_periodic": "CLS2 + sleep decay: periodic gradual, total 10%",
    "cls2_decay_periodic_small": "CLS2 + sleep decay: periodic gradual, total 3%",
    "cls2_decay_continuous": "CLS2 + sleep decay: continuous weight decay 1e-4",
    "cls2_kd_none_h256": "CLS2 (fast width 256), no distillation",
    "cls2_kd_global": "CLS2 (fast 256) + global logit distillation",
    "cls2_kd_local": "CLS2 (fast 256) + local per-layer feature distillation",
    "cls2_kd_local_logits": "CLS2 (fast 256) + local features + logits",
}
METHOD_ORDER = ["finetune", "er", "er_x2ep", "er_balanced", "cls_plain", "cls_nokd", "cls_down", "cls_prune", "cls_pruneact",
                "cls_full", "cls2_dense", "cls2_kwta10", "cls2_kwta5", "cls2_kwta10_prune", "cls2_kwta10_lr",
                "cls2_decay_boundary", "cls2_decay_periodic", "cls2_decay_periodic_small", "cls2_decay_continuous",
                "cls2_kd_none_h256", "cls2_kd_global", "cls2_kd_local", "cls2_kd_local_logits", "joint"]


def analyze_exp3():
    for ds in ("split_mnist", "permuted_mnist"):
        groups = defaultdict(list)
        for f in glob.glob(os.path.join(RES, "exp3", ds, "*", "seed*.json")):
            r = load_json(f)
            name = r["method"] + (f"_{r['tag']}" if r["tag"] else "")
            groups[name].append(r)
        if not groups:
            continue
        rows = []
        for name in METHOD_ORDER:
            rs = groups.get(name)
            if not rs:
                continue
            unk = [np.mean([u["softmax_max"] for u in r["unknown_auroc"] if u.get("softmax_max") is not None])
                   for r in rs if r["unknown_auroc"]]
            agree = [np.mean([u["fast_slow_agreement"] for u in r["unknown_auroc"]])
                     for r in rs if r["unknown_auroc"] and "fast_slow_agreement" in r["unknown_auroc"][0]]
            for alt in sorted(set(k for r in rs for k in (r.get("continual_alt") or {}))):
                vals = [r["continual_alt"][alt] for r in rs if alt in (r.get("continual_alt") or {})]
                rows.append([METHOD_LABEL.get(name, name) + f" [predict: {alt}]", len(vals), ms([v["avg_acc"] for v in vals]),
                             ms([v["forgetting"] for v in vals]), ms([v.get("retention") for v in vals], "{:.3f}"),
                             "-", "-", "-", "-", "-"])
            rows.append([METHOD_LABEL.get(name, name), len(rs), ms([r["continual"]["avg_acc"] for r in rs]),
                         ms([r["continual"]["forgetting"] for r in rs]), ms([r["continual"].get("retention") for r in rs], "{:.3f}"),
                         ms([r["calibration"]["ece"] for r in rs], "{:.3f}"), sci(np.mean([r["cost"]["train_flops"] for r in rs])),
                         f"{int(np.mean([r['cost']['params_active'] for r in rs])):,}",
                         ms(unk, "{:.3f}") if unk else "-", ms(agree, "{:.3f}") if agree else "-"])
        write_table(f"exp3_{ds}", ["method", "seeds", "avg acc (final)", "forgetting", "retention", "ECE (all classes)",
                                   "train FLOPs", "active params (predictor)", "unknown-class AUROC (softmax)",
                                   "unknown-class AUROC (fast/slow agreement)"], rows,
                    "Split MNIST 는 Class-IL (단일 헤드 10 출력, 5 태스크 x 2 클래스), Permuted 는 10 태스크. "
                    "unknown-class AUROC: 각 시점에서 아직 안 배운 클래스의 테스트 표본을 확신도로 가려내는 성능 (0.5 = 못 가림).")
        # 그림: 태스크 진행에 따른 평균 정확도
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for i, name in enumerate(METHOD_ORDER):
            rs = groups.get(name)
            if not rs:
                continue
            Rs = np.array([r["R"] for r in rs])
            T = Rs.shape[1]
            avg = [Rs[:, t, :t + 1].mean(axis=1).mean() for t in range(T)]
            axes[0].plot(range(1, T + 1), avg, marker="o", color=SERIES[i % 8], linestyle="-" if i < 8 else "--",
                         label=METHOD_LABEL.get(name, name))
            # 첫 태스크 정확도 유지
            axes[1].plot(range(1, T + 1), Rs[:, :, 0].mean(axis=0), marker="o", color=SERIES[i % 8],
                         linestyle="-" if i < 8 else "--")
        axes[0].set_xlabel("tasks learned")
        axes[0].set_ylabel("mean accuracy on learned tasks")
        axes[0].set_title(f"{ds}: 학습 진행에 따른 평균 정확도")
        axes[1].set_xlabel("tasks learned")
        axes[1].set_ylabel("accuracy on task 1")
        axes[1].set_title(f"{ds}: 첫 태스크 정확도 (망각)")
        axes[0].legend(fontsize=7)
        save(fig, f"exp3_{ds}")


# ---------------------------------------------------------------------------
# 실험 4
# ---------------------------------------------------------------------------

def mnist_background_fraction(thresh: float = 0.01) -> dict:
    """STDP 희소화의 대조: 클래스 평균 이미지에서 배경 (최대값의 thresh 미만) 픽셀 비율. 뉴런이 숫자 프로토타입으로 수렴하면
    그 클래스의 배경 픽셀 가중치는 자연히 감쇠하므로, 이 비율이 프로토타입 형성만으로 설명되는 희소화의 상한이다."""
    from torchvision import datasets
    ds = datasets.MNIST(os.path.join(os.path.dirname(RES), "data"), train=True, download=False)
    x = ds.data.numpy().astype(np.float64) / 255.0
    y = ds.targets.numpy()
    per_class = []
    for c in range(10):
        m = x[y == c].mean(0)
        per_class.append(float((m < thresh * m.max()).mean()))
    g = x.mean(0)
    return {"class_mean_below_1pct": float(np.mean(per_class)), "per_class_min": float(min(per_class)),
            "per_class_max": float(max(per_class)), "global_mean_below_1pct": float((g < thresh * g.max()).mean()),
            "zero_in_99pct": float(((x == 0).mean(0) >= 0.99).mean())}


def analyze_exp4():
    groups = defaultdict(list)
    for f in glob.glob(os.path.join(RES, "exp4", "n*", "seed*.json")):
        r = load_json(f)
        groups[r["snn_cfg"]["n_e"]].append(r)
    if not groups:
        return
    rows = []
    for n_e in sorted(groups):
        rs = groups[n_e]
        rows.append([n_e, len(rs), ms([r["final_acc"] for r in rs]), ms([r["backprop_ref"]["final_acc"] for r in rs]),
                     sci(np.mean([r["ops"]["inference_per_image"]["total"] for r in rs])),
                     sci(rs[0]["backprop_ref"]["infer_flops"]),
                     sci(np.mean([r["ops"]["train_total_ops"] for r in rs])), sci(rs[0]["backprop_ref"]["train_flops_total"]),
                     ms([r["data_efficiency_80"]["snn"]["samples_to_target"] for r in rs], "{:.0f}"),
                     ms([r["data_efficiency_80"]["backprop"]["samples_to_target"] for r in rs], "{:.0f}"),
                     ms([r["weight_stats"][-1]["frac_below_1pct"] for r in rs], "{:.3f}"),
                     ms([r["train_time_s"] for r in rs], "{:.0f}")])
    bg = mnist_background_fraction()
    rows.append(["control: MNIST class-mean background pixels", "-", "-", "-", "-", "-", "-", "-", "-", "-",
                 f"{bg['class_mean_below_1pct']:.3f} (per class {bg['per_class_min']:.2f}-{bg['per_class_max']:.2f}; "
                 f"zero in 99% of images {bg['zero_in_99pct']:.3f})", "-"])
    write_table("exp4", ["n_e", "seeds", "STDP acc", "backprop MLP acc (same width, same samples)", "SNN infer SOP/img",
                         "MLP infer FLOPs/img", "SNN train ops (total)", "MLP train FLOPs (total)", "SNN samples to 80%",
                         "MLP samples to 80%", "weights < 1% wmax (final)", "train time (s)"], rows,
                "SOP = 사건 구동 시냅스 연산 (누산). 에너지 비교는 하지 않는다 (GPU 에서 실행). "
                "backprop MLP 는 784-n_e-10, Adam, 같은 표본을 한 번 훑음. "
                f"배경 픽셀 대조 (프로토타입 형성만으로 설명되는 희소화의 상한): MNIST 클래스 평균 이미지에서 최대값의 1% 미만인 픽셀 비율 "
                f"{bg['class_mean_below_1pct']:.3f} (클래스별 {bg['per_class_min']:.3f}~{bg['per_class_max']:.3f}), "
                f"전체 평균 이미지 {bg['global_mean_below_1pct']:.3f}, 학습 이미지의 99% 이상에서 0 인 픽셀 {bg['zero_in_99pct']:.3f}.")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, n_e in enumerate(sorted(groups)):
        rs = sorted(groups[n_e], key=lambda r: -len(r["curve"]))   # 평가점이 가장 촘촘한 시드를 앞에
        # 시드마다 평가 간격이 다를 수 있어 평균하지 않고 가장 촘촘한 시드 하나를 그린다
        c = rs[0]["curve"]
        axes[0].plot([x["samples_seen"] for x in c], [x["test_acc"] for x in c],
                     marker="o", color=SERIES[i % 8], label=f"STDP n_e={n_e} (seed {rs[0]['seed']})")
        b = rs[0]["backprop_ref"]["curve"]
        axes[0].plot([x["samples_seen"] for x in b], [x["test_acc"] for x in b],
                     linestyle="--", color=SERIES[i % 8], label=f"backprop MLP h={n_e}")
        w = rs[0]["weight_stats"]
        axes[1].plot([x["samples_seen"] for x in w], [x["frac_below_1pct"] for x in w], color=SERIES[i % 8],
                     label=f"n_e={n_e}: weights < 1% wmax")
        axes[1].plot([x["samples_seen"] for x in w], [x["gini"] for x in w], color=SERIES[i % 8], linestyle=":",
                     label=f"n_e={n_e}: Gini")
    axes[0].set_xlabel("training samples seen")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("STDP 대 역전파: 데이터 효율")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("training samples seen")
    axes[1].set_ylabel("fraction / Gini")
    axes[1].set_title("STDP 가중치 분포 궤적 (가지치기 관찰)")
    axes[1].legend(fontsize=7)
    save(fig, "exp4")


def analyze_exp2_finetune():
    """실험 2 보강: 스킵을 켠 채 미세조정한 뒤의 정확도 (학습 시 적응). 파일: results/exp2/<ds>/finetune_*.json"""
    for ds in ("mnist", "cifar10"):
        files = glob.glob(os.path.join(RES, "exp2", ds, "finetune_*_seed*.json"))
        if not files:
            continue
        rs = [load_json(f) for f in files]
        groups = defaultdict(list)
        for r in rs:
            groups[r["train_score"]].append(r)
        fracs = [0.0, 0.2, 0.3, 0.5, 0.7, 0.8]
        rows = []
        for train_score in ("predicted", "random"):
            g = groups.get(train_score)
            if not g:
                continue
            for eval_score in ("predicted", "random"):
                cells = []
                for s_ in fracs:
                    vals = []
                    for r in g:
                        for row in r["rows"]:
                            if row["skip_frac"] == s_ and (row["eval_score"] == eval_score or s_ == 0.0):
                                vals.append(row["acc"])
                                break
                    cells.append(ms(vals) if vals else "-")
                before = ms([r["before_finetune_skipped"]["acc"] for r in g]) if eval_score == train_score else "-"
                rows.append([f"{train_score} 기준 미세조정 ({len(g)} 시드)", eval_score, before] + cells)
        full = ms([r["baseline_full"]["acc"] for r in rs])
        write_table(f"exp2_finetune_{ds}", ["학습 시 스킵 기준", "평가 시 스킵 기준", "미세조정 전 (스킵 0.5)"] + [f"스킵 {f:g}" for f in fracs],
                    rows, f"원본 모델(스킵 없음) {full}. 미세조정은 스킵 0.5 로 {rs[0]['epochs']} 에폭, ViT 와 예측기를 함께 학습.")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        i = 0
        for train_score in ("predicted", "random"):
            g = groups.get(train_score)
            if not g:
                continue
            for eval_score in ("predicted", "random"):
                xs, ys = [], []
                for s_ in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                    vals = [row["acc"] for r in g for row in r["rows"] if row["skip_frac"] == s_ and row["eval_score"] == eval_score]
                    if vals:
                        xs.append(s_)
                        ys.append(np.mean(vals))
                ax.plot(xs, ys, marker="o", color=SERIES[i % 8], linestyle="-" if train_score == eval_score else "--",
                        label=f"train {train_score} / eval {eval_score}")
                i += 1
        # 사후 적용 (미세조정 없음) 참고선
        post = glob.glob(os.path.join(RES, "exp2", ds, "seed*.json"))
        if post:
            prs = [load_json(f) for f in post]
            for j, (score, color) in enumerate((("predicted", SERIES[4]), ("random", SERIES[5]))):
                xs, ys = [], []
                for s_ in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                    vals = [row["acc"] for r in prs for row in r["rows"] if row["score"] == score and row["substitute"] == "identity" and row["skip_frac"] == s_]
                    if vals:
                        xs.append(s_)
                        ys.append(np.mean(vals))
                ax.plot(xs, ys, marker="s", color=color, linestyle=":", label=f"no finetune / {score}")
        ax.axhline(np.mean([r["baseline_full"]["acc"] for r in rs]), color=MUTED, linewidth=1, linestyle=":")
        ax.set_xlabel("skip fraction (uniform over all blocks)")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"{ds}: 스킵을 켠 채 미세조정한 뒤의 정확도 (실선 = 학습과 같은 기준으로 평가)")
        ax.legend(fontsize=7, ncol=2, loc="lower left")
        save(fig, f"exp2_finetune_{ds}")


# ---------------------------------------------------------------------------
# 실험 A: 사전 학습 ResNet-18 -> CIFAR-10 적응 중 가지치기 (공짜 대리석)
# ---------------------------------------------------------------------------
PT_LABEL = {
    "pt_dense": "pre-trained, dense fine-tune (no pruning)",
    "pt_pd": "pre-trained, prune-during adaptation (global magnitude)",
    "pt_pd_erk": "pre-trained, prune-during adaptation (ERK layer budget)",
    "pt_oneshot": "pre-trained, one-shot prune then fine-tune",
    "pt_rigl": "pre-trained, random ERK mask + RigL regrow",
    "pt_rigl_mag": "pre-trained, magnitude mask + RigL regrow (keeps inherited structure)",
    "pt_pd_end50": "pre-trained, prune-during adaptation, pruning ends at 50% of adaptation (global magnitude)",
    "scratch_pd": "from scratch, prune-during (global magnitude)",
    "scratch_small": "from scratch, dense small (width-scaled)",
}
PT_ORDER = ["scratch_small", "scratch_pd", "pt_oneshot", "pt_rigl", "pt_rigl_mag", "pt_pd_erk", "pt_pd", "pt_pd_end50"]
# 범례용 짧은 라벨과 고정 색 (실선 = 사전 학습 출발, 점선 = 처음부터)
PT_SHORT = {
    "pt_dense": "pre-trained, dense fine-tune", "pt_pd": "pre-trained, prune-during (global)",
    "pt_pd_erk": "pre-trained, prune-during (ERK)", "pt_pd_end50": "pre-trained, prune-during, schedule ends at 50%",
    "pt_oneshot": "pre-trained, one-shot prune + fine-tune", "pt_rigl": "pre-trained, random mask + RigL",
    "pt_rigl_mag": "pre-trained, magnitude mask + RigL", "scratch_pd": "from scratch, prune-during",
    "scratch_small": "from scratch, dense small",
}
PT_COLOR = {"scratch_small": "#2a78d6", "scratch_pd": "#eb6834", "pt_oneshot": "#1baf7a", "pt_rigl": "#eda100",
            "pt_rigl_mag": "#e87ba4", "pt_pd_erk": "#008300", "pt_pd": "#4a3aa7", "pt_pd_end50": "#e34948", "pt_dense": "#303030"}


def analyze_core_pretrained():
    data = defaultdict(lambda: defaultdict(list))  # density -> arm -> [runs]
    for f in glob.glob(os.path.join(RES, "core_pretrained", "d*", "*", "seed*.json")):
        r = load_json(f)
        data[r["density"]][r["arm"]].append(r)
    if not data:
        return
    dense = data.get(1.0, {}).get("pt_dense", [])
    densities = sorted(d for d in data if d < 1.0)
    present = [a for a in PT_ORDER if any(a in data[d] for d in densities)]
    cidx = {a: i for i, a in enumerate(present)}
    rows = []
    for d in sorted(data, reverse=True):
        for arm in ["pt_dense"] + PT_ORDER:
            rs = data[d].get(arm)
            if not rs:
                continue
            de = [r.get("data_efficiency_90") or {} for r in rs]
            stt = [x.get("samples_to_target") for x in de if x.get("samples_to_target") is not None]
            if not stt:
                stt_s = "-"
            elif len(stt) == len(rs):
                stt_s = ms(stt, "{:.0f}")
            else:
                stt_s = ms(stt, "{:.0f}") + f" ({len(stt)}/{len(rs)} reached)"
            rows.append([f"{d:g}", PT_LABEL.get(arm, arm), len(rs), f"{int(np.mean([r['final_active'] for r in rs])):,}",
                         ms([r["init_acc"] for r in rs]), ms([r["final_acc"] for r in rs]), ms([r["best_acc"] for r in rs]),
                         ms([r["calibration"]["ece"] for r in rs], "{:.3f}"), sci(np.mean([r["infer_flops"] for r in rs])),
                         sci(np.mean([r["adapt_train_flops"] for r in rs])), stt_s])
    write_table("core_pretrained_summary", ["density", "arm", "seeds", "active conv weights", "init acc", "final acc", "best acc", "ECE",
                                            "infer FLOPs", "adaptation train FLOPs", "samples to 90%"], rows,
                "density 는 ImageNet 사전 학습 ResNet-18 의 conv 가중치(11,166,912) 대비 최종 활성 비율. 분류층(fc)은 모든 팔에서 밀집 유지. "
                "입력은 128x128 로 키운 CIFAR-10, 10 에폭 적응. init acc 는 적응 전(새 분류층) 정확도. "
                "적응 FLOPs 는 적응 10 에폭만 센 값이며 ImageNet 사전 학습 비용은 모든 pt_ 팔에 공통으로 들어 표에 넣지 않았다.")

    # 그림 1: 정확도 대 활성 연결 수
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in present:
        i = cidx[arm]
        xs, ys, es = [], [], []
        for d in densities:
            rs = data[d].get(arm)
            if rs:
                xs.append(np.mean([r["final_active"] for r in rs]))
                ys.append(np.mean([r["final_acc"] for r in rs]))
                es.append(np.std([r["final_acc"] for r in rs]))
        if xs:
            ax.errorbar(xs, ys, yerr=es, label=PT_SHORT.get(arm, arm), color=PT_COLOR.get(arm, "#303030"), marker="o", capsize=2,
                        linestyle="--" if arm.startswith("scratch") else "-")
    ax.set_xscale("log")
    if dense:
        acc = np.mean([r["final_acc"] for r in dense])
        ax.axhline(acc, color=MUTED, linewidth=1, linestyle=":")
        ax.text(0.01, acc + 0.002, "pre-trained dense fine-tune (11.2M conv weights)", color=INK2, fontsize=8,
                transform=ax.get_yaxis_transform())
    ax.set_xlabel("final active conv weights")
    ax.set_ylabel("CIFAR-10 test accuracy (128x128 input)")
    ax.set_title("실험 A: 사전 학습 ResNet-18 의 적응 중 가지치기 - 정확도 대 활성 연결 수 (시드 평균)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    save(fig, "core_pretrained_acc_vs_active")

    # 그림 2: 적응 학습 FLOPs 대 정확도
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in present:
        i = cidx[arm]
        xs, ys = [], []
        for d in densities:
            rs = data[d].get(arm)
            if rs:
                xs.append(np.mean([r["adapt_train_flops"] for r in rs]))
                ys.append(np.mean([r["final_acc"] for r in rs]))
        if xs:
            ax.plot(xs, ys, marker="o", color=PT_COLOR.get(arm, "#303030"), label=PT_SHORT.get(arm, arm),
                    linestyle="--" if arm.startswith("scratch") else "-")
    if dense:
        ax.scatter([np.mean([r["adapt_train_flops"] for r in dense])], [np.mean([r["final_acc"] for r in dense])],
                   color=MUTED, marker="s", label=PT_LABEL["pt_dense"], zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("adaptation training FLOPs (forward x3, actual density per step; ImageNet pre-training excluded)")
    ax.set_ylabel("CIFAR-10 test accuracy")
    ax.set_title("실험 A: 적응 비용 대 정확도 (점 = 밀도 단계)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "core_pretrained_trainflops_vs_acc")

    # 그림 3: 궤적 (가장 희소한 밀도, seed 0)
    if densities:
        d_show = min(densities)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for arm in present:
            i = cidx[arm]
            rs = data[d_show].get(arm)
            if not rs:
                continue
            r = sorted(rs, key=lambda x: x["seed"])[0]
            c = [x for x in r["curve"] if x.get("phase") in ("main", "finetune", "post_prune", "pre_prune", "final", "init")]
            ls = "--" if arm.startswith("scratch") else "-"
            axes[0].plot([x["step"] for x in c], [x["active"] for x in c], color=SERIES[i % 8], label=PT_LABEL[arm], linestyle=ls)
            axes[1].plot([x["step"] for x in c], [x["test_acc"] for x in c], color=SERIES[i % 8], linestyle=ls)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("adaptation step")
        axes[0].set_ylabel("active conv weights")
        axes[0].set_title(f"연결 수 궤적 (density {d_show:g}, seed 0)")
        axes[1].set_xlabel("adaptation step")
        axes[1].set_ylabel("test accuracy")
        axes[1].set_title("정확도 궤적")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False)
        fig.tight_layout(rect=[0, 0.16, 1, 1])
        fig.savefig(os.path.join(FIG, "core_pretrained_trajectory.png"), dpi=150)
        plt.close(fig)
        print("[figure] core_pretrained_trajectory")

    # 표: 밀도 x 팔 정확도 행렬
    rows = []
    for d in densities:
        row = [f"{d:g}"]
        for arm in PT_ORDER:
            rs = data[d].get(arm)
            row.append(ms([r["final_acc"] for r in rs], "{:.4f}") if rs else "-")
        rows.append(row)
    write_table("core_pretrained_acc_matrix", ["density"] + [PT_LABEL[a] for a in PT_ORDER], rows)


# ---------------------------------------------------------------------------
# 논문 2 토대: 수면 감쇠 스케줄 + 분산(국소) 증류 (실험 3 변형들만 따로 비교)
# ---------------------------------------------------------------------------
PAPER2_GROUPS = [
    ("sleep decay variants", "cls2_kwta10",
     ["cls2_decay_boundary", "cls2_decay_periodic", "cls2_decay_periodic_small", "cls2_decay_continuous"]),
    ("distillation locality (fast width 256)", "cls2_kd_none_h256",
     ["cls2_kd_global", "cls2_kd_local", "cls2_kd_local_logits"]),
]


def analyze_paper2():
    for ds in ("split_mnist", "permuted_mnist"):
        groups = defaultdict(list)
        for f in glob.glob(os.path.join(RES, "exp3", ds, "*", "seed*.json")):
            r = load_json(f)
            groups[r["method"] + (f"_{r['tag']}" if r["tag"] else "")].append(r)
        if not any(v in groups for _, _, vs in PAPER2_GROUPS for v in vs):
            continue
        rows = []
        fig, axes = plt.subplots(1, len(PAPER2_GROUPS), figsize=(12, 4.2))
        for gi, (gname, ref, variants) in enumerate(PAPER2_GROUPS):
            names = [ref] + variants
            ref_rs = groups.get(ref, [])
            ref_acc = np.mean([r["continual"]["avg_acc"] for r in ref_rs]) if ref_rs else None
            ax = axes[gi]
            for ni, name in enumerate(names):
                rs = groups.get(name)
                if not rs:
                    continue
                accs = [r["continual"]["avg_acc"] for r in rs]
                delta = "-" if (ref_acc is None or name == ref) else f"{(np.mean(accs) - ref_acc) * 100:+.2f}"
                rows.append([gname, METHOD_LABEL.get(name, name) + (" (reference)" if name == ref else ""), len(rs),
                             ms(accs), delta, ms([r["continual"]["forgetting"] for r in rs]),
                             ms([r["continual"].get("retention") for r in rs], "{:.3f}"),
                             ms([r["calibration"]["ece"] for r in rs], "{:.3f}"), sci(np.mean([r["cost"]["train_flops"] for r in rs]))])
                ax.bar(ni, np.mean(accs), yerr=np.std(accs), color=SERIES[ni % 8], width=0.7, capsize=3)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([n.replace("cls2_", "").replace("_", " ") for n in names], fontsize=7, rotation=15)
            ax.set_ylabel("final average accuracy")
            ax.set_title(f"{ds}: {gname}")
            vals = [np.mean([r["continual"]["avg_acc"] for r in groups[n]]) for n in names if n in groups]
            if vals:
                ax.set_ylim(max(0.0, min(vals) - 0.08), min(1.0, max(vals) + 0.04))
        write_table(f"paper2_{ds}", ["group", "method", "seeds", "avg acc (final)", "delta vs ref (pp)", "forgetting", "retention",
                                     "ECE", "train FLOPs"], rows,
                    "논문 2 토대. 수면 감쇠: 경계 1 회 하향(0.1) / 수면 중 50 스텝마다 점진(총 10% 또는 3%) / 수면 중 가중치 감쇠 1e-4 를 "
                    "같은 설정의 CLS2(k-WTA 10%, 감쇠 없음)와 비교. 증류: 같은 폭(256)의 무증류 기준 대비 전역 로짓 증류 / 층별 국소 특징 증류 / 국소+로짓.")
        fig.tight_layout()
        save(fig, f"paper2_{ds}")


# ---------------------------------------------------------------------------
# 실험 5: 추론 중 가지치기 (입력 의존적 동적 희소화) - 정적 대조군은 core_resnet 에서 가져온다
# ---------------------------------------------------------------------------
EXP5_LABEL = {
    "dyn_local": "dynamic, connection-level, local rule (abs(w) x input magnitude)",
    "dyn_random": "dynamic, connection-level, random selection",
    "kwta_in": "dynamic, channel-level k-WTA input gating",
    "pd_mag_global": "static: prune-during, magnitude (global)",
    "pd_mag_erk": "static: prune-during, magnitude (ERK)",
    "dense_small": "static: dense small (additive)",
    "rigl": "static: RigL",
    "ttp": "static: prune-after, one-shot magnitude + fine-tune",
}
EXP5_ORDER = ["dense_small", "ttp", "rigl", "pd_mag_erk", "pd_mag_global", "kwta_in", "dyn_random", "dyn_local"]


def analyze_exp5():
    data = defaultdict(lambda: defaultdict(list))  # density -> arm -> [runs]
    for f in glob.glob(os.path.join(RES, "exp5_dynamic", "d*", "*", "seed*.json")):
        r = load_json(f)
        data[r["density"]][r["arm"]].append(r)
    if not data:
        return
    densities = sorted(data)
    for f in glob.glob(os.path.join(RES, "core_resnet", "d*", "*", "seed*.json")):
        r = load_json(f)
        if r["density"] in data and r["arm"] in EXP5_LABEL:
            data[r["density"]][r["arm"]].append(r)
    rows = []
    for d in sorted(densities, reverse=True):
        for arm in EXP5_ORDER:
            rs = data[d].get(arm)
            if not rs:
                continue
            st = [r.get("mask_stats", {}).get("layers.7.conv2") for r in rs]
            st = [x for x in st if x]
            rows.append([f"{d:g}", EXP5_LABEL[arm], len(rs), f"{int(np.mean([r['final_active'] for r in rs])):,}",
                         ms([r["final_acc"] for r in rs]), ms([r["calibration"]["ece"] for r in rs], "{:.3f}"),
                         sci(np.mean([r["infer_flops"] for r in rs])), sci(np.mean([r["cum_train_flops"] for r in rs])),
                         ms([x["jaccard_same_class"] for x in st], "{:.3f}") if st else "-",
                         ms([x["jaccard_diff_class"] for x in st], "{:.3f}") if st else "-",
                         ms([x["union_coverage"] for x in st], "{:.3f}") if st else "-"])
    write_table("exp5_dynamic", ["density", "arm", "seeds", "active weights (per sample)", "test acc", "ECE", "infer FLOPs",
                                 "cum. train FLOPs", "stage-4 Jaccard same class", "Jaccard diff class", "union coverage"], rows,
                "CIFAR-10 ResNet-18. 동적 팔은 표본마다 다른 서브망을 쓰며 표본당 기대 활성 연결이 density 에 맞춰짐 (첫 conv, fc 는 밀집). "
                "정적 팔은 results/core_resnet 의 같은 밀도 결과. 자카드는 4 단계 블록 둘째 conv 의 표본별 마스크 유사도 "
                "(같은 클래스 쌍 / 다른 클래스 쌍, 무작위면 둘 다 약 density). union coverage 는 테스트 표본 200 개 중 하나라도 쓴 연결 비율.")
    # 그림: 정확도 대 추론 FLOPs
    fig, ax = plt.subplots(figsize=(8, 5))
    present = [a for a in EXP5_ORDER if any(a in data[d] for d in densities)]
    for i, arm in enumerate(present):
        xs, ys, es = [], [], []
        for d in densities:
            rs = data[d].get(arm)
            if rs:
                xs.append(np.mean([r["infer_flops"] for r in rs]))
                ys.append(np.mean([r["final_acc"] for r in rs]))
                es.append(np.std([r["final_acc"] for r in rs]))
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=2, color=SERIES[i % 8], label=EXP5_LABEL[arm],
                        linestyle="-" if arm.startswith(("dyn", "kwta")) else "--")
    ax.set_xscale("log")
    ax.set_xlabel("inference FLOPs per sample (expected)")
    ax.set_ylabel("CIFAR-10 test accuracy")
    ax.set_title("실험 5: 추론 중 가지치기 (실선 = 동적, 점선 = 정적 대조군)")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "exp5_dynamic")


ALL = {"baselines": analyze_baselines, "exp2_finetune": analyze_exp2_finetune,
       "core_cifar": lambda: analyze_core("core_cifar", "_cifar", (0.5, 0.95), "CIFAR-10", "2.2M"),
       "core_resnet": lambda: analyze_core("core_resnet", "_resnet", (0.6, 0.96), "CIFAR-10 ResNet-18", "11.2M"), "core": analyze_core, "exp1": analyze_exp1, "exp2": analyze_exp2,
       "exp3": analyze_exp3, "exp4": analyze_exp4, "core_pretrained": analyze_core_pretrained, "paper2": analyze_paper2, "exp5": analyze_exp5}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(ALL)
    for t in targets:
        try:
            ALL[t]()
        except Exception as e:  # 한 실험의 실패가 나머지를 막지 않게
            import traceback
            print(f"[error] {t}: {e}")
            traceback.print_exc()
