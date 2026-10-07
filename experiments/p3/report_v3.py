# -*- coding: utf-8 -*-
"""H2 v3 집계: results/p3/h3, h3long -> summary_v3.json, tables_v3.md, fig_v3.png, 예측 P1~P9 자동 판정.
지표는 시험 정확도(마지막 3 평가 평균, last3_test). 학습률·수락은 검증으로 골랐다. "확실히" = 평균 차 > 두 조건 표준편차 합 (ddof=0,
등록 규칙 그대로) — 표본 표준편차(ddof=1)로도 다시 판정해 함께 적는다."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO_ROOT, "results", "p3")
LEARNERS = ["bp", "dfa", "np", "fg"]
ARMS = [("dense", 1.0), ("pruned", 0.01), ("pruned_reinit", 0.01), ("random_degree", 0.01), ("random_mask", 0.01), ("dense_small", 0.01),
        ("pruned", 0.005), ("pruned_reinit", 0.005), ("random_degree", 0.005), ("random_mask", 0.005), ("dense_small", 0.005)]
LABEL = {"dense": "밀집 100%", "pruned": "학습 마스크", "pruned_reinit": "학습 마스크 + 다른 초기값", "random_degree": "차수 보존 무작위",
         "random_mask": "무작위 마스크", "dense_small": "같은 예산 작은 밀집망"}


def key(c, d):
    return f"{c}_d{d:g}"


def label(c, d):
    return LABEL[c] + ("" if c == "dense" else f" {100 * d:g}%")


def load(sub="h3"):
    out = {}
    for c, d in ARMS:
        for l in LEARNERS:
            rs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(RES, sub, key(c, d), l, "seed?.json")))]
            if rs:
                out[(c, d, l)] = rs
    return out


def stat(xs, ddof=0):
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=ddof) if a.size > ddof else 0.0), int(a.size)


def certain(a, b, ddof=0):
    """a > b 확실히? a, b 는 값 목록."""
    if len(a) < 2 or len(b) < 2:
        return None
    ma, sa, _ = stat(a, ddof)
    mb, sb, _ = stat(b, ddof)
    return (ma - mb) > (sa + sb)


def fmt(v):
    return "판단 불가" if v is None else ("맞음" if v else "틀림")


def main():
    R = load()
    acc = {k: [r["last3_test"] for r in rs] for k, rs in R.items()}
    cos = {k: float(np.mean([p["cos_each"] for r in rs for p in r["probes"]])) if any(r["probes"] for r in rs) else None for k, rs in R.items()}
    S = {"cells": {}, "judgement": {}}
    for (c, d, l), rs in R.items():
        m, s, n = stat(acc[(c, d, l)])
        S["cells"][f"{key(c, d)}/{l}"] = {
            "acc": m, "std": s, "n": n, "acc0": float(np.mean([r["acc0_test"] for r in rs])),
            "backoffs": float(np.mean([r["backoffs"] for r in rs])), "failed": int(sum(r["failed"] for r in rs)),
            "lr_selected": rs[0]["lr_selected"], "lr_final": [r["lr_final"] for r in rs], "n_perturbed": rs[0]["n_perturbed"],
            "alive_hidden": rs[0]["alive_hidden"], "cos_each": cos[(c, d, l)],
            "gap_to_bp": (stat(acc[(c, d, "bp")])[0] - m) if (c, d, "bp") in acc else None}

    def A(c, d, l):
        return acc.get((c, d, l), [])

    def M(c, d, l):
        return stat(A(c, d, l))[0] if A(c, d, l) else None

    J = S["judgement"]
    D = ("dense", 1.0)
    J["P1"] = {"bp>dfa": fmt(certain(A(*D, "bp"), A(*D, "dfa"))), "dfa>np": fmt(certain(A(*D, "dfa"), A(*D, "np"))),
               "np>fg": fmt(certain(A(*D, "np"), A(*D, "fg"))), "fg<=bp-10": fmt(M(*D, "fg") <= M(*D, "bp") - 0.10)}
    gaps = {lab: [b - x for b, x in zip(A(c, d, "bp"), A(c, d, "fg"))] for lab, (c, d) in
            (("dense", D), ("pruned 1%", ("pruned", 0.01)), ("pruned 0.5%", ("pruned", 0.005)))}
    J["P2_fg"] = {"gaps": {k: stat(v)[0] for k, v in gaps.items()},
                  "dense>0.5% 확실": fmt(certain(gaps["dense"], gaps["pruned 0.5%"])), "dense>1% 확실": fmt(certain(gaps["dense"], gaps["pruned 1%"])),
                  "단조": fmt(stat(gaps["dense"])[0] > stat(gaps["pruned 1%"])[0] > stat(gaps["pruned 0.5%"])[0])}
    J["P3"] = {f"{l} {100 * d:g}%": {"pruned": M("pruned", d, l), "small": M("dense_small", d, l),
                                      "pruned>small 확실": fmt(certain(A("pruned", d, l), A("dense_small", d, l))),
                                      "small>=pruned": fmt(M("dense_small", d, l) >= M("pruned", d, l))}
               for l in ("fg", "np", "dfa", "bp") for d in (0.01, 0.005)}
    J["P4"] = {l: {"pruned": M("pruned", 0.005, l), "random": M("random_mask", 0.005, l),
                   "확실": fmt(certain(A("pruned", 0.005, l), A("random_mask", 0.005, l))),
                   "확실(ddof1)": fmt(certain(A("pruned", 0.005, l), A("random_mask", 0.005, l), ddof=1))} for l in LEARNERS}
    c_d, c_p = cos.get((*D, "fg")), cos.get(("pruned", 0.005, "fg"))
    n_d, n_p = cos.get((*D, "np")), cos.get(("pruned", 0.005, "np"))
    J["P5"] = {"fg_ratio": (c_p / c_d) if c_d and c_p else None, "np_ratio": (n_p / n_d) if n_d and n_p else None}
    J["P5"]["fg>=10x"] = fmt(None if J["P5"]["fg_ratio"] is None else J["P5"]["fg_ratio"] >= 10)
    J["P5"]["np within 2x"] = fmt(None if J["P5"]["np_ratio"] is None else 0.5 <= J["P5"]["np_ratio"] <= 2.0)
    gd = [b - x for b, x in zip(A(*D, "bp"), A(*D, "dfa"))]
    g5 = [b - x for b, x in zip(A("pruned", 0.005, "bp"), A("pruned", 0.005, "dfa"))]
    J["P6"] = {"gap_dense": stat(gd)[0], "gap_pruned0.5": stat(g5)[0],
               "차이 없음": fmt(not (certain(gd, g5) or certain(g5, gd)))}
    J["P7"] = {f"{100 * d:g}%": {"acc0_reinit": S["cells"][f"{key('pruned_reinit', d)}/fg"]["acc0"],
                                 "fg pruned": M("pruned", d, "fg"), "fg reinit": M("pruned_reinit", d, "fg"), "fg small": M("dense_small", d, "fg"),
                                 "reinit<pruned 확실": fmt(certain(A("pruned", d, "fg"), A("pruned_reinit", d, "fg"))),
                                 "reinit>small 확실": fmt(certain(A("pruned_reinit", d, "fg"), A("dense_small", d, "fg")))} for d in (0.01, 0.005)}
    J["P8"] = {l: {"random": M("random_mask", 0.005, l), "degree": M("random_degree", 0.005, l), "pruned": M("pruned", 0.005, l),
                   "degree>random 확실": fmt(certain(A("random_degree", 0.005, l), A("random_mask", 0.005, l))),
                   "pruned>degree 확실": fmt(certain(A("pruned", 0.005, l), A("random_degree", 0.005, l)))} for l in ("fg", "np")}
    J["P9"] = {"random 0.5% fg": M("random_mask", 0.005, "fg"), "판정": fmt(M("random_mask", 0.005, "fg") >= 0.60)}
    long = {}
    for p in sorted(glob.glob(os.path.join(RES, "h3long", "*", "*", "seed0.json"))):
        r = json.load(open(p, encoding="utf-8"))
        a, l = p.split(os.sep)[-3], p.split(os.sep)[-2]
        q = os.path.join(RES, "h3", a, l, "seed0.json")
        r15 = json.load(open(q, encoding="utf-8")) if os.path.exists(q) else None
        long[f"{a}/{l}"] = {"ep45": r["last3_test"], "ep15_seed0": r15["last3_test"] if r15 else None, "backoffs": r["backoffs"],
                            "failed": r["failed"], "lr": [r["lr_selected"], r["lr_final"]]}
    S["long"] = long
    with open(os.path.join(RES, "summary_v3.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, ensure_ascii=False, indent=1)

    def cell(c, d, l):
        k = f"{key(c, d)}/{l}"
        if k not in S["cells"]:
            return "-"
        x = S["cells"][k]
        tag = f" (실패 {x['failed']}/{x['n']})" if x["failed"] else ""
        return f"{100 * x['acc']:.1f}±{100 * x['std']:.1f}{tag}"

    T = ["## H2 v3 — 시험 정확도 % (마지막 3 평가 평균, 시드 3, 학습률·수락은 검증으로 선택)", "",
         "| 회사 | 학습 0걸음 | 역전파 | DFA | 노드 섭동 | 순방향 기울기 |", "|---|---|---|---|---|---|"]
    for c, d in ARMS:
        a0 = S["cells"].get(f"{key(c, d)}/bp", {}).get("acc0")
        T.append(f"| {label(c, d)} | {100 * a0:.1f} | " + " | ".join(cell(c, d, l) for l in LEARNERS) + " |")
    T += ["", "## H2 v3 — 역전파 대비 격차 (점) / 추정 방향 코사인 / 섭동 매개변수 수", "",
          "| 회사 | DFA | 노드 섭동 | 순방향 기울기 | 코사인 (fg / np / dfa) | 섭동 수 |", "|---|---|---|---|---|---|"]
    for c, d in ARMS:
        g = [S["cells"].get(f"{key(c, d)}/{l}", {}).get("gap_to_bp") for l in ("dfa", "np", "fg")]
        co = [S["cells"].get(f"{key(c, d)}/{l}", {}).get("cos_each") for l in ("fg", "np", "dfa")]
        npert = S["cells"].get(f"{key(c, d)}/fg", {}).get("n_perturbed")
        T.append(f"| {label(c, d)} | " + " | ".join("-" if x is None else f"{100 * x:.1f}" for x in g) + " | "
                 + " / ".join("-" if x is None else f"{x:.3f}" for x in co) + f" | {npert:,} |")
    T += ["", "## H2 v3 — 45 에폭 (시드 0) vs 15 에폭 (시드 0)", "", "| 회사/학습기 | 15 에폭 | 45 에폭 | 발산 복원 | 실패 |", "|---|---|---|---|---|"]
    for k, v in long.items():
        T.append(f"| {k} | {100 * v['ep15_seed0']:.1f} | {100 * v['ep45']:.1f} | {v['backoffs']} | {'예' if v['failed'] else '아니오'} |")
    T += ["", "## 자동 판정", "", "```json", json.dumps(J, ensure_ascii=False, indent=1), "```"]
    md = "\n".join(T)
    with open(os.path.join(RES, "tables_v3.md"), "w", encoding="utf-8") as f:
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
    order = [("pruned", 0.005), ("pruned_reinit", 0.005), ("random_degree", 0.005), ("random_mask", 0.005), ("dense_small", 0.005), ("dense", 1.0)]
    colors = {"pruned": "#d9480f", "pruned_reinit": "#f4a261", "random_degree": "#7048e8", "random_mask": "#adb5bd", "dense_small": "#2b8a3e", "dense": "#212529"}
    fig, ax = plt.subplots(figsize=(10, 5.0))
    w = 0.13
    for i, (c, d) in enumerate(order):
        ys, es = [], []
        for l in LEARNERS:
            x = S["cells"].get(f"{key(c, d)}/{l}")
            ys.append(100 * x["acc"] if x else 0)
            es.append(100 * x["std"] if x else 0)
        ax.bar([j + (i - len(order) / 2 + 0.5) * w for j in range(len(LEARNERS))], ys, width=w, yerr=es, capsize=2, color=colors[c],
               label=label(c, d), edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(len(LEARNERS)))
    ax.set_xticklabels(["역전파", "DFA", "노드 섭동", "순방향 기울기"])
    ax.set_ylim(50, 100)
    ax.set_ylabel("시험 정확도 (%)")
    ax.set_title("감사팀 없이 배우기 (v3): 0.5% 회사들과 밀집망")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_v3.png"), dpi=140)
    plt.close(fig)
    print(md)


if __name__ == "__main__":
    main()
