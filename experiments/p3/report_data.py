# -*- coding: utf-8 -*-
"""p3 집계: results/p3/{h2,h1,lr_select.json} -> summary.json, tables.md, 그림 3 장, 예측 자동 판정.
부분 결과(시드 부족, 단계 미완)도 집계한다. 판정 규칙 "확실히": 평균 차이 > 두 조건 표준편차 합. 시드 < 2 면 판단 불가."""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RES = os.path.join(REPO_ROOT, "results", "p3")
LEARNERS = ["bp", "dfa", "fg", "wp", "np"]
ARMS = [("dense", 1.0), ("pruned", 0.01), ("random_mask", 0.01), ("dense_small", 0.01),
        ("pruned", 0.005), ("random_mask", 0.005), ("dense_small", 0.005)]
ARM_LABEL = {("dense", 1.0): "dense 100%", ("pruned", 0.01): "pruned 1%", ("pruned", 0.005): "pruned 0.5%",
             ("random_mask", 0.01): "random 1%", ("random_mask", 0.005): "random 0.5%",
             ("dense_small", 0.01): "small 1%", ("dense_small", 0.005): "small 0.5%"}
CRITERIA = ["magnitude", "rank1_pre", "rank1_prepost", "rank1_now", "conn_drive", "rank1_reward", "rank1_conf", "random"]


H2_SUB = "h2v2" if os.path.isdir(os.path.join(RES, "h2v2")) else "h2"   # v2 (코사인 + 재시도) 가 있으면 그것이 정본


def arm_dir(c, d):
    return os.path.join(RES, H2_SUB, f"{c}_d{d:g}")


def load_h2() -> Dict[tuple, Dict[str, List[dict]]]:
    out: Dict[tuple, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for c, d in ARMS:
        for l in LEARNERS:
            for p in sorted(glob.glob(os.path.join(arm_dir(c, d), l, "seed*.json"))):
                if "_try" in os.path.basename(p):
                    continue
                try:
                    out[(c, d)][l].append(json.load(open(p, encoding="utf-8")))
                except Exception:
                    pass
    return out


def ms(xs: List[float]):
    if not xs:
        return None, None, 0
    a = np.array(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=0)), len(a)


def certain(m1, s1, n1, m2, s2, n2) -> Optional[bool]:
    """m1 > m2 확실히? 시드 부족이면 None."""
    if m1 is None or m2 is None or n1 < 2 or n2 < 2:
        return None
    return (m1 - m2) > (s1 + s2)


def samples_to(curve, thr):
    for c in curve:
        if c["test_acc"] >= thr:
            return c["samples"]
    return None


def summarize_h2(h2):
    table = {}
    for arm in ARMS:
        key = f"{arm[0]}_d{arm[1]:g}"
        table[key] = {}
        bp = h2[arm].get("bp", [])
        bp_m, bp_s, bp_n = ms([r["last3_mean"] for r in bp])
        for l in LEARNERS:
            runs = h2[arm].get(l, [])
            m, s, n = ms([r["last3_mean"] for r in runs])
            cos = ms([pr["cos_each"] for r in runs for pr in r["probes"]])[0] if runs else None
            cos0 = ms([r["probes"][0]["cos_each"] for r in runs if r["probes"]])[0] if runs else None
            snr = ms([pr["snr"] for r in runs for pr in r["probes"] if "snr" in pr])[0] if runs else None
            s90 = [samples_to(r["curve"], 0.90) for r in runs]
            s90 = [x for x in s90 if x is not None]
            pk_m, pk_s, _ = ms([r["best_acc"] for r in runs])
            table[key][l] = {
                "acc_mean": m, "acc_std": s, "n": n, "peak_mean": pk_m, "peak_std": pk_s,
                "diverged": sum(1 for r in runs if r.get("diverged")),
                "attempts": sorted(set(int(r["cfg"].get("attempt", 0)) for r in runs)),
                "gap_to_bp": (bp_m - m) if (m is not None and bp_m is not None) else None,
                "gap_std": math.sqrt((bp_s or 0) ** 2 + (s or 0) ** 2) if (m is not None and bp_m is not None) else None,
                "cos_each": cos, "cos_each_init": cos0, "snr": snr,
                "samples_to_90": (float(np.mean(s90)) if s90 else None), "reached_90": f"{len(s90)}/{n}",
                "lr": runs[0]["cfg"]["lr"] if runs else None,
                "active": runs[0]["active_weights"] if runs else None,
                "alive_hidden": runs[0]["alive_hidden"] if runs else None,
                "hidden": runs[0]["hidden_neurons"] if runs else None,
            }
    return table


def judge_h2(t):
    def g(arm, l, k="acc_mean"):
        return t.get(f"{arm[0]}_d{arm[1]:g}", {}).get(l, {}).get(k)

    def cert_gt(arm1, l1, arm2, l2):
        a = t.get(f"{arm1[0]}_d{arm1[1]:g}", {}).get(l1, {})
        b = t.get(f"{arm2[0]}_d{arm2[1]:g}", {}).get(l2, {})
        return certain(a.get("acc_mean"), a.get("acc_std"), a.get("n", 0), b.get("acc_mean"), b.get("acc_std"), b.get("n", 0))

    def fmt(v):
        return "판단 불가" if v is None else ("맞음" if v else "틀림")

    J = {}
    D = ("dense", 1.0)
    # P1: dense 사다리
    order = [(l, g(D, l)) for l in LEARNERS]
    J["P1"] = {"dense_acc": {l: g(D, l) for l in LEARNERS},
               "bp>dfa": fmt(cert_gt(D, "bp", D, "dfa")), "dfa>np": fmt(cert_gt(D, "dfa", D, "np")),
               "np>fg": fmt(cert_gt(D, "np", D, "fg")),
               "fg,wp <= bp-10": fmt(None if g(D, "fg") is None or g(D, "wp") is None or g(D, "bp") is None
                                     else (g(D, "fg") <= g(D, "bp") - 0.10 and g(D, "wp") <= g(D, "bp") - 0.10))}
    # P2: fg/wp 격차가 밀도 내려갈수록 줄어드는가 (dense -> pruned 1% -> pruned 0.5%)
    P2 = {}
    for l in ("fg", "wp"):
        gaps = {ARM_LABEL[a]: g(a, l, "gap_to_bp") for a in (D, ("pruned", 0.01), ("pruned", 0.005))}
        gd, g1, g5 = gaps.values()
        sd = g(D, l, "gap_std"); s5 = g(("pruned", 0.005), l, "gap_std")
        nd = g(D, l, "n") or 0; n5 = g(("pruned", 0.005), l, "n") or 0
        mono = None if None in (gd, g1, g5) else (gd > g1 > g5)
        cert = None if (None in (gd, g5) or nd < 2 or n5 < 2) else ((gd - g5) > (sd + s5))
        P2[l] = {"gaps": gaps, "monotone": fmt(mono), "dense_vs_0.5%_certain": fmt(cert)}
    J["P2"] = P2
    # P3: pruned vs dense_small
    P3 = {}
    for l in ("fg", "wp", "np", "dfa", "bp"):
        P3[l] = {ARM_LABEL[a]: {"pruned": g(a, l), "small": g(("dense_small", a[1]), l),
                                "pruned>small 확실히": fmt(cert_gt(a, l, ("dense_small", a[1]), l)),
                                "small>=pruned": fmt(None if g(a, l) is None or g(("dense_small", a[1]), l) is None
                                                     else g(("dense_small", a[1]), l) >= g(a, l))}
                 for a in (("pruned", 0.01), ("pruned", 0.005))}
    J["P3"] = P3
    # P4: pruned vs random_mask at 0.5%
    J["P4"] = {l: {"pruned": g(("pruned", 0.005), l), "random": g(("random_mask", 0.005), l),
                   "확실히": fmt(cert_gt(("pruned", 0.005), l, ("random_mask", 0.005), l))} for l in LEARNERS}
    # P5: SNR
    P5 = {}
    for l in ("fg", "wp", "np", "dfa"):
        cd, c5 = g(D, l, "cos_each"), g(("pruned", 0.005), l, "cos_each")
        ratio = (c5 / cd) if (cd and c5) else None
        P5[l] = {"cos_dense": cd, "cos_pruned0.5": c5, "ratio": ratio}
    P5["fg_ratio>=10"] = fmt(None if P5["fg"]["ratio"] is None else P5["fg"]["ratio"] >= 10)
    P5["np_ratio_within_2x"] = fmt(None if P5["np"]["ratio"] is None else 0.5 <= P5["np"]["ratio"] <= 2.0)
    J["P5"] = P5
    # P6: dfa gap independent of density
    gd, g5 = g(D, "dfa", "gap_to_bp"), g(("pruned", 0.005), "dfa", "gap_to_bp")
    J["P6"] = {"gap_dense": gd, "gap_pruned0.5": g5,
               "no_certain_difference": fmt(None if None in (gd, g5) else abs(gd - g5) <= 0.01)}
    return J


def load_h1():
    runs = []
    for p in sorted(glob.glob(os.path.join(RES, "h1", "seed*.json"))):
        try:
            runs.append(json.load(open(p, encoding="utf-8")))
        except Exception:
            pass
    return runs


def summarize_h1(runs):
    out = {"n": len(runs), "dense_acc": ms([r["dense_acc"] for r in runs])[0], "oneshot": {}, "gradual": {}}
    for mode in ("oneshot", "gradual"):
        for crit in CRITERIA:
            for r in runs:
                for D, v in r.get(mode, {}).get(crit, {}).items():
                    out[mode].setdefault(crit, {}).setdefault(D, []).append(v["acc"])
        for crit in list(out[mode].keys()):
            for D in out[mode][crit]:
                m, s, n = ms(out[mode][crit][D])
                out[mode][crit][D] = {"mean": m, "std": s, "n": n}
    return out


def judge_h1(h):
    def a(mode, crit, D):
        return h.get(mode, {}).get(crit, {}).get(D, {}).get("mean")

    def fmt(v):
        return "판단 불가" if v is None else ("맞음" if v else "틀림")

    def diff(x, y):
        return None if (x is None or y is None) else (x - y)

    J = {}
    J["Q1"] = {"rank1_prepost@1%": a("oneshot", "rank1_prepost", "0.01"), "magnitude@1%": a("oneshot", "magnitude", "0.01")}
    d = diff(J["Q1"]["rank1_prepost@1%"], J["Q1"]["magnitude@1%"])
    J["Q1"]["verdict"] = fmt(None if d is None else d >= -0.01)
    J["Q2"] = {"rank1_now@1%": a("oneshot", "rank1_now", "0.01"), "magnitude@1%": a("oneshot", "magnitude", "0.01")}
    d = diff(J["Q2"]["magnitude@1%"], J["Q2"]["rank1_now@1%"])
    J["Q2"]["verdict"] = fmt(None if d is None else d >= 0.10)
    J["Q3"] = {"gradual@0.5%": a("gradual", "rank1_prepost", "0.005"), "oneshot@0.5%": a("oneshot", "rank1_prepost", "0.005")}
    d = diff(J["Q3"]["gradual@0.5%"], J["Q3"]["oneshot@0.5%"])
    J["Q3"]["verdict"] = fmt(None if d is None else d >= 0.01)
    J["Q4"] = {"reward@1%": a("oneshot", "rank1_reward", "0.01"), "conf@1%": a("oneshot", "rank1_conf", "0.01"),
               "pre@1%": a("oneshot", "rank1_pre", "0.01")}
    d = diff(J["Q4"]["reward@1%"], J["Q4"]["pre@1%"])
    J["Q4"]["verdict"] = fmt(None if d is None else d >= 0.003)
    J["Q5"] = {D: {"conn_drive": a("oneshot", "conn_drive", D), "rank1_prepost": a("oneshot", "rank1_prepost", D)} for D in ("0.01", "0.005")}
    ds = [diff(J["Q5"][D]["conn_drive"], J["Q5"][D]["rank1_prepost"]) for D in ("0.01", "0.005")]
    J["Q5"]["verdict"] = fmt(None if any(x is None for x in ds) else all(abs(x) <= 0.005 for x in ds))
    return J


def pct(x, nd=1):
    return "-" if x is None else f"{100 * x:.{nd}f}"


def tables_md(t, J, h1, J1) -> str:
    L = ["## H2 — 세 회사 × 학습기 (시험 정확도 %, 마지막 3 평가 평균, 시드 평균±표준편차)", "",
         "| 회사 | " + " | ".join(LEARNERS) + " |", "|---|" + "---|" * len(LEARNERS)]
    for arm in ARMS:
        row = t[f"{arm[0]}_d{arm[1]:g}"]
        cells = []
        for l in LEARNERS:
            r = row[l]
            cells.append("-" if r["acc_mean"] is None else f"{pct(r['acc_mean'])}±{pct(r['acc_std'])} (n{r['n']})")
        L.append(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
    L += ["", "## H2 — 학습 중 최고 정확도 % (발산 횟수 / 재시도 단계)", "", "| 회사 | " + " | ".join(LEARNERS) + " |", "|---|" + "---|" * len(LEARNERS)]
    for arm in ARMS:
        row = t[f"{arm[0]}_d{arm[1]:g}"]
        cells = []
        for l in LEARNERS:
            r = row[l]
            if r["acc_mean"] is None:
                cells.append("-")
            else:
                extra = []
                if r["diverged"]:
                    extra.append(f"발산 {r['diverged']}/{r['n']}")
                if r["attempts"] and r["attempts"] != [0]:
                    extra.append("재시도 " + ",".join(str(a) for a in r["attempts"]))
                cells.append(f"{pct(r['peak_mean'])}±{pct(r['peak_std'])}" + (f" ({'; '.join(extra)})" if extra else ""))
        L.append(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
    L += ["", "## H2 — 역전파 대비 격차 (bp − 학습기, 점)", "", "| 회사 | " + " | ".join(LEARNERS[1:]) + " |", "|---|" + "---|" * (len(LEARNERS) - 1)]
    for arm in ARMS:
        row = t[f"{arm[0]}_d{arm[1]:g}"]
        L.append(f"| {ARM_LABEL[arm]} | " + " | ".join(pct(row[l]["gap_to_bp"]) for l in LEARNERS[1:]) + " |")
    L += ["", "## H2 — 추정기와 참 기울기의 코사인 (탐침 평균; 활성 연결 / 살아 있는 은닉 뉴런)", "",
          "| 회사 | 활성 연결 | 살아 있는 뉴런 | " + " | ".join(LEARNERS[1:]) + " |", "|---|---|---|" + "---|" * (len(LEARNERS) - 1)]
    for arm in ARMS:
        row = t[f"{arm[0]}_d{arm[1]:g}"]
        anyr = next((row[l] for l in LEARNERS if row[l]["active"] is not None), None)
        act = f"{anyr['active']:,}" if anyr else "-"
        alive = f"{anyr['alive_hidden']}/{anyr['hidden']}" if anyr else "-"
        L.append(f"| {ARM_LABEL[arm]} | {act} | {alive} | " + " | ".join(
            "-" if row[l]["cos_each"] is None else f"{row[l]['cos_each']:.4f}" for l in LEARNERS[1:]) + " |")
    L += ["", "## H2 — 선택된 학습률 (SGD+모멘텀)", "", "| 회사 | " + " | ".join(LEARNERS) + " |", "|---|" + "---|" * len(LEARNERS)]
    for arm in ARMS:
        row = t[f"{arm[0]}_d{arm[1]:g}"]
        L.append(f"| {ARM_LABEL[arm]} | " + " | ".join("-" if row[l]["lr"] is None else f"{row[l]['lr']:g}" for l in LEARNERS) + " |")
    L += ["", f"## H1 — 배포 결산 가지치기 (시험 정확도 %, 시드 {h1['n']}개 평균; 밀집 {pct(h1['dense_acc'])})", "",
          "| 기준 | one-shot 2% | one-shot 1% | one-shot 0.5% | 4회 결산 1% | 4회 결산 0.5% |", "|---|---|---|---|---|---|"]
    for crit in CRITERIA:
        def v(mode, D):
            x = h1.get(mode, {}).get(crit, {}).get(D)
            return "-" if not x or x["mean"] is None else f"{pct(x['mean'])}±{pct(x['std'])}"
        L.append(f"| {crit} | {v('oneshot', '0.02')} | {v('oneshot', '0.01')} | {v('oneshot', '0.005')} | {v('gradual', '0.01')} | {v('gradual', '0.005')} |")
    L += ["", "## 자동 판정", "", "```json", json.dumps({"H2": J, "H1": J1}, ensure_ascii=False, indent=1), "```"]
    return "\n".join(L)


def figures(t, h1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("Malgun Gothic", "NanumGothic"):
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False
    labels = [ARM_LABEL[a] for a in ARMS]
    # 그림 1: 정확도와 격차
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for l in LEARNERS:
        ys = [t[f"{a[0]}_d{a[1]:g}"][l]["acc_mean"] for a in ARMS]
        es = [t[f"{a[0]}_d{a[1]:g}"][l]["acc_std"] or 0 for a in ARMS]
        xs = [i for i, y in enumerate(ys) if y is not None]
        if xs:
            ax[0].errorbar(xs, [100 * ys[i] for i in xs], yerr=[100 * es[i] for i in xs], marker="o", label=l, capsize=3)
        if l != "bp":
            gs = [t[f"{a[0]}_d{a[1]:g}"][l]["gap_to_bp"] for a in ARMS]
            xg = [i for i, g in enumerate(gs) if g is not None]
            if xg:
                ax[1].plot(xg, [100 * gs[i] for i in xg], marker="o", label=l)
    for a_ in ax:
        a_.set_xticks(range(len(labels)))
        a_.set_xticklabels(labels, rotation=30, ha="right")
        a_.grid(alpha=0.3)
        a_.legend(fontsize=8)
    ax[0].set_ylabel("시험 정확도 (%)")
    ax[0].set_title("세 회사 × 학습기")
    ax[1].set_ylabel("역전파 대비 격차 (점)")
    ax[1].set_title("격차 (bp − 학습기)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_h2_acc_gap.png"), dpi=130)
    plt.close(fig)
    # 그림 2: 코사인 vs 활성 연결 수
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for l in LEARNERS[1:]:
        pts = [(t[f"{a[0]}_d{a[1]:g}"][l]["active"], t[f"{a[0]}_d{a[1]:g}"][l]["cos_each"], ARM_LABEL[a]) for a in ARMS]
        pts = [p for p in pts if p[0] and p[1]]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], label=l)
            for p in pts:
                ax.annotate(p[2], (p[0], p[1]), fontsize=6, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("활성 연결 수")
    ax.set_ylabel("추정기·참 기울기 코사인")
    ax.set_title("추정기 품질 대 희소성")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_h2_cos.png"), dpi=130)
    plt.close(fig)
    # 그림 3: H1
    if h1["n"]:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        Ds = ["0.02", "0.01", "0.005"]
        w = 0.1
        for i, crit in enumerate(CRITERIA):
            ys = [h1["oneshot"].get(crit, {}).get(D, {}).get("mean") for D in Ds]
            ax.bar([j + (i - len(CRITERIA) / 2) * w for j in range(len(Ds))], [100 * (y or 0) for y in ys], width=w, label=crit)
        ax.set_xticks(range(len(Ds)))
        ax.set_xticklabels(["2%", "1%", "0.5%"])
        ax.set_ylabel("시험 정확도 (%)")
        ax.set_title("H1 one-shot 결산 가지치기 (미세조정 없음)")
        ax.legend(fontsize=7, ncol=4)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "fig_h1.png"), dpi=130)
        plt.close(fig)


def main():
    h2 = load_h2()
    t = summarize_h2(h2)
    J = judge_h2(t)
    runs = load_h1()
    h1 = summarize_h1(runs)
    J1 = judge_h1(h1)
    lr_sel = json.load(open(os.path.join(RES, "lr_select.json"), encoding="utf-8")) if os.path.exists(os.path.join(RES, "lr_select.json")) else {}
    n_runs = sum(len(v) for arm in h2.values() for v in arm.values())
    summary = {"n_h2_runs": n_runs, "n_h1_seeds": h1["n"], "h2": t, "h2_judgement": J, "h1": h1, "h1_judgement": J1, "lr_select": lr_sel}
    with open(os.path.join(RES, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    md = tables_md(t, J, h1, J1)
    with open(os.path.join(RES, "tables.md"), "w", encoding="utf-8") as f:
        f.write(md)
    try:
        figures(t, h1)
    except Exception as e:  # noqa: BLE001
        print(f"[figures] skipped: {type(e).__name__}: {e}")
    print(md)
    print(f"\nH2 runs: {n_runs}, H1 seeds: {h1['n']}")


if __name__ == "__main__":
    main()
