# -*- coding: utf-8 -*-
"""논문 2 탐색 결과 집계: results/p2/summary.json + 그림 (fig_e1_speedup, fig_e2_order, fig_e4_block, fig_x1_kwta).
예측 P1~P6, X1, X2 의 자동 판정 ('확실히' = 시드 평균 차이 > 두 조건 표준편차 합).
  python experiments/p2/report_data.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(REPO_ROOT, "results")
P2 = os.path.join(RES, "p2")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "yellow": "#eda100", "pink": "#e87ba4", "dgreen": "#008300", "purple": "#4a3aa7", "red": "#e34948", "grey": "#8a8a8a"}


def load_accs(pattern, key="final_acc"):
    vals = []
    for f in sorted(glob.glob(pattern)):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        vals.append(d[key] * 100 if key in ("final_acc",) else d[key])
    return vals


def ms(v):
    if not v:
        return None
    return {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v), "values": [float(x) for x in v]}


def clearly(a, b):
    """a 가 b 보다 확실히 큰가 / 작은가 / 판단 불가. a, b 는 ms() 결과."""
    if not a or not b or a["n"] < 3 or b["n"] < 3:
        return "판단 불가"
    diff = a["mean"] - b["mean"]
    if abs(diff) > a["std"] + b["std"]:
        return "확실히 높음" if diff > 0 else "확실히 낮음"
    return "차이 불확실"


def main():
    S = {"e2": {}, "e3": {}, "e4": {}, "x1": {}, "x2": {}, "e1": {}, "tiles": {}, "predictions": {}}
    # ---------------- E2
    for model, dens in (("mnist", [0.01, 0.005]), ("cnn", [0.03, 0.01])):
        for d in dens:
            row = {}
            if model == "cnn":
                row["global"] = ms(load_accs(os.path.join(RES, "core_cifar", f"d{d:g}", "pd_mag_global", "seed*.json")))
            else:
                row["global"] = ms(load_accs(os.path.join(P2, "e2", model, f"d{d:g}", "global", "seed*.json")))
            for c in ("sync", "bottom_up", "top_down"):
                row[c] = ms(load_accs(os.path.join(P2, "e2", model, f"d{d:g}", c, "seed*.json")))
            S["e2"][f"{model}_d{d:g}"] = row
    # ---------------- E3
    for model, d in (("mnist", 0.005), ("cnn", 0.01)):
        row = {"sync": S["e2"][f"{model}_d{d:g}"]["sync"]}
        for c in ("two_waves", "progress_gated"):
            row[c] = ms(load_accs(os.path.join(P2, "e3", model, f"d{d:g}", c, "seed*.json")))
        starts = []
        for f in glob.glob(os.path.join(P2, "e3", model, f"d{d:g}", "progress_gated", "seed*.json")):
            dd = json.load(open(f, encoding="utf-8"))
            starts.append(dd["prune_start_step"] / dd["total_steps"] if dd["prune_start_step"] is not None else None)
        row["gated_start_frac"] = starts
        S["e3"][f"{model}_d{d:g}"] = row
    # ---------------- E4
    for model, dens in (("mnist", [0.01, 0.005]), ("cnn", [0.03, 0.01])):
        base = os.path.join(RES, "core" if model == "mnist" else "core_cifar")
        for d in dens:
            row = {"unstructured_during": ms(load_accs(os.path.join(base, f"d{d:g}", "pd_mag_global", "seed*.json"))),
                   "unstructured_oneshot": ms(load_accs(os.path.join(base, f"d{d:g}", "ttp" if model == "mnist" else "ttp_ftlr05", "seed*.json"))),
                   "block16_during": ms(load_accs(os.path.join(P2, "e4", model, f"d{d:g}", "block16_during", "seed*.json"))),
                   "block16_oneshot": ms(load_accs(os.path.join(P2, "e4", model, f"d{d:g}", "block16_oneshot", "seed*.json"))),
                   "block16_during_pl": ms(load_accs(os.path.join(P2, "e4", model, f"d{d:g}", "block16_during_pl", "seed*.json"))),
                   "block16_oneshot_pl": ms(load_accs(os.path.join(P2, "e4", model, f"d{d:g}", "block16_oneshot_pl", "seed*.json")))}
            if row["unstructured_during"] and row["block16_during_pl"]:
                row["gap_during_pl"] = row["unstructured_during"]["mean"] - row["block16_during_pl"]["mean"]
            if row["unstructured_oneshot"] and row["block16_oneshot_pl"]:
                row["gap_oneshot_pl"] = row["unstructured_oneshot"]["mean"] - row["block16_oneshot_pl"]["mean"]
            # 경로 단절: 완전히 빈 층 수 (시드 0 마스크)
            import torch
            for cond in ("block16_during", "block16_oneshot", "block16_during_pl", "block16_oneshot_pl"):
                mp = os.path.join(P2, "masks", f"e4_{model}_d{d:g}_{cond}_s0.pt")
                if os.path.exists(mp):
                    mk = torch.load(mp)
                    row[f"empty_layers_{cond}"] = int(sum(int(v.sum()) == 0 for v in mk.values()))
            if row["unstructured_during"] and row["block16_during"]:
                row["gap_during"] = row["unstructured_during"]["mean"] - row["block16_during"]["mean"]
            if row["unstructured_oneshot"] and row["block16_oneshot"]:
                row["gap_oneshot"] = row["unstructured_oneshot"]["mean"] - row["block16_oneshot"]["mean"]
            S["e4"][f"{model}_d{d:g}"] = row
    # ---------------- X1
    for c, dens in (("dense", 1.0), ("dense_small", 0.005), ("global", 0.005)):
        for k in (1.0, 0.1, 0.05):
            S["x1"][f"{c}_k{k:g}"] = ms(load_accs(os.path.join(P2, "x1", "mnist", f"d{dens:g}", f"{c}_k{k:g}", "seed*.json")))
    # ---------------- X2
    for ds in ("split_mnist", "permuted_mnist"):
        row = {}
        for name, path in (("finetune", os.path.join(RES, "exp3", ds, "finetune")), ("finetune_lr2e4", os.path.join(P2, "x2", ds, "finetune_lr2e4")),
                           ("er", os.path.join(RES, "exp3", ds, "er")),
                           ("fastslow_decay", os.path.join(P2, "x2", ds, "fastslow_decay")), ("fastslow_merge", os.path.join(P2, "x2", ds, "fastslow_merge"))):
            vals = []
            for f in sorted(glob.glob(os.path.join(path, "seed*.json"))):
                vals.append(json.load(open(f, encoding="utf-8"))["continual"]["avg_acc"] * 100)
            row[name] = ms(vals)
        S["x2"][ds] = row
    # ---------------- E1 micro
    e1 = []
    p = os.path.join(P2, "e1_micro.csv")
    if os.path.exists(p):
        e1 = list(csv.DictReader(open(p, encoding="utf-8")))
    best = {}
    for r in e1:
        if r["status"] != "ok" or r["format"] == "dense":
            continue
        key = (int(r["size"]), int(r["N"]), r["format"], r["dtype"])
        sp = float(r["speedup"])
        best.setdefault(key, []).append((float(r["density"]), sp))
    S["e1"]["unsupported"] = sorted(set(f"{r['format']} {r['dtype']}: {r['error'][:80]}" for r in e1 if r["status"] != "ok"))
    S["e1"]["max_speedup_small_N64plus"] = max([float(r["speedup"]) for r in e1 if r["status"] == "ok" and r["format"] != "dense"
                                                and int(r["size"]) == 1024 and int(r["N"]) >= 64] or [0])
    S["e1"]["csr4096_speedup_by_density_N1"] = {r["density"]: float(r["speedup"]) for r in e1 if r["status"] == "ok" and r["format"] == "csr"
                                                 and int(r["size"]) == 4096 and int(r["N"]) == 1 and r["dtype"] == "fp32" and r["shape"] == "unstructured"}
    S["e1"]["csr4096_speedup_by_density_N128"] = {r["density"]: float(r["speedup"]) for r in e1 if r["status"] == "ok" and r["format"] == "csr"
                                                   and int(r["size"]) == 4096 and int(r["N"]) == 128 and r["dtype"] == "fp32" and r["shape"] == "unstructured"}
    S["e1"]["best_rows"] = sorted([(float(r["speedup"]), r["size"], r["N"], r["density"], r["shape"], r["format"], r["dtype"]) for r in e1 if r["status"] == "ok" and r["format"] != "dense"], reverse=True)[:12]
    # ---------------- tiles
    p = os.path.join(P2, "tile_occupancy.csv")
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        for r in rows:
            if r["B"] != "16":
                continue
            S["tiles"].setdefault(r["mask_set"], {})[r["layer"]] = {"density": float(r["density"]), "empty16": float(r["empty_tile_frac"]),
                                                                    "theory16": float(r["random_theory"]), "dead_rows": float(r["dead_rows"]), "dead_cols": float(r["dead_cols"])}
    p = os.path.join(P2, "e1_real.csv")
    if os.path.exists(p):
        S["e1"]["real"] = [r for r in csv.DictReader(open(p, encoding="utf-8"))]

    # ---------------- 판정
    P = S["predictions"]
    if e1:
        P["P1"] = "맞음" if S["e1"]["max_speedup_small_N64plus"] <= 1.0 else f"틀림 (최대 {S['e1']['max_speedup_small_N64plus']:.2f}x)"
        c = S["e1"]["csr4096_speedup_by_density_N128"]
        lo = [v for k, v in c.items() if float(k) <= 0.02]
        P["P2"] = ("판단 불가" if not lo else ("맞음" if max(lo) > 1.0 else "틀림")) + f" (4096, N=128, 밀도<=2% 최대 {max(lo) if lo else 0:.2f}x; N=1: {max([v for k, v in S['e1']['csr4096_speedup_by_density_N1'].items() if float(k) <= 0.02] or [0]):.2f}x)"
    if S["tiles"]:
        res = []
        for key, layers in S["tiles"].items():
            if key.endswith("_learned"):
                rnd = S["tiles"].get(key.replace("_learned", "_random"), {})
                for l, v in layers.items():
                    res.append((key, l, v["empty16"], rnd.get(l, {}).get("empty16"), v["theory16"]))
        # 빈 타일이 아예 없는 층 (학습·무작위 모두 0) 은 비교에서 뺀다
        res_nz = [x for x in res if x[2] > 0 or (x[3] if x[3] is not None else x[4]) > 1e-3]
        higher = all(e > (r if r is not None else t) for _, _, e, r, t in res_nz)
        P["P3"] = ("맞음" if higher else "틀림") + " (" + "; ".join(f"{k.split('_d')[0]} {l}: 학습 {e:.2f} vs 무작위 {r if r is None else round(r, 3)}" for k, l, e, r, t in res[:4]) + ")"
    e2c = S["e2"].get("cnn_d0.01", {})
    e2m = S["e2"].get("mnist_d0.005", {})
    def _p4(row):
        if not row.get("sync") or not row.get("bottom_up") or not row.get("top_down"):
            return "판단 불가"
        bu_ok = row["bottom_up"]["mean"] >= row["sync"]["mean"] - 0.3
        td_bad = row["top_down"]["mean"] <= row["sync"]["mean"] - 0.5 and clearly(row["top_down"], row["sync"]) == "확실히 낮음"
        return f"bottom_up {'OK' if bu_ok else 'X'} (sync {row['sync']['mean']:.2f}, bu {row['bottom_up']['mean']:.2f}), top_down {'OK' if td_bad else 'X'} (td {row['top_down']['mean']:.2f}, {clearly(row['top_down'], row['sync'])})"
    P["P4"] = {"cnn_d0.01": _p4(e2c), "mnist_d0.005": _p4(e2m)}
    p5 = {}
    for k, row in S["e4"].items():
        if "gap_during" in row and "gap_oneshot" in row:
            p5[k] = f"{'맞음' if row['gap_during'] < row['gap_oneshot'] else '틀림'} (격차 during {row['gap_during']:.2f} vs oneshot {row['gap_oneshot']:.2f})"
    P["P5"] = p5 or "판단 불가"
    p6 = {}
    for k, row in S["e3"].items():
        if row.get("two_waves") and row.get("progress_gated") and row.get("sync"):
            tw = abs(row["two_waves"]["mean"] - row["sync"]["mean"]) <= 0.3
            pg = row["progress_gated"]["mean"] >= row["sync"]["mean"] - 0.3 or clearly(row["progress_gated"], row["sync"]) != "확실히 낮음"
            p6[k] = f"two_waves {'OK' if tw else 'X'} ({row['two_waves']['mean']:.2f} vs sync {row['sync']['mean']:.2f}), gated {'OK' if pg else 'X'} ({row['progress_gated']['mean']:.2f}, {clearly(row['progress_gated'], row['sync'])}, start {np.mean([s for s in row['gated_start_frac'] if s is not None]) if row['gated_start_frac'] else None})"
    P["P6"] = p6 or "판단 불가"
    x1 = S["x1"]
    if x1.get("global_k0.05") and x1.get("global_k1") and x1.get("dense_small_k0.05"):
        drop = x1["global_k1"]["mean"] - x1["global_k0.05"]["mean"]
        P["X1"] = f"{'맞음' if drop <= 1.0 and x1['global_k0.05']['mean'] > x1['dense_small_k0.05']['mean'] else '틀림'} (가지치기망 k=5%: {x1['global_k0.05']['mean']:.2f}, k=100%: {x1['global_k1']['mean']:.2f}, 손실 {drop:.2f}; 작은 dense k=5%: {x1['dense_small_k0.05']['mean']:.2f})"
    x2 = S["x2"].get("split_mnist", {})
    if x2.get("fastslow_decay") and x2.get("finetune") and x2.get("er"):
        bestv = max(x2["fastslow_decay"]["mean"], x2.get("fastslow_merge", {}).get("mean", 0) if x2.get("fastslow_merge") else 0)
        P["X2"] = f"{'맞음' if bestv >= x2['finetune']['mean'] + 5 and bestv < x2['er']['mean'] else '틀림'} (Split: finetune {x2['finetune']['mean']:.1f}, fast/slow decay {x2['fastslow_decay']['mean']:.1f}, merge {x2['fastslow_merge']['mean'] if x2.get('fastslow_merge') else float('nan'):.1f}, ER {x2['er']['mean']:.1f})"

    with open(os.path.join(P2, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, indent=1, ensure_ascii=False)

    # ---------------- 그림
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    if e1:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
        for i, size in enumerate((1024, 4096)):
            for j, N in enumerate((1, 128)):
                ax = axes[i][j]
                for (fmt, dt, shape), c, ls in (
                        (("csr", "fp32", "unstructured"), COL["blue"], "-"), (("csr", "fp16", "unstructured"), COL["blue"], "--"),
                        (("bsr16", "fp32", "block16"), COL["orange"], "-"), (("bsr16", "fp16", "block16"), COL["orange"], "--"),
                        (("csr", "fp32", "block16"), COL["green"], "-")):
                    pts = sorted((float(r["density"]), float(r["speedup"])) for r in e1 if r["status"] == "ok" and r["format"] == fmt
                                 and r["dtype"] == dt and r["shape"] == shape and int(r["size"]) == size and int(r["N"]) == N)
                    if pts:
                        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=c, linestyle=ls, label=f"{fmt} {dt} ({shape})")
                ax.axhline(1.0, color=COL["grey"], linestyle=":")
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_title(f"{size}x{size}, N={N}")
                ax.set_xlabel("density"); ax.set_ylabel("speed-up vs dense (x)")
                if i == 0 and j == 0:
                    ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(P2, "fig_e1_speedup.png"), dpi=150); plt.close(fig)
    # E2
    keys = [k for k in S["e2"] if S["e2"][k].get("sync")]
    if keys:
        fig, ax = plt.subplots(figsize=(9, 4))
        conds = ["global", "sync", "bottom_up", "top_down"]
        cols = [COL["grey"], COL["blue"], COL["green"], COL["orange"]]
        w = 0.2
        for ci, c in enumerate(conds):
            xs, ys, es = [], [], []
            for ki, k in enumerate(keys):
                v = S["e2"][k].get(c)
                if v:
                    xs.append(ki + (ci - 1.5) * w); ys.append(v["mean"]); es.append(v["std"])
            ax.bar(xs, ys, width=w, yerr=es, color=cols[ci], label=c, capsize=2)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys)
        lo = min(S["e2"][k][c]["mean"] for k in keys for c in conds if S["e2"][k].get(c)) - 2
        ax.set_ylim(lo, 100); ax.set_ylabel("test accuracy (%)"); ax.legend(fontsize=8, ncol=4)
        ax.set_title("E2: order of pruning across layers (3 seeds)")
        fig.tight_layout(); fig.savefig(os.path.join(P2, "fig_e2_order.png"), dpi=150); plt.close(fig)
    # E4
    keys = [k for k in S["e4"] if "gap_during" in S["e4"][k] or "gap_oneshot" in S["e4"][k]]
    if keys:
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, (g, c, lab) in enumerate((("gap_during", COL["blue"], "prune-during, global tiles"), ("gap_oneshot", COL["orange"], "one-shot, global tiles"),
                                         ("gap_during_pl", COL["green"], "prune-during, per-layer tile budgets"), ("gap_oneshot_pl", COL["yellow"], "one-shot, per-layer tile budgets"))):
            ax.bar([ki + (i - 1.5) * 0.2 for ki, k in enumerate(keys)], [S["e4"][k].get(g, np.nan) for k in keys], width=0.2, color=c, label=lab)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys)
        ax.axhline(0, color=COL["grey"], linewidth=0.8)
        ax.set_ylabel("accuracy lost by 16x16 tiles (points)"); ax.legend(fontsize=8)
        ax.set_title("E4: unstructured minus block16 (lower = tiles cost less)")
        fig.tight_layout(); fig.savefig(os.path.join(P2, "fig_e4_block.png"), dpi=150); plt.close(fig)
    # X1
    if any(S["x1"].values()):
        fig, ax = plt.subplots(figsize=(7, 4))
        for c, col in (("dense", COL["grey"]), ("dense_small", COL["blue"]), ("global", COL["yellow"])):
            ks = [k for k in (1.0, 0.1, 0.05) if S["x1"].get(f"{c}_k{k:g}")]
            ax.errorbar([str(k) for k in ks], [S["x1"][f"{c}_k{k:g}"]["mean"] for k in ks], yerr=[S["x1"][f"{c}_k{k:g}"]["std"] for k in ks],
                        marker="o", color=col, capsize=2, label={"dense": "dense big (1.86M)", "dense_small": "dense small (0.5% budget)", "global": "prune-during to 0.5%"}[c])
        ax.set_xlabel("fraction of hidden units allowed active (k-WTA)"); ax.set_ylabel("MNIST test accuracy (%)"); ax.legend(fontsize=8)
        ax.set_title("X1: activity sparsity on top of connection sparsity")
        fig.tight_layout(); fig.savefig(os.path.join(P2, "fig_x1_kwta.png"), dpi=150); plt.close(fig)

    print(json.dumps(S["predictions"], ensure_ascii=False, indent=1))
    for sec in ("e2", "e3", "e4", "x1", "x2"):
        print(f"--- {sec}")
        for k, row in S[sec].items():
            if isinstance(row, dict) and "mean" in row:
                print(f"  {k}: {row['mean']:.2f} ± {row['std']:.2f} (n={row['n']})")
            elif isinstance(row, dict):
                print("  " + k + ": " + ", ".join(f"{c}={v['mean']:.2f}±{v['std']:.2f}" for c, v in row.items() if isinstance(v, dict) and "mean" in v)
                      + "".join(f", {c}={v:.2f}" for c, v in row.items() if isinstance(v, float)))
    print("--- e1", S["e1"].get("unsupported"), S["e1"].get("best_rows", [])[:5])


if __name__ == "__main__":
    main()
