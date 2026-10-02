# -*- coding: utf-8 -*-
"""실험 1+2 통합 (사전 필터링) 집계: results/exp12/<ds>/<mode>_<late_attn>_s<smax>_seed<seed>.json

  python scripts/analyze_exp12.py
출력: docs/report/tables/exp12_<ds>.md, docs/report/figures/exp12_<ds>.png
표: 학습 모드(미세조정 때 쓴 선택 기준 x 뒤쪽 어텐션) 별로, 평가 스킵 비율에 따른 정확도 (학습과 같은 기준으로 평가) 와 FLOPs 비율.
그림: FLOPs 비율 대 정확도. 실험 2 사후 적용(뒤쪽 블록만, identity) 참고선 포함.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import FIG, INK2, MUTED, RES, SERIES, load_json, ms, plt, save, write_table  # noqa: E402


def main():
    for ds in ("mnist", "cifar10"):
        files = glob.glob(os.path.join(RES, "exp12", ds, "*_seed*.json"))
        if not files:
            continue
        rs = [load_json(f) for f in files]
        groups = defaultdict(list)
        for r in rs:
            groups[(r["mode"], r["late_attn"], r.get("substitute", "predicted"))].append(r)
        fracs = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]
        rows = []
        for key in sorted(groups):
            g = groups[key]
            if key[0] == "drop_late":
                dl = [row for r in g for row in r["rows"] if row["eval_mode"] == "drop_late"]
                rows.append(["late blocks removed, same fine-tuning (control)", len(g), ms([r["baseline_full"]["acc"] for r in g]),
                             ms([r["after_warmup_at_smax"]["acc"] for r in g]) + " (removed, no fine-tune)"] + ["-"] * len(fracs)
                            + [f"{np.mean([x['flops_ratio'] for x in dl]):.2f}", ms([x["acc"] for x in dl])])
                continue
            cells = []
            for s in fracs:
                vals = [row["acc"] for r in g for row in r["rows"]
                        if row["skip_frac"] == s and (row["eval_mode"] == key[0] or s == 0.0)]
                cells.append(ms(vals) if vals else "-")
            fl = {s: np.mean([row["flops_ratio"] for r in g for row in r["rows"] if row["skip_frac"] == s]) for s in fracs}
            rows.append([f"{key[0]} / late attn {key[1]}" + (" / identity substitution" if key[2] == "identity" else ""), len(g), ms([r["baseline_full"]["acc"] for r in g]),
                         ms([r["after_warmup_at_smax"]["acc"] for r in g])] + cells
                        + [" / ".join(f"{fl[s]:.2f}" for s in fracs), "-"])
            # 교차 평가: 학습 기준과 다른 기준으로 평가
            for ev in ("thalamic", "layerwise", "random"):
                if ev == key[0]:
                    continue
                cells = []
                for s in fracs[1:]:
                    vals = [row["acc"] for r in g for row in r["rows"] if row["skip_frac"] == s and row["eval_mode"] == ev]
                    cells.append(ms(vals) if vals else "-")
                rows.append([f"  (eval {ev})", len(g), "", ""] + ["" ] + cells + ["", ""])
        write_table(f"exp12_{ds}", ["train mode / late attn", "seeds", "원본 (스킵 없음)", "워밍업만, 스킵 smax"]
                    + [f"스킵 {s:g}" for s in fracs] + ["FLOPs 비율 (스킵 순서대로)", "late blocks removed: acc"], rows,
                    "뒤쪽 절반 블록에만 적용. 건너뛴 토큰은 예측 잔차를 더해 통과. 미세조정은 스킵 비율을 0 에서 0.7 로 올리며 진행. "
                    "late attn pre = 뒤쪽 블록 어텐션을 사전 라우팅(키 25%) 으로 교체. "
                    "late blocks removed = 뒤쪽 블록을 통째로 떼고 같은 에폭을 미세조정한 대조군 (워밍업만 열은 떼기만 하고 미세조정 전); "
                    "FLOPs 비율은 앞쪽 블록만의 비용.")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for i, key in enumerate(sorted(groups)):
            g = groups[key]
            if key[0] == "drop_late":
                dl = [row for r in g for row in r["rows"] if row["eval_mode"] == "drop_late"]
                ax.plot([np.mean([x["flops_ratio"] for x in dl])], [np.mean([x["acc"] for x in dl])], marker="*", markersize=11,
                        color=INK2, linestyle="none", label="late blocks removed, same fine-tuning")
                continue
            xs = [np.mean([row["flops_ratio"] for r in g for row in r["rows"] if row["skip_frac"] == s]) for s in fracs]
            ys = [np.mean([row["acc"] for r in g for row in r["rows"] if row["skip_frac"] == s and (row["eval_mode"] == key[0] or s == 0.0)]) for s in fracs]
            ax.plot(xs, ys, marker="o", color=SERIES[i % 8], label=f"{key[0]} / late attn {key[1]}" + (" / identity substitution" if key[2] == "identity" else ""), linestyle="-" if key[1] == "full" else "--")
        post = glob.glob(os.path.join(RES, "exp2", ds, "seed*.json"))
        if post:
            prs = [load_json(f) for f in post]
            xs, ys = [], []
            for s in (0.3, 0.5, 0.7):
                v = [row for r in prs for row in r["rows"] if row["substitute"] == "identity_late_only" and row["skip_frac"] == s]
                if v:
                    xs.append(np.mean([x["flops_ratio"] for x in v]))
                    ys.append(np.mean([x["acc"] for x in v]))
            if xs:
                ax.plot(xs, ys, marker="s", color=MUTED, linestyle=":", label="post-hoc identity skipping, late blocks only, no fine-tuning")
        base = np.mean([r["baseline_full"]["acc"] for r in rs])
        ax.axhline(base, color=MUTED, linewidth=1, linestyle=":")
        ax.set_xlabel("FLOPs ratio vs full model")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"{ds}: 사전 필터링 (시상 라우터 + 예측 잔차 + 점진 스킵 미세조정), 스킵 0 ~ 0.9")
        ax.legend(fontsize=7, loc="lower right")
        save(fig, f"exp12_{ds}")


if __name__ == "__main__":
    main()
