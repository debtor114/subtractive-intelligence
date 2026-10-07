# -*- coding: utf-8 -*-
"""p4 실행기. 순차, 완료 런 건너뜀, 마감(results/p4/_deadline.txt) 지나면 새 런 시작 안 함.

  python experiments/p4/run_p4.py npb lr main      # P4-B → P4-A 학습률 선택 → P4-A 본 실행 (우선순위 순)
  python experiments/p4/run_p4.py summarize
P4-B: MNIST 밀집 노드 섭동 + (행별 가중치 정규화 | 가중치 감쇠) — p3 v3 프로토콜(h3.run_h3) 그대로, 학습률은 5 에폭 격자.
P4-A: experiments/p4/iso_memory.run_one. 정책: 재현 버퍼 최대(스트림 배치 1, 재현 배치 1) — 아낀 바이트가 재현으로 가는 경로를 가장 유리하게.
"""
from __future__ import annotations

import gc
import glob
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import experiments.p4.iso_memory as iso                                  # noqa: E402
from experiments.p3.h3 import run_h3                                     # noqa: E402

RES = os.path.join(REPO_ROOT, "results", "p4")
KB = 1024
SEEDS = [0, 1, 2]
BUDGETS_PRIORITY = [384 * KB, 512 * KB, 4096 * KB, 1024 * KB]           # 결정 실험(A2) 먼저, 다음 A1
LEARNERS = ["bp_mem", "np", "bp", "fg"]
LR_GRID = {"bp": [0.1, 0.03, 0.01], "np": [0.03, 0.01, 0.003, 0.001], "fg": [0.01, 0.003, 0.001, 0.0003]}
NPB_GRID = [3e-2, 1.5e-2, 7e-3, 3.5e-3, 1.7e-3, 8e-4, 4e-4, 2e-4, 1e-4]


def log(s):
    print(s, flush=True)


def deadline_passed() -> bool:
    p = os.path.join(RES, "_deadline.txt")
    return os.path.exists(p) and time.time() > float(open(p).read().strip())


def guarded(fn, cfg, out):
    if os.path.exists(out):
        return "exists"
    if deadline_passed():
        log(f"[deadline] skip {out}")
        return "deadline"
    try:
        fn(dict(cfg, out=out), log=log)
        return "ok"
    except Exception:
        log(f"[error] {cfg}\n{traceback.format_exc()}")
        with open(os.path.join(RES, "_failures.log"), "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + json.dumps(cfg) + "\n" + traceback.format_exc() + "\n")
        return "error"
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def stage_npb():
    for cond, extra in (("rownorm", {"rownorm": True}), ("wd", {"wd": 5e-4})):
        best = None
        for lr in NPB_GRID:
            out = os.path.join(RES, "npb", "lr", cond, f"lr{lr:g}.json")
            guarded(run_h3, {"company": "dense", "density": 1.0, "seed": 0, "learner": "np", "lr": lr, "epochs": 5, "probe_every": 0, **extra}, out)
            if os.path.exists(out):
                r = json.load(open(out, encoding="utf-8"))
                if not r.get("failed") and (best is None or r["last3_val"] > best[1]):
                    best = (lr, r["last3_val"])
        log(f"[npb] {cond} lr -> {best}")
        if best is None:
            continue
        for s in SEEDS:
            out = os.path.join(RES, "npb", cond, f"seed{s}.json")
            guarded(run_h3, {"company": "dense", "density": 1.0, "seed": s, "learner": "np", "lr": best[0], "epochs": 15, **extra}, out)


def lr_file():
    return os.path.join(RES, "lr_select.json")


def stage_lr():
    sel = json.load(open(lr_file(), encoding="utf-8")) if os.path.exists(lr_file()) else {}
    for lname in ("bp", "np", "fg"):
        if lname in sel:
            continue
        best = None
        for lr in LR_GRID[lname]:
            out = os.path.join(RES, "iso_lr", f"{lname}_lr{lr:g}.json")
            guarded(iso.run_one, {"budget": 1024 * KB, "net": "sparse10", "learner": lname, "seed": 0, "lr": lr, "n_tasks": 2,
                                  "eval_on": "val", "policy": "max_buffer"}, out)
            if os.path.exists(out):
                r = json.load(open(out, encoding="utf-8"))
                if r.get("fits") and not r.get("failed") and (best is None or r["a_auc"] > best[1]):
                    best = (lr, r["a_auc"])
        if best:
            sel[lname] = {"lr": best[0], "val_auc_2tasks": best[1]}
            json.dump(sel, open(lr_file(), "w", encoding="utf-8"), indent=1)
        log(f"[lr] {lname} -> {best}")
    sel["bp_mem"] = sel.get("bp", {"lr": 0.03})            # 같은 기울기·같은 갱신 (메모리만 다름)
    json.dump(sel, open(lr_file(), "w", encoding="utf-8"), indent=1)


def stage_main():
    sel = json.load(open(lr_file(), encoding="utf-8")) if os.path.exists(lr_file()) else {}
    combos = [(b, "sparse10") for b in BUDGETS_PRIORITY] + [(4096 * KB, "dense"), (1024 * KB, "dense")]
    for budget, net in combos:
        for s in SEEDS:
            for lname in LEARNERS:
                lr = sel.get(lname, {}).get("lr", LR_GRID["bp" if lname == "bp_mem" else lname][1])
                out = os.path.join(RES, "iso", f"{budget // KB}KB", net, lname, f"seed{s}.json")
                guarded(iso.run_one, {"budget": budget, "net": net, "learner": lname, "seed": s, "lr": lr, "policy": "max_buffer"}, out)


# ---------------------------------------------------------------------------
def summarize():
    rows = {}
    for p in glob.glob(os.path.join(RES, "iso", "*", "*", "*", "seed*.json")):
        parts = p.split(os.sep)
        key = (parts[-4], parts[-3], parts[-2])
        rows.setdefault(key, []).append(json.load(open(p, encoding="utf-8")))
    L = ["## P4-A 같은 메모리 상한 (Split CIFAR-10 온라인 클래스 증분, CNN16; 시험 정확도 %, 시드 평균±표준편차)", "",
         "| 상한 | 망 | 학습기 | 들어감 | 배치 b / 재현 r / 버퍼 | A_AUC | A_last | 망각 | 순전파 등가/표본 | 시간(초) |", "|---|---|---|---|---|---|---|---|---|---|"]
    def kb(s):
        return int(s.replace("KB", ""))
    for key in sorted(rows, key=lambda k: (kb(k[0]), k[1], k[2])):
        rs = rows[key]
        fit = [r for r in rs if r.get("fits")]
        if not fit:
            L.append(f"| {key[0]} | {key[1]} | {key[2]} | 아니오 | - | - | - | - | - | - |")
            continue
        pl = fit[0]["plan"]
        auc = [100 * r["a_auc"] for r in fit]
        last = [100 * r["a_last"] for r in fit]
        fg_ = [100 * r["forgetting"] for r in fit]
        fe = fit[0]["fwd_eq_per_update"] / max(fit[0]["samples_per_update"], 1)
        fails = sum(r.get("failed", False) for r in fit)
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | 예{' (실패 ' + str(fails) + ')' if fails else ''} | {pl['b']} / {pl['r']} / {pl['replay_cap']} | "
                 f"{np.mean(auc):.1f}±{np.std(auc):.1f} (n{len(fit)}) | {np.mean(last):.1f}±{np.std(last):.1f} | {np.mean(fg_):.1f} | {fe:.1f} | "
                 f"{np.mean([r['elapsed_s'] for r in fit]):.0f} |")
    npb = ["", "## P4-B 노드 섭동 안정성 대조 (MNIST 밀집 MLP, v3 프로토콜, 시험 정확도 %)", "", "| 조건 | 정확도 | 실패 | 학습률 |", "|---|---|---|---|"]
    for cond in ("rownorm", "wd"):
        rs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(RES, "npb", cond, "seed*.json")))]
        if rs:
            a = [100 * r["last3_test"] for r in rs]
            npb.append(f"| 밀집 + {'행별 가중치 정규화' if cond == 'rownorm' else '가중치 감쇠 5e-4'} | {np.mean(a):.1f}±{np.std(a):.1f} (n{len(rs)}) | "
                       f"{sum(r['failed'] for r in rs)}/{len(rs)} | {rs[0]['lr_selected']:g} |")
    md = "\n".join(L + npb)
    open(os.path.join(RES, "tables.md"), "w", encoding="utf-8").write(md)
    print(md)


STAGES = {"npb": stage_npb, "lr": stage_lr, "main": stage_main, "summarize": summarize}

if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    for name in sys.argv[1:]:
        log(f"==== stage {name} start {time.strftime('%H:%M:%S')}")
        STAGES[name]()
        log(f"==== stage {name} end {time.strftime('%H:%M:%S')}")
