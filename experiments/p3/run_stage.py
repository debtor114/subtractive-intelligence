# -*- coding: utf-8 -*-
"""p3 단계 실행기. 순차, 완료 런 건너뜀, 마감(results/p3/_deadline.txt, epoch 초) 지나면 새 런 시작 안 함.

  python experiments/p3/run_stage.py lr     # 학습기별 학습률 선택 (dense, 3 에폭, seed 0)
  python experiments/p3/run_stage.py h2     # 세 회사 x 학습기 x 시드 (lr_select.json 사용)
  python experiments/p3/run_stage.py h1     # 배포 뒤 결산 가지치기 (시드 3)
같은 단계에서 3 번 연속 실패하면 단계를 건너뛴다.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.p3.common import RES_P3                      # noqa: E402
from experiments.p3.h1 import run_h1                          # noqa: E402
from experiments.p3.h2 import out_path, run_h2                # noqa: E402

SEEDS = [0, 1, 2]
LEARNERS = ["bp", "dfa", "fg", "wp", "np"]
# SGD+momentum 학습률 격자 (학습기마다 추정기 크기가 달라 범위가 다르다). 선택은 회사·밀도·학습기마다 3 에폭 seed 0.
LR_GRID = {
    "bp": [3e-1, 1e-1, 3e-2, 1e-2, 3e-3],
    "dfa": [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3],
    "fg": [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
    "wp": [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
    "np": [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
}
COMPANIES = [("dense", 1.0)] + [(c, d) for d in (0.01, 0.005) for c in ("pruned", "random_mask", "dense_small")]
FAIL_LOG = os.path.join(RES_P3, "_failures.log")
LR_FILE = os.path.join(RES_P3, "lr_select.json")


def deadline_passed() -> bool:
    p = os.path.join(RES_P3, "_deadline.txt")
    if not os.path.exists(p):
        return False
    return time.time() > float(open(p).read().strip())


def log(s):
    print(s, flush=True)


class Stage:
    def __init__(self, name):
        self.name, self.fails, self.skipped = name, 0, False

    def run_one(self, fn, cfg, out):
        if os.path.exists(out):
            return "exists"
        if self.skipped:
            return "stage-skipped"
        if deadline_passed():
            log(f"[deadline] skip {out}")
            return "deadline"
        try:
            fn(dict(cfg, out=out), log=log)
            self.fails = 0
            return "ok"
        except Exception:
            self.fails += 1
            msg = f"[error] {self.name} {cfg}\n{traceback.format_exc()}"
            log(msg)
            os.makedirs(RES_P3, exist_ok=True)
            with open(FAIL_LOG, "a", encoding="utf-8") as f:
                f.write(time.strftime("%H:%M:%S ") + msg + "\n")
            if self.fails >= 3:
                self.skipped = True
                log(f"[skip-stage] {self.name}: 3 consecutive failures")
            return "error"
        finally:
            gc.collect()
            torch.cuda.empty_cache()


def lr_key(company: str, d: float, lname: str) -> str:
    return f"{company}_d{d:g}/{lname}"


def stage_lr(st: Stage):
    """학습률 선택: 회사·밀도·학습기마다 3 에폭(seed 0) 격자, 마지막 3 평가 평균이 가장 높은 값."""
    sel = json.load(open(LR_FILE, encoding="utf-8")) if os.path.exists(LR_FILE) else {}
    for company, d in COMPANIES:
        for lname in LEARNERS:
            best = None
            for lr in LR_GRID[lname]:
                out = os.path.join(RES_P3, "lr", f"{company}_d{d:g}", f"{lname}_lr{lr:g}.json")
                st.run_one(run_h2, {"company": company, "density": d, "seed": 0, "learner": lname, "lr": lr, "epochs": 3,
                                    "probe_every": 0}, out)
                if os.path.exists(out):
                    acc = json.load(open(out, encoding="utf-8"))["last3_mean"]
                    if best is None or acc > best[1]:
                        best = (lr, acc)
            if best:
                sel[lr_key(company, d, lname)] = {"lr": best[0], "acc3ep": best[1]}
            with open(LR_FILE, "w", encoding="utf-8") as f:
                json.dump(sel, f, indent=1)
            log(f"[lr] {lr_key(company, d, lname)} -> {best}")


def stage_h2(st: Stage):
    sel = json.load(open(LR_FILE, encoding="utf-8")) if os.path.exists(LR_FILE) else {}
    for seed in SEEDS:
        for company, d in COMPANIES:
            for lname in LEARNERS:
                lr = sel.get(lr_key(company, d, lname), {}).get("lr", LR_GRID[lname][len(LR_GRID[lname]) // 2])
                st.run_one(run_h2, {"company": company, "density": d, "seed": seed, "learner": lname, "lr": lr},
                           out_path(company, d, lname, seed))


def stage_h1(st: Stage):
    for seed in SEEDS:
        st.run_one(run_h1, {"seed": seed}, os.path.join(RES_P3, "h1", f"seed{seed}.json"))


STAGES = {"lr": stage_lr, "h2": stage_h2, "h1": stage_h1}

if __name__ == "__main__":
    for name in sys.argv[1:]:
        log(f"==== stage {name} start {time.strftime('%H:%M:%S')}")
        st = Stage(name)
        STAGES[name](st)
        log(f"==== stage {name} end {time.strftime('%H:%M:%S')}")
