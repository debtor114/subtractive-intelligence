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


V2_ARMS = [("pruned", 0.005), ("dense_small", 0.005), ("random_mask", 0.005), ("dense", 1.0),
           ("pruned", 0.01), ("dense_small", 0.01), ("random_mask", 0.01)]
V2_LEARNERS = ["bp", "dfa", "np", "fg", "wp"]


def stage_h2v2(st: Stage):
    """v2 프로토콜 (2026-10-07 06:15): v1 은 상수 lr 로 섭동 학습기가 2,500~6,700 스텝에서 NaN 발산.
    코사인 스케줄(1 에폭 워밍업) + 발산 시 lr/3, lr/10 로 재시도(최대 3 회). 결과는 results/p3/h2v2/."""
    import shutil
    sel = json.load(open(LR_FILE, encoding="utf-8")) if os.path.exists(LR_FILE) else {}
    for seed in SEEDS:
        for company, d in V2_ARMS:
            for lname in V2_LEARNERS:
                out = out_path(company, d, lname, seed, sub="h2v2")
                if os.path.exists(out):
                    continue
                lr0 = sel.get(lr_key(company, d, lname), {}).get("lr", LR_GRID[lname][len(LR_GRID[lname]) // 2])
                for attempt, lr in enumerate([lr0, lr0 / 3.0, lr0 / 10.0]):
                    tmp = out.replace(".json", f"_try{attempt}.json")
                    if not os.path.exists(tmp):
                        status = st.run_one(run_h2, {"company": company, "density": d, "seed": seed, "learner": lname,
                                                     "lr": lr, "sched": "cosine", "attempt": attempt}, tmp)
                        if status in ("deadline", "stage-skipped", "error"):
                            break
                    r = json.load(open(tmp, encoding="utf-8"))
                    ok = (not r.get("diverged")) and (r["final_acc"] >= 0.15 or r["best_acc"] < 0.3)
                    if ok or attempt == 2:
                        shutil.copyfile(tmp, out)
                        log(f"[v2] {lr_key(company, d, lname)} s{seed}: lr {lr:g} (attempt {attempt}) final {r['final_acc']:.4f}"
                            + (" diverged" if r.get("diverged") else ""))
                        break


def _lr_select_for(st: Stage, arms):
    sel = json.load(open(LR_FILE, encoding="utf-8")) if os.path.exists(LR_FILE) else {}
    for company, d in arms:
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


def _v2_runs_for(st: Stage, arms, seeds=SEEDS, learners=V2_LEARNERS, epochs=15, sub="h2v2"):
    import shutil
    sel = json.load(open(LR_FILE, encoding="utf-8")) if os.path.exists(LR_FILE) else {}
    for seed in seeds:
        for company, d in arms:
            for lname in learners:
                out = out_path(company, d, lname, seed, sub=sub)
                if os.path.exists(out):
                    continue
                lr0 = sel.get(lr_key(company, d, lname), {}).get("lr", LR_GRID[lname][len(LR_GRID[lname]) // 2])
                for attempt, lr in enumerate([lr0, lr0 / 3.0, lr0 / 10.0]):
                    tmp = out.replace(".json", f"_try{attempt}.json")
                    if not os.path.exists(tmp):
                        status = st.run_one(run_h2, {"company": company, "density": d, "seed": seed, "learner": lname,
                                                     "lr": lr, "sched": "cosine", "attempt": attempt, "epochs": epochs}, tmp)
                        if status in ("deadline", "stage-skipped", "error"):
                            break
                    r = json.load(open(tmp, encoding="utf-8"))
                    ok = (not r.get("diverged")) and (r["final_acc"] >= 0.15 or r["best_acc"] < 0.3)
                    if ok or attempt == 2:
                        shutil.copyfile(tmp, out)
                        log(f"[v2] {lr_key(company, d, lname)} s{seed} ep{epochs}: lr {lr:g} (attempt {attempt}) final {r['final_acc']:.4f}"
                            + (" diverged" if r.get("diverged") else ""))
                        break


def stage_h2deg(st: Stage):
    """검토 반영 (2026-10-07 오전): 차수 보존 무작위 마스크 회사 (random_degree) — 학습 마스크와 모든 뉴런의 입·출력 연결 수가 같고
    '어느 쌍이 연결됐나' 만 다르다. 끊긴 뉴런 때문에 무작위 마스크가 불리했는지 가른다."""
    arms = [("random_degree", 0.005), ("random_degree", 0.01)]
    _lr_select_for(st, arms)
    _v2_runs_for(st, arms)


def stage_h2long(st: Stage):
    """검토 반영: 수렴 확인 — fg·np 를 45 에폭 (코사인) 으로, 핵심 회사 4 개, seed 0. results/p3/h2long/."""
    arms = [("pruned", 0.005), ("dense_small", 0.005), ("random_degree", 0.005), ("random_mask", 0.005), ("dense", 1.0)]
    _v2_runs_for(st, arms, seeds=[0], learners=["fg", "np"], epochs=45, sub="h2long")


def stage_h1b(st: Stage):
    """검토 반영: Wanda 충실 재현 (abs(w) x ||X_j||_2, 출력 행별 top-k) 과 행별 변형을 포함한 H1 재실행. results/p3/h1b/."""
    from experiments.p3.h1 import run_h1 as _run_h1
    for seed in SEEDS:
        st.run_one(_run_h1, {"seed": seed, "criteria": "all"}, os.path.join(RES_P3, "h1b", f"seed{seed}.json"))


STAGES = {"lr": stage_lr, "h2": stage_h2, "h1": stage_h1, "h2v2": stage_h2v2, "h2deg": stage_h2deg, "h2long": stage_h2long, "h1b": stage_h1b}

if __name__ == "__main__":
    for name in sys.argv[1:]:
        log(f"==== stage {name} start {time.strftime('%H:%M:%S')}")
        st = Stage(name)
        STAGES[name](st)
        log(f"==== stage {name} end {time.strftime('%H:%M:%S')}")
