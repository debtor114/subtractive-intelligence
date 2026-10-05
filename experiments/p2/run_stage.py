# -*- coding: utf-8 -*-
"""논문 2 탐색 단계 실행기. 순차, 완료된 런은 건너뜀, 마감(results/p2/_deadline.txt, epoch 초) 지나면 새 런을 시작하지 않음.

  python experiments/p2/run_stage.py e2      # 발달 순서 (기준 런 포함)
  python experiments/p2/run_stage.py e4      # 타일 맞춤 (마스크 저장)
  python experiments/p2/run_stage.py e3      # 두 물결, 진도 맞춤
  python experiments/p2/run_stage.py x1      # 활동 희소성 x 연결 희소성 (MNIST)
런이 오류로 죽으면 기록하고 다음으로. 같은 단계에서 3 번 연속 실패하면 단계를 건너뛴다.
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

from experiments.p2.train import RES_P2, run   # noqa: E402

SEEDS = [0, 1, 2]
DENS = {"mnist": [0.01, 0.005], "cnn": [0.03, 0.01]}
LOWEST = {"mnist": 0.005, "cnn": 0.01}
P1_REF = {"mnist": os.path.join(REPO_ROOT, "results", "core"), "cnn": os.path.join(REPO_ROOT, "results", "core_cifar")}
FAIL_LOG = os.path.join(RES_P2, "_failures.log")


def deadline_passed() -> bool:
    p = os.path.join(RES_P2, "_deadline.txt")
    if not os.path.exists(p):
        return False
    return time.time() > float(open(p).read().strip())


def out_path(stage, model, density, cond, seed, extra=""):
    return os.path.join(RES_P2, stage, model, f"d{density:g}", cond + extra, f"seed{seed}.json")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def reference_targets(model, density, seed, stage_dir="e2"):
    """층별 최종 밀도: CNN 은 논문 1 의 pd_mag_global 결과, MNIST 는 p2 기준 런(global) 에서 가져온다."""
    if model == "cnn":
        p = os.path.join(P1_REF["cnn"], f"d{density:g}", "pd_mag_global", f"seed{seed}.json")
        if os.path.exists(p):
            return load(p)["layer_densities"], p
    p = out_path(stage_dir, model, density, "global", seed)
    if os.path.exists(p):
        return load(p)["layer_densities"], p
    return None, None


class Stage:
    def __init__(self, name):
        self.name, self.fails, self.skipped = name, 0, False

    def run_one(self, cfg):
        if os.path.exists(cfg["out"]):
            return "exists"
        if self.skipped:
            return "stage-skipped"
        if deadline_passed():
            print(f"[deadline] skip {cfg['out']}", flush=True)
            return "deadline"
        try:
            run(cfg, log=lambda s: print(s, flush=True))
            self.fails = 0
            return "ok"
        except Exception:
            self.fails += 1
            msg = f"[error] {self.name} {cfg.get('model')} {cfg.get('condition')} d={cfg.get('density')} s={cfg.get('seed')}\n{traceback.format_exc()}"
            print(msg, flush=True)
            with open(FAIL_LOG, "a", encoding="utf-8") as f:
                f.write(time.strftime("%H:%M:%S ") + msg + "\n")
            if self.fails >= 3:
                self.skipped = True
                print(f"[skip-stage] {self.name}: 3 consecutive failures", flush=True)
            return "error"
        finally:
            gc.collect()
            torch.cuda.empty_cache()


def stage_e2(st: Stage):
    for model in ("mnist", "cnn"):
        for d in DENS[model]:
            for s in SEEDS:
                if model == "mnist":   # 기준 런 (논문 1 과 같은 프로토콜의 전역 크기 가지치기)
                    st.run_one({"model": model, "condition": "global", "density": d, "seed": s,
                                "out": out_path("e2", model, d, "global", s), "mask_tag": f"e2_{model}_d{d:g}_global_s{s}"})
                targets, src = reference_targets(model, d, s)
                if targets is None:
                    print(f"[e2] no reference targets for {model} d={d} s={s}, skipping conditions", flush=True)
                    continue
                for cond in ("sync", "bottom_up", "top_down"):
                    st.run_one({"model": model, "condition": cond, "density": d, "seed": s, "targets": targets,
                                "targets_from": src, "out": out_path("e2", model, d, cond, s),
                                "mask_tag": f"e2_{model}_d{d:g}_{cond}_s{s}"})


def stage_e4(st: Stage):
    for model in ("mnist", "cnn"):
        for d in DENS[model]:
            for s in SEEDS:
                for cond in ("block16_during", "block16_oneshot"):
                    st.run_one({"model": model, "condition": cond, "density": d, "seed": s,
                                "out": out_path("e4", model, d, cond, s), "mask_tag": f"e4_{model}_d{d:g}_{cond}_s{s}"})


def stage_e4pl(st: Stage):
    """E4 추가: 층별 타일 예산 (E2 기준 런의 층별 최종 밀도) 으로 한 번에 / 학습 중 타일 가지치기. 전역 타일 점수가 층을 통째로 비워
    경로를 끊는 문제 (one-shot 이 우연 수준) 를 층별 배분으로 분리한다."""
    for model in ("mnist", "cnn"):
        for d in DENS[model]:
            for s in SEEDS:
                targets, src = reference_targets(model, d, s)
                if targets is None:
                    continue
                for cond in ("block16_oneshot_pl", "block16_during_pl"):
                    st.run_one({"model": model, "condition": cond, "density": d, "seed": s, "targets": targets, "targets_from": src,
                                "out": out_path("e4", model, d, cond, s), "mask_tag": f"e4_{model}_d{d:g}_{cond}_s{s}"})


def stage_e3(st: Stage):
    for model in ("mnist", "cnn"):
        d = LOWEST[model]
        for s in SEEDS:
            targets, src = reference_targets(model, d, s)
            if targets is None:
                print(f"[e3] no reference targets for {model} d={d} s={s}", flush=True)
                continue
            for cond in ("two_waves", "progress_gated"):
                st.run_one({"model": model, "condition": cond, "density": d, "seed": s, "targets": targets,
                            "targets_from": src, "out": out_path("e3", model, d, cond, s)})


def stage_x1(st: Stage):
    """활동 희소성 (k-WTA) x 연결 희소성: MNIST, 연결 0.5%, k = 1.0 / 0.1 / 0.05, 팔 = dense big / dense small / 학습 중 가지치기."""
    d = 0.005
    for s in SEEDS:
        for k in (1.0, 0.1, 0.05):
            for cond, dens in (("dense", 1.0), ("dense_small", d), ("global", d)):
                st.run_one({"model": "mnist", "condition": cond, "density": dens, "seed": s, "kwta": k,
                            "out": out_path("x1", "mnist", dens, cond, s, extra=f"_k{k:g}")})


STAGES = {"e2": stage_e2, "e4": stage_e4, "e4pl": stage_e4pl, "e3": stage_e3, "x1": stage_x1}

if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(f"==== stage {name} start {time.strftime('%H:%M:%S')}", flush=True)
        st = Stage(name)
        STAGES[name](st)
        print(f"==== stage {name} end {time.strftime('%H:%M:%S')}", flush=True)
