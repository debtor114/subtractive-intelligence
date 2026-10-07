# -*- coding: utf-8 -*-
"""H2 v3 프로토콜 (코드 검토 반영, 2026-10-07 오후). v1/v2 와 다른 점:
  - 검증 분할: 훈련 55k / 검증 5k / 시험 10k. 학습률 선택·마지막 3 평가(last3_val)·발산 판정은 검증, 시험은 보고에만.
  - 학습률 선택도 같은 스케줄(코사인, 5 에폭)로 돌린다 (v2 는 3 에폭 상수 lr 로 골라 15 에폭 코사인에서 발산).
  - 발산하면 처음부터 다시 돌리지 않고(lr/3 재시도 X), 마지막 체크포인트(100 스텝마다)로 되돌린 뒤 기본 lr 을 절반으로 줄여
    같은 자리에서 계속 간다 (backoff, 최대 5 회). 선택 lr 대역을 벗어나지 않는다.
  - 학습 0걸음 정확도(acc0)를 곡선에 넣는다 (슈퍼마스크 효과 정량화).
  - 회사 추가: random_degree (차수 보존 무작위), pruned_reinit (학습 마스크 + 다른 초기값). 학습기: bp, dfa, fg, np (wp 는 fg 와 같은 방향을 쓰므로 제외).
  - 죽은 뉴런의 편향은 섭동·갱신하지 않는다 (learners.param_masks).
결과 JSON: results/p3/<sub>/<company>_d<density>/<learner>/seed<k>.json
"""
from __future__ import annotations

import copy
import json
from collections import deque
import math
import os
import sys
import time
from typing import Dict, List

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.p3.common import (RES_P3, alive_hidden_neurons, build_company, count_active, evaluate,  # noqa: E402
                                   get_mnist_split, hidden_neurons)
from experiments.p3.h2 import cosine_lr, params_finite                                            # noqa: E402
from experiments.p3.learners import make_learner, probe                                            # noqa: E402
from utils.seed import set_seed                                                                     # noqa: E402
from utils.tensor_data import TensorBatches                                                         # noqa: E402

BS = 128
EVAL_EVERY = 100
MAX_BACKOFF = 5
LOSS_BLOWUP = 20.0       # MNIST CE 는 우연 수준에서 2.3. 배치 손실이 20 을 넘으면 발산으로 본다 (v3.1)
RING = 3                 # 체크포인트 3 개(최대 300 스텝 전)를 들고 있다가 가장 오래된 것으로 되돌린다 (v3.1)


def out_path(company: str, density: float, learner: str, seed: int, sub: str = "h3") -> str:
    return os.path.join(RES_P3, sub, f"{company}_d{density:g}", learner, f"seed{seed}.json")


def run_h3(cfg: dict, log=print) -> dict:
    company, density = cfg["company"], float(cfg.get("density", 1.0))
    seed, lname, lr0 = int(cfg["seed"]), cfg["learner"], float(cfg["lr"])
    epochs = int(cfg.get("epochs", 15))
    probe_every = int(cfg.get("probe_every", 300))
    device = torch.device("cuda")
    x_tr, y_tr, x_val, y_val, x_te, y_te = get_mnist_split(device)
    model, info = build_company(company, density, seed, device)
    set_seed(seed * 1000 + 7)
    learner = make_learner(lname, model, lr0, dict(cfg, seed=seed))
    n_train = x_tr.shape[0]
    steps_per_epoch = math.ceil(n_train / BS)
    total = epochs * steps_per_epoch
    curve: List[Dict] = []
    probes: List[Dict] = []
    base_lr, backoffs, failed = lr0, 0, False
    t0 = time.time()

    def snapshot():
        return (copy.deepcopy(model.state_dict()), copy.deepcopy(learner.opt.state_dict()))

    def record(step, samples, loss):
        va, te = evaluate(model, x_val, y_val), evaluate(model, x_te, y_te)
        curve.append({"step": step, "samples": samples, "val_acc": va, "test_acc": te, "loss": loss, "base_lr": base_lr})
        return va, te

    record(0, 0, float("nan"))                                       # 학습 0걸음
    ckpts = deque(maxlen=RING)
    ckpts.append((0, 0, snapshot()))
    backoff_log: List[Dict] = []
    step, samples = 0, 0
    model.train()
    while step < total and not failed:
        diverged_now = False
        for idx in TensorBatches(torch.arange(n_train, device=device), BS):
            xb, yb = x_tr[idx], y_tr[idx]
            if probe_every and step % probe_every == 0 and step > 0:
                pr = probe(learner, xb, yb)
                pr["step"] = step
                probes.append(pr)
            for g in learner.opt.param_groups:
                g["lr"] = cosine_lr(step, total, steps_per_epoch, base_lr)
            loss = learner.step(xb, yb)
            step += 1
            samples += int(idx.numel())
            if (not math.isfinite(loss)) or loss > LOSS_BLOWUP or (step % EVAL_EVERY == 0 and not params_finite(model)):
                diverged_now = True
                break
            if step % EVAL_EVERY == 0 or step == total:
                va, te = record(step, samples, loss)
                ckpts.append((step, samples, snapshot()))
                if step % (EVAL_EVERY * 10) == 0 or step == total:
                    log(f"  [{company} d={density:g} {lname} lr={base_lr:g} s{seed}] step {step}/{total} val {va:.4f} test {te:.4f} "
                        f"loss {loss:.3f} bo {backoffs} ({time.time() - t0:.0f}s)")
            if step >= total:
                break
        if diverged_now:
            backoffs += 1
            if backoffs > MAX_BACKOFF:
                failed = True
                log(f"  [{company} d={density:g} {lname} s{seed}] FAILED after {MAX_BACKOFF} backoffs at step {step}")
                break
            # v3.1: 직전 체크포인트는 이미 '터지기 직전(유한하지만 큰 가중치)' 일 수 있어 가장 오래된 링 체크포인트로 되돌린다
            div_step = step
            s0, n0, (ms, os_) = ckpts[0]
            model.load_state_dict(ms)
            learner.opt.load_state_dict(os_)
            step, samples = s0, n0
            while len(curve) > 1 and curve[-1]["step"] > s0:
                curve.pop()
            ckpts.clear()
            ckpts.append((s0, n0, (ms, os_)))
            base_lr *= 0.5
            backoff_log.append({"diverged_at": div_step, "restored_to": s0, "new_base_lr": base_lr})
            log(f"  [{company} d={density:g} {lname} s{seed}] diverged at step {div_step}: restore to {s0} + lr -> {base_lr:g} (backoff {backoffs})")
            model.train()
    last3 = curve[-3:]
    res = {
        "cfg": dict(cfg), "info": info, "protocol": "v3",
        "acc0_val": curve[0]["val_acc"], "acc0_test": curve[0]["test_acc"],
        "last3_val": sum(c["val_acc"] for c in last3) / len(last3), "last3_test": sum(c["test_acc"] for c in last3) / len(last3),
        "final_val": curve[-1]["val_acc"], "final_test": curve[-1]["test_acc"],
        "best_val": max(c["val_acc"] for c in curve), "test_at_best_val": max(curve, key=lambda c: c["val_acc"])["test_acc"],
        "backoffs": backoffs, "lr_selected": lr0, "lr_final": base_lr, "failed": failed, "backoff_log": backoff_log,
        "protocol_rev": "v3.1",
        "active_weights": count_active(model), "hidden_neurons": hidden_neurons(model), "alive_hidden": alive_hidden_neurons(model),
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_perturbed": int(sum((learner.masks[id(p)] != 0).sum().item() if id(p) in learner.masks else p.numel() for p in model.parameters())),
        "curve": curve, "probes": probes, "elapsed_s": time.time() - t0, "total_steps": total,
    }
    out = cfg.get("out") or out_path(company, density, lname, seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log(f"  -> {os.path.relpath(out, REPO_ROOT)} val {res['last3_val']:.4f} test {res['last3_test']:.4f} acc0 {res['acc0_test']:.3f} "
        f"backoffs {backoffs} perturbed {res['n_perturbed']:,} ({res['elapsed_s']:.0f}s)")
    return res
