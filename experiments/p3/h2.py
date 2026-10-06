# -*- coding: utf-8 -*-
"""H2 파일럿: 세 회사(dense / pruned / random_mask / dense_small) × 학습기 사다리(bp/dfa/fg/wp/np) × 시드.
고정 마스크로 처음부터 15 에폭 학습, 100 스텝마다 시험 정확도, probe_every 스텝마다 SNR 탐침.
결과 JSON: results/p3/h2/<company>_d<density>/<learner>/seed<k>.json
"""
from __future__ import annotations

import json
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
                                   get_mnist, hidden_neurons)
from experiments.p3.learners import make_learner, probe                                            # noqa: E402
from utils.seed import set_seed                                                                     # noqa: E402
from utils.tensor_data import TensorBatches                                                         # noqa: E402

BS = 128
EVAL_EVERY = 100


def out_path(company: str, density: float, learner: str, seed: int, sub: str = "h2") -> str:
    return os.path.join(RES_P3, sub, f"{company}_d{density:g}", learner, f"seed{seed}.json")


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    return 0.5 * base * (1.0 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def params_finite(model) -> bool:
    return all(torch.isfinite(p).all().item() for p in model.parameters())


def run_h2(cfg: dict, log=print) -> dict:
    company, density = cfg["company"], float(cfg.get("density", 1.0))
    seed, lname, lr = int(cfg["seed"]), cfg["learner"], float(cfg["lr"])
    epochs = int(cfg.get("epochs", 15))
    probe_every = int(cfg.get("probe_every", 300))
    sched = cfg.get("sched", "const")            # v1: const. v2: cosine (1 에폭 워밍업) — 상수 lr 로는 섭동 학습기가 뒤늦게 발산
    diverged, diverged_step = False, None
    device = torch.device("cuda")
    x_tr, y_tr, x_te, y_te = get_mnist(device)
    model, info = build_company(company, density, seed, device)
    set_seed(seed * 1000 + 7)                     # 학습기 난수(섭동·DFA 행렬·배치 순서)는 초기화와 분리
    learner = make_learner(lname, model, lr, dict(cfg, seed=seed))
    n_train = x_tr.shape[0]
    steps_per_epoch = math.ceil(n_train / BS)
    total = epochs * steps_per_epoch
    curve: List[Dict] = []
    probes: List[Dict] = []
    step, samples, t0 = 0, 0, time.time()
    model.train()
    done = False
    while not done:
        for idx in TensorBatches(torch.arange(n_train, device=device), BS):
            xb, yb = x_tr[idx], y_tr[idx]
            if probe_every and step % probe_every == 0:
                pr = probe(learner, xb, yb)
                pr["step"] = step
                probes.append(pr)
            if sched == "cosine":
                for g in learner.opt.param_groups:
                    g["lr"] = cosine_lr(step, total, steps_per_epoch, lr)
            loss = learner.step(xb, yb)
            step += 1
            samples += int(idx.numel())
            if not math.isfinite(loss):
                diverged, diverged_step = True, step
            if step % EVAL_EVERY == 0 or step == total or diverged:
                if not diverged and not params_finite(model):
                    diverged, diverged_step = True, step
                acc = evaluate(model, x_te, y_te) if not diverged else 0.0
                curve.append({"step": step, "samples": samples, "test_acc": acc, "loss": loss})
                if step % (EVAL_EVERY * 10) == 0 or step == total or diverged:
                    log(f"  [{company} d={density:g} {lname} lr={lr:g} s{seed}] step {step}/{total} acc {acc:.4f} "
                        f"loss {loss:.3f} ({time.time() - t0:.0f}s)" + (" DIVERGED" if diverged else ""))
            if step >= total or diverged:
                done = True
                break
    final = curve[-1]["test_acc"]
    res = {
        "cfg": dict(cfg), "info": info, "final_acc": final, "best_acc": max(c["test_acc"] for c in curve),
        "diverged": diverged, "diverged_step": diverged_step,
        "last3_mean": sum(c["test_acc"] for c in curve[-3:]) / len(curve[-3:]),
        "active_weights": count_active(model), "hidden_neurons": hidden_neurons(model),
        "alive_hidden": alive_hidden_neurons(model), "n_params": sum(p.numel() for p in model.parameters()),
        "curve": curve, "probes": probes, "elapsed_s": time.time() - t0, "total_steps": total,
    }
    out = cfg.get("out") or out_path(company, density, lname, seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log(f"  -> {os.path.relpath(out, REPO_ROOT)} final {final:.4f} alive {res['alive_hidden']}/{res['hidden_neurons']} "
        f"active {res['active_weights']:,} ({res['elapsed_s']:.0f}s)")
    return res


if __name__ == "__main__":
    # 연기 시험: python experiments/p3/h2.py <company> <density> <learner> [epochs]
    c, d, l = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    ep = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    lr = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-2
    run_h2({"company": c, "density": d, "learner": l, "seed": 0, "lr": lr, "epochs": ep, "probe_every": 100,
            "out": os.path.join(RES_P3, "_smoke", f"{c}_d{d:g}_{l}_lr{lr:g}.json")})
