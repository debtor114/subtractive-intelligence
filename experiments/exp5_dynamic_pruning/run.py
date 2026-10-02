# -*- coding: utf-8 -*-
"""실험 5: 추론 중 가지치기 (입력 의존적 동적 희소화), CIFAR-10 ResNet-18.

  python experiments/exp5_dynamic_pruning/run.py --arm dyn_local --density 0.05 --seed 0

arm (core/dynamic_layers.py):
  dyn_local  : 연결 단위, 국소 규칙 (|w| x 입력 채널 크기, 출력 뉴런별 상위 k). 표본마다 다른 서브망.
  dyn_random : 연결 단위, 표본마다 무작위 k (선택 기준 대조군).
  kwta_in    : 뉴런(채널) 단위 k-WTA 입력 게이팅 (구조적, 실제 속도가 나는 형태).
정적 대조군 (학습 중 전역 크기 가지치기, dense small, RigL, 학습 후 가지치기) 은 results/core_resnet 의 같은 밀도 결과를 쓴다.
스케줄: 밀도 1.0 -> density 를 학습의 10%~70% 구간에서 cubic 으로 내린다 (학습 중 가지치기와 같은 일정). 첫 conv 와 fc 는 밀집 유지.
학습: run_cifar.py 와 동일 (SGD nesterov lr 0.1 cosine, wd 5e-4, 배치 128, AMP, 20 에폭, GPU 증강).
지표: 정확도, 표본당 기대 FLOPs (dense x 층별 비율), 보정, 데이터 효율, 마스크 통계 (클래스 안/밖 자카드, 합집합 커버리지).
결과: results/exp5_dynamic/d<density>/<arm>/seed<seed>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baselines.resnet import prunable_weights as resnet_weights, resnet18                      # noqa: E402
from core.dynamic_layers import (convert_to_dynamic, expected_active_weights, layer_fractions,   # noqa: E402
                                 mask_statistics, set_dynamic_density)
from core.pruning import cubic_density                                                          # noqa: E402
from experiments.core_prune_during_learning.run_cifar import cosine_lr, effective_fwd_flops, eval_probs   # noqa: E402
from utils.cifar_gpu import augment, load_cifar10_gpu, normalize                                # noqa: E402
from utils.metrics import calibration_metrics, count_flops, data_efficiency                    # noqa: E402
from utils.seed import set_seed                                                                # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "exp5_dynamic")
STAT_LAYERS = ("layers.1.conv2", "layers.4.conv2", "layers.7.conv2")   # 1 단계 / 3 단계 / 4 단계 블록의 둘째 conv
ARMS = ("dyn_local", "dyn_random", "kwta_in")


def run_one(cfg: dict, log=print) -> dict:
    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device("cuda")
    (x_tr, y_tr), (x_te, y_te), (mean, std) = load_cifar10_gpu(device=device)
    y_te_np = y_te.cpu().numpy()

    arm: str = cfg["arm"]
    assert arm in ARMS, arm
    density = float(cfg["density"])
    epochs = int(cfg.get("epochs", 20))
    bs = int(cfg.get("batch_size", 128))
    lr = float(cfg.get("lr", 0.1))
    wd = float(cfg.get("weight_decay", 5e-4))
    every = int(cfg.get("update_every", 100))
    # 연결 단위 동적 경로는 평가도 느리다 (표본별 가중치): 학습 중에는 테스트 앞 2000 장으로 200 스텝마다, 마지막만 전체 1 만 장
    dyn_conn = arm != "kwta_in"
    eval_every = int(cfg.get("eval_every", 200 if dyn_conn else 100))
    eval_subset = int(cfg.get("eval_subset", 2000 if dyn_conn else 10000))
    n_train = x_tr.shape[0]
    steps_per_epoch = math.ceil(n_train / bs)
    total_steps = int(cfg.get("max_steps") or epochs * steps_per_epoch)
    warmup = min(steps_per_epoch, total_steps // 10)
    begin = int(float(cfg.get("prune_begin", 0.1)) * total_steps)
    end = int(float(cfg.get("prune_end", 0.7)) * total_steps)
    # 연결 단위 동적 경로는 표본별 가중치 (B x params) 를 만들므로 평가 배치를 줄인다 (1000 이면 층 하나에 4.7GB)
    eval_bs = int(cfg.get("eval_bs", 1000 if arm == "kwta_in" else 125))

    model = convert_to_dynamic(resnet18(1.0), arm).to(device)
    big_n = resnet_weights(resnet18(1.0))
    leaf_dense = count_flops(model, (3, 32, 32), device).leaf_dense      # 밀도 1.0 이라 dense 경로로 센다

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                          lr=lr, momentum=0.9, nesterov=True)
    scaler = torch.amp.GradScaler("cuda")
    curve: List[Dict] = []
    sched_log: List[Dict] = []
    state = {"step": 0, "cum_flops": 0.0, "samples": 0}
    t0 = time.time()

    def record(extra=None, full: bool = False):
        n_eval = x_te.shape[0] if full else min(eval_subset, x_te.shape[0])
        p = eval_probs(model, x_te[:n_eval], mean, std, bs=eval_bs)
        acc = float((p.argmax(1).cpu().numpy() == y_te_np[:n_eval]).mean())
        rec = {"step": state["step"], "samples_seen": state["samples"], "test_acc": acc, "n_eval": n_eval,
               "active": expected_active_weights(model), "cum_train_flops": state["cum_flops"],
               "density": model_density()}
        if extra:
            rec.update(extra)
        curve.append(rec)
        return acc

    def model_density() -> float:
        return expected_active_weights(model) / big_n

    model.train()
    done = False
    while not done:
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, bs):
            if state["step"] >= total_steps:
                done = True
                break
            idx = perm[s:s + bs]
            for g in opt.param_groups:
                g["lr"] = cosine_lr(state["step"], total_steps, warmup, lr)
            xb = normalize(augment(x_tr[idx]), mean, std)
            yb = y_tr[idx]
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            fr = layer_fractions(model)
            state["cum_flops"] += 3.0 * idx.numel() * effective_fwd_flops(leaf_dense, fr)
            state["samples"] += idx.numel()
            state["step"] += 1
            st = state["step"]
            if st % every == 0 and begin <= st <= end:
                d_now = cubic_density(st, begin, end, 1.0, density)
                set_dynamic_density(model, d_now)
                sched_log.append({"step": st, "density": d_now, "active": expected_active_weights(model)})
            if st % eval_every == 0:
                acc = record({"phase": "main"})
                if st % (eval_every * 10) == 0:
                    log(f"  [{arm} d={density:g} s{seed}] step {st}/{total_steps} acc {acc:.4f} "
                        f"active {expected_active_weights(model):,.0f} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
                model.train()
    if sched_log and sched_log[-1]["density"] > density:   # 마지막 갱신이 end 에 못 미쳤으면 목표 밀도로 맞춘다
        set_dynamic_density(model, density)
        sched_log.append({"step": state["step"], "density": density, "active": expected_active_weights(model)})
    elif not sched_log:
        set_dynamic_density(model, density)

    final_acc = record({"phase": "final"}, full=True)
    probs = eval_probs(model, x_te, mean, std, bs=eval_bs).cpu().numpy()
    cal = calibration_metrics(probs, y_te_np)
    main_curve = [c for c in curve if c.get("phase") in ("main", "final")]
    xs = [c["samples_seen"] for c in main_curve]
    ys = [c["test_acc"] for c in main_curve]
    # 마스크 통계: 클래스별 n 개 테스트 표본
    n_per = int(cfg.get("stat_per_class", 20))
    sel = torch.cat([torch.nonzero(y_te == c).flatten()[:n_per] for c in range(10)])
    stats = mask_statistics(model, STAT_LAYERS, normalize(x_te[sel], mean, std), y_te[sel])
    fr = layer_fractions(model)
    out = {
        "cfg": cfg, "arm": arm, "density": density, "seed": seed, "model": {"name": "resnet18", "dynamic": arm},
        "final_active": expected_active_weights(model), "final_density_vs_big": model_density(), "layer_fractions": fr,
        "final_acc": final_acc, "best_acc": max(c["test_acc"] for c in curve), "eval_subset": eval_subset,
        "infer_flops": effective_fwd_flops(leaf_dense, fr), "dense_flops": float(sum(leaf_dense.values())),
        "cum_train_flops": state["cum_flops"], "samples_seen": state["samples"], "steps": state["step"],
        "calibration": cal, "data_efficiency_80": data_efficiency(xs, ys, 0.80), "data_efficiency_85": data_efficiency(xs, ys, 0.85),
        "mask_stats": stats, "curve": curve, "sched_log": sched_log, "time_s": time.time() - t0,
    }
    d = os.path.join(RESULTS_DIR, f"d{density:g}", arm)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    s4 = stats.get("layers.7.conv2", {})
    log(f"[done] exp5 {arm} d={density:g} seed {seed}: acc {final_acc:.4f} (best {out['best_acc']:.4f}) "
        f"active/sample {out['final_active']:,.0f} infer {out['infer_flops']:.3e} train {state['cum_flops']:.3e} ece {cal['ece']:.3f} "
        f"| stage4 J_same {s4.get('jaccard_same_class', float('nan')):.3f} J_diff {s4.get('jaccard_diff_class', float('nan')):.3f} "
        f"coverage {s4.get('union_coverage', float('nan')):.3f} ({out['time_s']:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = {"arm": a.arm, "density": a.density, "seed": a.seed, "epochs": a.epochs}
    for p in a.set:
        k, v = p.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    run_one(cfg)


if __name__ == "__main__":
    main()
