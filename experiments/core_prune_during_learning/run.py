# -*- coding: utf-8 -*-
"""핵심 실험: 학습 중 가지치기(감산) 대 가산 / 학습 후 가지치기 / 동적 희소 학습.

같은 '최종 활성 연결 수' 예산 B 에서 비교한다. MNIST, MLP 784-h-h-10, GPU 텐서 데이터.

arm (방식):
  dense_small   : 처음부터 예산 B 에 맞춘 작은 dense 망 (가산적 기준선)
  dense_big     : 과잉 초기화 망을 가지치기 없이 학습 (정확도 상한 참고, 예산 무시)
  static_sparse : 과잉 망에 무작위 고정 마스크 (밀도 d), 재성장 없음
  set / rigl    : 밀도 d 로 시작하는 동적 희소 학습 (끊고 다시 잇기: 무작위 / gradient)
  pd_<rule>_<scope> : 과잉 망을 학습 중에 cubic 스케줄로 밀도 1 -> d 로 깎음. 재성장 없음 (순수 감산).
                      rule = mag | act | actmag | random, scope = layer | global
  ttp           : 과잉 망을 dense 로 끝까지 학습 -> 한 번에 전역 크기 가지치기 -> 미세조정 (학습 후 가지치기)
  ttp_gradual   : dense 로 끝까지 학습 -> 미세조정 구간 안에서 cubic 스케줄로 점진 가지치기 (점진성 vs 학습 중 타이밍 분리 대조군)
  <arm>_x<k>    : 같은 팔을 k 배 긴 학습으로 (예: rigl_x3). 결과는 별도 팔 이름으로 저장
  --set tag=... : 결과 폴더/팔 이름에 _<tag> 를 붙여 변형 설정을 구분

기록: 정확도(최종/최고), 학습곡선(표본 수, 활성 연결, 누적 학습 FLOPs), 가지치기 궤적, 추론 FLOPs,
      확신도 보정, 데이터 효율.
FLOPs: MLP 의 샘플당 순전파 FLOPs = 2 x (활성 가중치 수). 학습 = 순전파 x 3. 누적은 매 스텝 실제 밀도로 합산.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baselines.mlp import MLP                                                     # noqa: E402
from core.masked_layers import (MASKED_TYPES, active_connections, apply_masks,   # noqa: E402
                                convert_to_masked, mask_density, masked_modules)
from core.pruning import (ActivityTracker, PruningScheduler, dynamic_sparse_step,  # noqa: E402
                          prune_to_density, random_sparse_init)
from utils.metrics import calibration_metrics, data_efficiency                   # noqa: E402
from utils.seed import set_seed                                                  # noqa: E402
from utils.tensor_data import TensorBatches, load_mnist_tensors                  # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "core")
IN_DIM, OUT_DIM = 784, 10


def n_weights(h: int) -> int:
    return IN_DIM * h + h * h + h * OUT_DIM


def dense_hidden_for_budget(budget: int) -> int:
    """784h + h^2 + 10h = budget 을 푸는 h (반올림)."""
    b = IN_DIM + OUT_DIM
    return max(1, int(round((-b + math.sqrt(b * b + 4 * budget)) / 2.0)))


@torch.no_grad()
def linear_active_weights(model: nn.Module) -> int:
    tot = 0
    for m in model.modules():
        if isinstance(m, MASKED_TYPES):
            tot += int(m.weight_mask.sum().item())
        elif isinstance(m, nn.Linear):
            tot += m.weight.numel()
    return tot


@torch.no_grad()
def eval_probs(model: nn.Module, x: torch.Tensor, bs: int = 5000) -> torch.Tensor:
    was = model.training
    model.eval()
    out = [torch.softmax(model(x[s:s + bs]).float(), 1) for s in range(0, x.shape[0], bs)]
    model.train(was)
    return torch.cat(out)


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def run_one(cfg: dict, log=print) -> dict:
    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (x_tr, y_tr), (x_te, y_te) = load_mnist_tensors(device=device)
    y_te_np = y_te.cpu().numpy()

    arm: str = cfg["arm"]
    kind, mult = arm, 1
    if "_x" in arm and arm.rsplit("_x", 1)[1].isdigit():
        kind, mult = arm.rsplit("_x", 1)[0], int(arm.rsplit("_x", 1)[1])
    tag = str(cfg.get("tag", "") or "")
    arm_out = arm + (f"_{tag}" if tag else "")
    big_h = int(cfg.get("big_hidden", 1024))
    density = float(cfg.get("density", 1.0))
    budget = int(round(density * n_weights(big_h)))
    epochs = int(cfg.get("epochs", 20)) * mult
    bs = int(cfg.get("batch_size", 128))
    lr = float(cfg.get("lr", 1e-3))
    every = int(cfg.get("update_every", 100))
    eval_every = int(cfg.get("eval_every", 100))
    steps_per_epoch = math.ceil(x_tr.shape[0] / bs)
    total_steps = epochs * steps_per_epoch
    warmup = steps_per_epoch

    # ---- 모델 조립 ----
    tracker = None
    sched = None
    dynamic = None            # ("random"|"gradient")
    ttp = kind in ("ttp", "ttp_gradual")
    sched_phase = "main"
    if kind == "dense_small":
        h = dense_hidden_for_budget(budget)
        model = MLP((1, 28, 28), 10, hidden=(h, h)).to(device)
    elif kind == "dense_big":
        model = MLP((1, 28, 28), 10, hidden=(big_h, big_h)).to(device)
    else:
        model = convert_to_masked(MLP((1, 28, 28), 10, hidden=(big_h, big_h))).to(device)
        if kind == "static_sparse":
            random_sparse_init(model, density)
        elif kind in ("set", "rigl"):
            random_sparse_init(model, density)
            dynamic = "random" if kind == "set" else "gradient"
            if dynamic == "gradient":
                for _, m in masked_modules(model):
                    m.dense_grad = True
        elif kind.startswith("pd_"):
            _, rule, scope = kind.split("_")
            rule_name = {"mag": "magnitude", "act": "activity", "actmag": "activity_mag", "random": "random",
                         "drive": "drive"}[rule]
            if rule_name.startswith("activity") or rule_name == "drive":
                tracker = ActivityTracker(model, momentum=float(cfg.get("act_momentum", 0.99)))
            begin = int(float(cfg.get("prune_begin", 0.1)) * total_steps)
            end = int(float(cfg.get("prune_end", 0.7)) * total_steps)
            sched = PruningScheduler(model, rule=rule_name, d_final=density, begin_step=begin, end_step=end,
                                     every=every, scope="global" if scope == "global" else "layerwise",
                                     tracker=tracker)
        elif ttp:
            pass  # dense 로 학습 후 아래에서 처리
        else:
            raise KeyError(f"unknown arm '{arm}'")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dyn_end = int(float(cfg.get("dyn_end", 0.75)) * total_steps)
    dyn_frac0 = float(cfg.get("dyn_frac", 0.3))

    curve: List[Dict] = []
    prune_log: List[Dict] = []
    cum_flops = 0.0
    samples_seen = 0
    step = 0
    t0 = time.time()

    def record(step_, extra=None):
        p = eval_probs(model, x_te)
        acc = float((p.argmax(1).cpu().numpy() == y_te_np).mean())
        rec = {"step": step_, "samples_seen": samples_seen, "test_acc": acc,
               "active": linear_active_weights(model), "cum_train_flops": cum_flops}
        if extra:
            rec.update(extra)
        curve.append(rec)
        return acc

    def train_steps(n_steps: int, lr_fn, phase: str):
        nonlocal step, cum_flops, samples_seen
        done = 0
        model.train()
        while done < n_steps:
            for idx in TensorBatches(torch.arange(x_tr.shape[0], device=device), bs):
                if done >= n_steps:
                    break
                for g in opt.param_groups:
                    g["lr"] = lr_fn(done)
                xb, yb = x_tr[idx], y_tr[idx]
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                apply_masks(model)
                # 비용: 이 스텝의 실제 활성 연결 기준
                cum_flops += 3.0 * idx.numel() * 2.0 * linear_active_weights(model)
                samples_seen += idx.numel()
                step += 1
                done += 1
                if sched is not None and phase == sched_phase and sched.step(step):
                    prune_log.append({"step": step, "density": sched.log[-1][1], "removed": sched.log[-1][2],
                                      "active": linear_active_weights(model)})
                if dynamic is not None and phase == "main" and step % every == 0 and step <= dyn_end:
                    frac = dyn_frac0 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, dyn_end)))
                    dynamic_sparse_step(model, frac, dynamic)
                if step % eval_every == 0:
                    acc = record(step, {"phase": phase})
                    if step % (eval_every * 10) == 0:
                        log(f"  [{arm} d={density} s{seed}] step {step}/{total_steps} acc {acc:.4f} "
                            f"active {linear_active_weights(model):,} loss {loss.item():.3f}")
                model.train()

    # ---- 본 학습 ----
    train_steps(total_steps, lambda s: cosine_lr(s, total_steps, warmup, lr), "main")

    # ---- 학습 후 가지치기 (ttp) ----
    if ttp:
        acc_before = record(step, {"phase": "pre_prune"})
        ft_steps = int(float(cfg.get("ft_epochs", epochs / 2)) * steps_per_epoch)
        if kind == "ttp_gradual":
            # 학습 후 점진 가지치기: 미세조정 구간의 10~70% 에 걸쳐 cubic 스케줄로 1 -> d (총 스텝은 ttp 와 같음)
            begin = step + int(float(cfg.get("prune_begin", 0.1)) * ft_steps)
            end = step + int(float(cfg.get("prune_end", 0.7)) * ft_steps)
            sched = PruningScheduler(model, rule="magnitude", d_final=density, begin_step=begin, end_step=end,
                                     every=every, scope="global")
            sched_phase = "finetune"
        else:
            prune_to_density(model, density, "magnitude", None, scope="global")
        acc_after = record(step, {"phase": "post_prune"})
        prune_log.append({"step": step, "density": mask_density(model), "removed": None,
                          "active": linear_active_weights(model), "acc_before": acc_before, "acc_after": acc_after})
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        train_steps(ft_steps, lambda s: cosine_lr(s, ft_steps, 0, lr), "finetune")

    if tracker is not None:
        tracker.remove()

    # ---- 최종 지표 ----
    final_acc = record(step, {"phase": "final"})
    probs = eval_probs(model, x_te).cpu().numpy()
    cal = calibration_metrics(probs, y_te_np)
    main_curve = [c for c in curve if c.get("phase") in ("main", "final", "finetune", "post_prune")]
    de = data_efficiency([c["samples_seen"] for c in main_curve], [c["test_acc"] for c in main_curve],
                         target_acc=float(cfg.get("target_acc", 0.97)))
    de98 = data_efficiency([c["samples_seen"] for c in main_curve], [c["test_acc"] for c in main_curve], target_acc=0.98)
    active = linear_active_weights(model)
    out = {
        "cfg": cfg, "arm": arm_out, "density": density, "budget": budget, "seed": seed, "epochs_effective": epochs,
        "hidden": model.hidden_dims if hasattr(model, "hidden_dims") else None,
        "final_active": active, "final_density_vs_big": active / n_weights(big_h),
        "final_acc": final_acc, "best_acc": max(c["test_acc"] for c in curve),
        "infer_flops": 2.0 * active, "cum_train_flops": cum_flops, "samples_seen": samples_seen, "steps": step,
        "calibration": cal, "data_efficiency_97": de, "data_efficiency_98": de98,
        "curve": curve, "prune_log": prune_log, "time_s": time.time() - t0,
    }
    d = os.path.join(RESULTS_DIR, f"d{density:g}", arm_out)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log(f"[done] {arm} d={density:g} seed {seed}: acc {final_acc:.4f} (best {out['best_acc']:.4f}) "
        f"active {active:,} infer {out['infer_flops']:.3e} train {cum_flops:.3e} ece {cal['ece']:.3f} "
        f"to97 {de['samples_to_target']} ({out['time_s']:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--density", type=float, default=0.1)
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
