# -*- coding: utf-8 -*-
"""핵심 실험 CIFAR-10 확장: 소형 CNN 에서 학습 중 가지치기 대 가산 / 학습 후 가지치기 / RigL.

  python experiments/core_prune_during_learning/run_cifar.py --arm pd_mag_global --density 0.03 --seed 0

arm:
  dense_small    : 예산에 맞춘 작은 dense CNN (width 배율로 채널 축소, 가중치 수 = 예산)
  dense_big      : 과잉 CNN (64-128-256, fc 256, 가중치 2.2M) 가지치기 없음
  pd_mag_global  : 과잉 CNN 을 학습 중 cubic 스케줄로 전역 크기 가지치기
  pd_mag_layer   : 같은 것, 층별
  pd_drive_erk   : 시냅스 구동 규칙 (gradient 없음), 층별 목표는 ERK 배분. CNN 에서는 채널이 죽어 붕괴 (2 에폭 검증)
  pd_drivenorm_erk : 구동을 후뉴런별로 정규화한 항상성 판. 본 스윕에 쓰는 지역 규칙
  pd_mag_erk     : 크기 규칙, ERK 배분 (drive 와 같은 배분으로 규칙만 비교)
  ttp            : dense 학습 -> 전역 크기 한 번에 가지치기 -> 미세조정
  rigl           : ERK 밀도로 희소 시작, 끊고 gradient 로 다시 잇기
  static_sparse  : ERK 밀도 무작위 고정 마스크

학습: SGD(momentum 0.9, nesterov) lr 0.05 cosine, wd 5e-4, 배치 128, AMP, 20 에폭. GPU 증강 (크롭 + 반전).
FLOPs: conv/linear 리프의 dense FLOPs x 층별 밀도의 합을 샘플당 순전파로 보고, 학습은 x3 을 매 스텝 누적.
결과: results/core_cifar/d<density>/<arm>/seed<seed>.json
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

from baselines.cnn import SmallCNN, prunable_weight_count, scaled_config, width_for_budget   # noqa: E402
from baselines.resnet import prunable_weights as resnet_weights, resnet18, width_for_budget as resnet_width  # noqa: E402
from core.masked_layers import (MASKED_TYPES, apply_masks, convert_to_masked,            # noqa: E402
                                mask_density, masked_modules)
from core.pruning import (ActivityTracker, PruningScheduler, dynamic_sparse_step,         # noqa: E402
                          erk_densities, prune_to_density, random_sparse_init)
from utils.cifar_gpu import augment, load_cifar10_gpu, normalize                          # noqa: E402
from utils.metrics import calibration_metrics, count_flops, data_efficiency              # noqa: E402
from utils.seed import set_seed                                                          # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "core_cifar")
RESULTS_DIR_RESNET = os.path.join(REPO_ROOT, "results", "core_resnet")
BIG_CH, BIG_FC = (64, 128, 256), 256


@torch.no_grad()
def layer_densities(model: nn.Module) -> Dict[str, float]:
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, MASKED_TYPES):
            out[name] = float(m.weight_mask.mean().item())
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            out[name] = 1.0
    return out


@torch.no_grad()
def active_weights(model: nn.Module) -> int:
    tot = 0
    for m in model.modules():
        if isinstance(m, MASKED_TYPES):
            tot += int(m.weight_mask.sum().item())
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            tot += m.weight.numel()
    return tot


def effective_fwd_flops(leaf_dense: Dict[str, float], dens: Dict[str, float]) -> float:
    return float(sum(leaf_dense[n] * dens.get(n, 1.0) for n in leaf_dense))


@torch.no_grad()
def eval_probs(model: nn.Module, x_u8: torch.Tensor, mean, std, bs: int = 1000) -> torch.Tensor:
    was = model.training
    model.eval()
    out = []
    for s in range(0, x_u8.shape[0], bs):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(normalize(x_u8[s:s + bs], mean, std))
        out.append(torch.softmax(logits.float(), 1))
    model.train(was)
    return torch.cat(out)


def cosine_lr(step: int, total: int, warmup: int, base: float, min_ratio: float = 0.0) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p))))


def run_one(cfg: dict, log=print) -> dict:
    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device("cuda")
    (x_tr, y_tr), (x_te, y_te), (mean, std) = load_cifar10_gpu(device=device)
    y_te_np = y_te.cpu().numpy()

    arm: str = cfg["arm"]
    model_name = str(cfg.get("model", "cnn"))
    density = float(cfg.get("density", 1.0))
    big_n = resnet_weights(resnet18(1.0)) if model_name == "resnet18" else prunable_weight_count(BIG_CH, BIG_FC)
    budget = int(round(density * big_n))
    epochs = int(cfg.get("epochs", 20))
    bs = int(cfg.get("batch_size", 128))
    lr = float(cfg.get("lr", 0.1 if model_name == "resnet18" else 0.05))

    def make_big():
        return resnet18(1.0) if model_name == "resnet18" else SmallCNN(channels=BIG_CH, fc=BIG_FC)

    def make_small(b):
        if model_name == "resnet18":
            return resnet18(resnet_width(b))
        w = width_for_budget(b)
        ch, fc = scaled_config(w)
        return SmallCNN(channels=ch, fc=fc)
    wd = float(cfg.get("weight_decay", 5e-4))
    every = int(cfg.get("update_every", 100))
    eval_every = int(cfg.get("eval_every", 100))
    n_train = x_tr.shape[0]
    steps_per_epoch = math.ceil(n_train / bs)
    total_steps = epochs * steps_per_epoch
    warmup = steps_per_epoch

    tracker, sched, dynamic = None, None, None
    ttp = arm == "ttp"
    if arm == "dense_small":
        model = make_small(budget).to(device)
    elif arm == "dense_big":
        model = make_big().to(device)
    else:
        model = convert_to_masked(make_big()).to(device)
        if arm in ("static_sparse", "rigl", "set"):
            random_sparse_init(model, density, per_layer=erk_densities(model, density))
            if arm == "rigl":
                dynamic = "gradient"
                for _, m in masked_modules(model):
                    m.dense_grad = True
            elif arm == "set":
                dynamic = "random"
        elif arm.startswith("pd_"):
            _, rule, scope = arm.split("_")
            rule_name = {"mag": "magnitude", "drive": "drive", "drivenorm": "drive_norm", "act": "activity",
                         "actmag": "activity_mag", "random": "random"}[rule]
            if rule_name in ("drive", "drive_norm", "activity", "activity_mag"):
                tracker = ActivityTracker(model, momentum=float(cfg.get("act_momentum", 0.99)))
            begin = int(float(cfg.get("prune_begin", 0.1)) * total_steps)
            end = int(float(cfg.get("prune_end", 0.7)) * total_steps)
            scope_name = {"global": "global", "layer": "layerwise", "erk": "erk"}[scope]
            sched = PruningScheduler(model, rule=rule_name, d_final=density, begin_step=begin, end_step=end,
                                     every=every, scope=scope_name, tracker=tracker)
        elif ttp:
            pass
        else:
            raise KeyError(f"unknown arm '{arm}'")

    leaf_dense = count_flops(model, (3, 32, 32), device).leaf_dense   # 리프별 dense FLOPs (마스크 무관)
    dyn_end = int(float(cfg.get("dyn_end", 0.75)) * total_steps)
    dyn_frac0 = float(cfg.get("dyn_frac", 0.3))

    def make_opt(base_lr: float):
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            (no_decay if p.ndim <= 1 else decay).append(p)
        return torch.optim.SGD([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                               lr=base_lr, momentum=0.9, nesterov=True)

    opt = make_opt(lr)
    scaler = torch.amp.GradScaler("cuda")
    curve: List[Dict] = []
    prune_log: List[Dict] = []
    state = {"step": 0, "cum_flops": 0.0, "samples": 0}
    t0 = time.time()

    def record(extra=None):
        p = eval_probs(model, x_te, mean, std)
        acc = float((p.argmax(1).cpu().numpy() == y_te_np).mean())
        rec = {"step": state["step"], "samples_seen": state["samples"], "test_acc": acc,
               "active": active_weights(model), "cum_train_flops": state["cum_flops"]}
        if extra:
            rec.update(extra)
        curve.append(rec)
        return acc

    def train_steps(n_steps: int, lr_fn, phase: str):
        done = 0
        model.train()
        while done < n_steps:
            perm = torch.randperm(n_train, device=device)
            for s in range(0, n_train, bs):
                if done >= n_steps:
                    break
                idx = perm[s:s + bs]
                for g in opt.param_groups:
                    g["lr"] = lr_fn(done)
                xb = normalize(augment(x_tr[idx]), mean, std)
                yb = y_tr[idx]
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                apply_masks(model)
                dens = layer_densities(model)
                state["cum_flops"] += 3.0 * idx.numel() * effective_fwd_flops(leaf_dense, dens)
                state["samples"] += idx.numel()
                state["step"] += 1
                done += 1
                st = state["step"]
                if sched is not None and phase == "main" and sched.step(st):
                    prune_log.append({"step": st, "density": sched.log[-1][1], "removed": sched.log[-1][2],
                                      "active": active_weights(model)})
                if dynamic is not None and phase == "main" and st % every == 0 and st <= dyn_end:
                    frac = dyn_frac0 * 0.5 * (1.0 + math.cos(math.pi * st / max(1, dyn_end)))
                    dynamic_sparse_step(model, frac, dynamic)
                if st % eval_every == 0:
                    acc = record({"phase": phase})
                    if st % (eval_every * 10) == 0:
                        log(f"  [{arm} d={density:g} s{seed}] step {st}/{total_steps} acc {acc:.4f} "
                            f"active {active_weights(model):,} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
                    model.train()

    train_steps(total_steps, lambda s: cosine_lr(s, total_steps, warmup, lr), "main")

    if ttp:
        acc_before = record({"phase": "pre_prune"})
        prune_to_density(model, density, "magnitude", None, scope="global")
        acc_after = record({"phase": "post_prune"})
        prune_log.append({"step": state["step"], "density": mask_density(model), "removed": None,
                          "active": active_weights(model), "acc_before": acc_before, "acc_after": acc_after})
        ft_steps = int(float(cfg.get("ft_epochs", epochs / 2)) * steps_per_epoch)
        opt = make_opt(float(cfg.get("ft_lr", 0.01)))
        train_steps(ft_steps, lambda s: cosine_lr(s, ft_steps, 0, float(cfg.get("ft_lr", 0.01))), "finetune")

    if tracker is not None:
        tracker.remove()

    final_acc = record({"phase": "final"})
    probs = eval_probs(model, x_te, mean, std).cpu().numpy()
    cal = calibration_metrics(probs, y_te_np)
    main_curve = [c for c in curve if c.get("phase") in ("main", "finetune", "post_prune", "final")]
    xs = [c["samples_seen"] for c in main_curve]
    ys = [c["test_acc"] for c in main_curve]
    active = active_weights(model)
    dens = layer_densities(model)
    out = {
        "cfg": cfg, "arm": arm, "density": density, "budget": budget, "seed": seed,
        "model": ({"name": "resnet18", "widths": list(model.widths)} if model_name == "resnet18"
                  else {"name": "cnn", "channels": list(model.channels), "fc": model.fc}),
        "final_active": active, "final_density_vs_big": active / big_n, "layer_densities": dens,
        "final_acc": final_acc, "best_acc": max(c["test_acc"] for c in curve),
        "infer_flops": effective_fwd_flops(leaf_dense, dens), "cum_train_flops": state["cum_flops"],
        "samples_seen": state["samples"], "steps": state["step"],
        "calibration": cal, "data_efficiency_80": data_efficiency(xs, ys, 0.80), "data_efficiency_85": data_efficiency(xs, ys, 0.85),
        "curve": curve, "prune_log": prune_log, "time_s": time.time() - t0,
    }
    d = os.path.join(RESULTS_DIR_RESNET if model_name == "resnet18" else RESULTS_DIR, f"d{density:g}", arm)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log(f"[done] cifar/{model_name} {arm} d={density:g} seed {seed}: acc {final_acc:.4f} (best {out['best_acc']:.4f}) active {active:,} "
        f"infer {out['infer_flops']:.3e} train {state['cum_flops']:.3e} ece {cal['ece']:.3f} to85 "
        f"{out['data_efficiency_85']['samples_to_target']} ({out['time_s']:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--density", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = {"arm": a.arm, "density": a.density, "seed": a.seed, "epochs": a.epochs, "model": a.model}
    for p in a.set:
        k, v = p.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    run_one(cfg)


if __name__ == "__main__":
    main()
