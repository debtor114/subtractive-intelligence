# -*- coding: utf-8 -*-
"""실험 A: 사전 학습(ImageNet) ResNet-18 에서 출발해 CIFAR-10 에 적응하면서 깎기 ('공짜 대리석').

  python experiments/core_prune_during_learning/run_pretrained.py --arm pt_pd --density 0.02 --seed 0

arm:
  pt_dense      : 사전 학습 가중치, 가지치기 없이 미세조정 (상한)
  pt_pd         : 사전 학습 가중치 + 학습 중 전역 크기 가지치기 (cubic, 10%~70% 구간)
  pt_pd_erk     : 같은 것, ERK 층별 배분 (추론 FLOPs 를 dense small 수준으로 묶음)
  pt_oneshot    : 사전 학습 가중치를 즉시 한 번에 전역 크기 가지치기 -> 미세조정 (학습 후 가지치기의 전이판)
  pt_rigl       : 사전 학습 가중치 위에 ERK 무작위 마스크 -> RigL (끊고 gradient 로 잇기)
  scratch_pd    : 무작위 초기화 + 학습 중 전역 크기 가지치기 (같은 해상도/에폭, 비용 대조)
  scratch_small : 무작위 초기화 dense small (예산에 맞춘 폭)

입력: CIFAR 32x32 를 GPU 에서 128x128 로 올려 넣는다 (사전 학습 스템 7x7/2 + maxpool 그대로). ImageNet 정규화.
학습: SGD nesterov, 사전 학습 lr 0.01 / 무작위 초기화 lr 0.1, cosine, wd 5e-4, 배치 128, AMP, 10 에폭.
비용: 적응 단계 FLOPs 만 센다. 사전 학습 비용은 '이미 지불됨' 으로 보고한다.
결과: results/core_pretrained/d<density>/<arm>/seed<seed>.json
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
from torchvision.models import ResNet18_Weights, resnet18 as tv_resnet18

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.masked_layers import MASKED_TYPES, apply_masks, convert_to_masked, mask_density, masked_modules  # noqa: E402
from core.pruning import PruningScheduler, dynamic_sparse_step, erk_densities, prune_to_density, random_sparse_init  # noqa: E402
from utils.cifar_gpu import augment, load_cifar10_gpu                              # noqa: E402
from utils.metrics import calibration_metrics, count_flops, data_efficiency        # noqa: E402
from utils.seed import set_seed                                                    # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "core_pretrained")
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def build_tv_resnet18(pretrained: bool, width: float = 1.0) -> nn.Module:
    """torchvision ResNet-18. width < 1 이면 같은 구조의 작은 dense 망 (사전 학습 불가)."""
    if width == 1.0:
        m = tv_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        assert not pretrained
        m = tv_resnet18(weights=None, width_per_group=64)
        # 폭 축소: 모든 conv/bn/fc 를 다시 만든다 (torchvision 은 폭 인자가 없어 직접 축소)
        m = _shrink_resnet(m, width)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m


def _shrink_resnet(m: nn.Module, width: float) -> nn.Module:
    """torchvision resnet18 의 채널을 width 배율로 줄인 새 모델."""
    import torchvision
    base = [64, 128, 256, 512]
    w = [max(1, int(round(c * width))) for c in base]

    class Small(torchvision.models.resnet.ResNet):
        def __init__(self):
            super().__init__(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=1000)
            self.inplanes = w[0]
            self.conv1 = nn.Conv2d(3, w[0], 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(w[0])
            self.layer1 = self._make_layer(torchvision.models.resnet.BasicBlock, w[0], 2)
            self.layer2 = self._make_layer(torchvision.models.resnet.BasicBlock, w[1], 2, stride=2)
            self.layer3 = self._make_layer(torchvision.models.resnet.BasicBlock, w[2], 2, stride=2)
            self.layer4 = self._make_layer(torchvision.models.resnet.BasicBlock, w[3], 2, stride=2)
            self.fc = nn.Linear(w[3], 1000)
    return Small()


def prunable_weights(model: nn.Module) -> int:
    """conv 가중치만 센다. 분류층(fc, 5,120 개) 은 가지치기에서 제외한다: 새로 붙인 층이라 초기 가중치가 작아
    전역 크기 기준에 통째로 잘려 망이 죽는다 (pt_oneshot 2% 가 10% 에 고정됐던 원인). 표준 관행대로 dense 로 둔다."""
    return sum(m.weight.numel() for n, m in model.named_modules() if isinstance(m, nn.Conv2d))


def width_for_budget(budget: int) -> float:
    lo, hi = 0.02, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if prunable_weights(build_tv_resnet18(False, mid)) < budget:
            lo = mid
        else:
            hi = mid
    return hi


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
    """conv 활성 가중치 수 (fc 제외, prunable_weights 와 같은 기준)."""
    tot = 0
    for m in model.modules():
        if isinstance(m, MASKED_TYPES) and isinstance(m, nn.Conv2d):
            tot += int(m.weight_mask.sum().item())
        elif isinstance(m, nn.Conv2d):
            tot += m.weight.numel()
    return tot


def prep(x_u8: torch.Tensor, mean, std, res: int) -> torch.Tensor:
    x = x_u8.float().div_(255.0)
    x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    return (x - mean) / std


@torch.no_grad()
def eval_probs(model, x_u8, mean, std, res, bs=500):
    was = model.training
    model.eval()
    out = []
    for s in range(0, x_u8.shape[0], bs):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(prep(x_u8[s:s + bs], mean, std, res))
        out.append(torch.softmax(logits.float(), 1))
    model.train(was)
    return torch.cat(out)


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def run_one(cfg: dict, log=print) -> dict:
    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device("cuda")
    (x_tr, y_tr), (x_te, y_te), _ = load_cifar10_gpu(device=device)
    mean = torch.tensor(IMNET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMNET_STD, device=device).view(1, 3, 1, 1)
    y_te_np = y_te.cpu().numpy()
    arm = cfg["arm"]
    density = float(cfg.get("density", 1.0))
    res = int(cfg.get("res", 128))
    epochs = int(cfg.get("epochs", 10))
    bs = int(cfg.get("batch_size", 128))
    every = int(cfg.get("update_every", 100))
    eval_every = int(cfg.get("eval_every", 100))
    big_n = prunable_weights(build_tv_resnet18(False, 1.0))
    budget = int(round(density * big_n))
    n_train = x_tr.shape[0]
    steps_per_epoch = math.ceil(n_train / bs)
    total_steps = epochs * steps_per_epoch
    pretrained = arm.startswith("pt_")
    lr = float(cfg.get("lr", 0.01 if pretrained else 0.1))

    sched, dynamic = None, None
    if arm == "scratch_small":
        model = build_tv_resnet18(False, width_for_budget(budget)).to(device)
    elif arm == "pt_dense":
        model = build_tv_resnet18(True, 1.0).to(device)
    else:
        model = convert_to_masked(build_tv_resnet18(pretrained, 1.0), skip=("fc",)).to(device)
        if arm == "pt_oneshot":
            prune_to_density(model, density, "magnitude", None, scope="global")
        elif arm == "pt_rigl":
            random_sparse_init(model, density, per_layer=erk_densities(model, density))
            dynamic = "gradient"
            for _, m in masked_modules(model):
                m.dense_grad = True
        elif arm in ("pt_pd", "scratch_pd", "pt_pd_erk"):
            begin = int(float(cfg.get("prune_begin", 0.1)) * total_steps)
            end = int(float(cfg.get("prune_end", 0.7)) * total_steps)
            sched = PruningScheduler(model, rule="magnitude", d_final=density, begin_step=begin, end_step=end,
                                     every=every, scope="erk" if arm == "pt_pd_erk" else "global")
        else:
            raise KeyError(arm)

    leaf_dense = count_flops(model, (3, res, res), device).leaf_dense
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": 5e-4}, {"params": no_decay, "weight_decay": 0.0}],
                          lr=lr, momentum=0.9, nesterov=True)
    scaler = torch.amp.GradScaler("cuda")
    dyn_end = int(0.75 * total_steps)
    curve: List[Dict] = []
    prune_log: List[Dict] = []
    state = {"step": 0, "cum_flops": 0.0, "samples": 0}
    t0 = time.time()

    def eff_flops():
        d = layer_densities(model)
        return float(sum(leaf_dense[n] * d.get(n, 1.0) for n in leaf_dense))

    def record(extra=None):
        p = eval_probs(model, x_te, mean, std, res)
        acc = float((p.argmax(1).cpu().numpy() == y_te_np).mean())
        rec = {"step": state["step"], "samples_seen": state["samples"], "test_acc": acc,
               "active": active_weights(model), "cum_train_flops": state["cum_flops"]}
        if extra:
            rec.update(extra)
        curve.append(rec)
        return acc

    acc0 = record({"phase": "init"})
    log(f"  [{arm} d={density:g} s{seed}] init acc {acc0:.4f} active {active_weights(model):,}")
    model.train()
    warmup = steps_per_epoch if not pretrained else steps_per_epoch // 2
    done = 0
    while done < total_steps:
        perm = torch.randperm(n_train, device=device)
        for s in range(0, n_train, bs):
            if done >= total_steps:
                break
            idx = perm[s:s + bs]
            for g in opt.param_groups:
                g["lr"] = cosine_lr(done, total_steps, warmup, lr)
            xb = prep(augment(x_tr[idx]), mean, std, res)
            yb = y_tr[idx]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            apply_masks(model)
            state["cum_flops"] += 3.0 * idx.numel() * eff_flops()
            state["samples"] += idx.numel()
            state["step"] += 1
            done += 1
            st = state["step"]
            if sched is not None and sched.step(st):
                prune_log.append({"step": st, "density": sched.log[-1][1], "active": active_weights(model)})
            if dynamic is not None and st % every == 0 and st <= dyn_end:
                frac = 0.3 * 0.5 * (1.0 + math.cos(math.pi * st / max(1, dyn_end)))
                dynamic_sparse_step(model, frac, dynamic)
            if st % eval_every == 0:
                acc = record({"phase": "main"})
                if st % (eval_every * 10) == 0:
                    log(f"  [{arm} d={density:g} s{seed}] step {st}/{total_steps} acc {acc:.4f} active {active_weights(model):,} "
                        f"loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
                model.train()

    final_acc = record({"phase": "final"})
    probs = eval_probs(model, x_te, mean, std, res).cpu().numpy()
    cal = calibration_metrics(probs, y_te_np)
    main = [c for c in curve if c.get("phase") in ("main", "final")]
    xs, ys = [c["samples_seen"] for c in main], [c["test_acc"] for c in main]
    out = {
        "cfg": cfg, "arm": arm, "density": density, "budget": budget, "seed": seed, "res": res, "pretrained": pretrained,
        "init_acc": acc0, "final_active": active_weights(model), "final_acc": final_acc, "best_acc": max(ys),
        "infer_flops": eff_flops(), "adapt_train_flops": state["cum_flops"], "samples_seen": state["samples"],
        "calibration": cal, "data_efficiency_90": data_efficiency(xs, ys, 0.90), "curve": curve, "prune_log": prune_log,
        "layer_densities": layer_densities(model), "time_s": time.time() - t0,
    }
    d = os.path.join(RESULTS_DIR, f"d{density:g}", arm)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log(f"[done] pretrained {arm} d={density:g} seed {seed}: acc {final_acc:.4f} (best {out['best_acc']:.4f}) active {out['final_active']:,} "
        f"infer {out['infer_flops']:.3e} adapt-train {state['cum_flops']:.3e} ({out['time_s']:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--density", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = {"arm": a.arm, "density": a.density, "seed": a.seed, "epochs": a.epochs}
    for p in a.set:
        k, v = p.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    run_one(cfg)


if __name__ == "__main__":
    main()
