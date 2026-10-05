# -*- coding: utf-8 -*-
"""논문 2 탐색 (OVERNIGHT_P2.md) 공용 학습 루프.

논문 1 의 MNIST MLP (run.py) / CIFAR-10 CNN (run_cifar.py) 프로토콜을 그대로 쓰되, '언제 어느 층을 어떤 모양으로 깎는가' 를
플러그인(pruner) 으로 바꾼다. 데이터는 GPU 상주, DataLoader 없음, 결과는 results/p2/ 아래 JSON.

condition:
  global          : 논문 1 의 전역 크기 기준 학습 중 가지치기 (기준 런; 층별 최종 밀도를 다른 조건에 넘겨준다)
  sync            : 모든 층이 [0.10, 0.70] 에서 cubic, 층별 최종 밀도는 cfg["targets"] 로 고정
  bottom_up       : 층 l 의 창 = [0.10 + 0.30 l/(L-1), +0.30] (아래층 먼저)
  top_down        : bottom_up 의 반대 순서
  two_waves       : [0.10, 0.35] 에서 sqrt(d_l) 까지, 쉼, [0.45, 0.70] 에서 d_l 까지
  progress_gated  : 학습 정확도 EMA(0.99) 가 기준을 넘는 순간(늦어도 0.40) 시작, 0.70 에 끝
  block16_during  : 16x16 타일 단위 전역 크기 가지치기를 cubic 스케줄로 학습 중에
  block16_oneshot : dense 학습 -> 타일 단위 한 번에 -> 절반 에폭 미세조정 (학습률 그대로 cosine 재시작)
  dense           : 가지치기 없음 (X1 용, kwta 옵션과 함께)
  dense_small     : 예산에 맞춘 작은 dense 망 (X1 용)
옵션: kwta=<k_frac> 은 MLP 은닉 활성 중 상위 k 비율만 남긴다 (활동 희소성, X1).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baselines.cnn import SmallCNN                                                    # noqa: E402
from baselines.mlp import MLP                                                         # noqa: E402
from core.masked_layers import apply_masks, convert_to_masked, masked_modules         # noqa: E402
from core.pruning import PruningScheduler, _keep_topk, cubic_density                  # noqa: E402
from experiments.core_prune_during_learning.run import dense_hidden_for_budget, n_weights  # noqa: E402
from utils.cifar_gpu import augment, load_cifar10_gpu, normalize                      # noqa: E402
from utils.metrics import calibration_metrics, count_flops                            # noqa: E402
from utils.seed import set_seed                                                       # noqa: E402
from utils.tensor_data import TensorBatches, load_mnist_tensors                       # noqa: E402

RES_P2 = os.path.join(REPO_ROOT, "results", "p2")
MASK_DIR = os.path.join(RES_P2, "masks")
PROTO = {
    "mnist": dict(epochs=15, bs=128, lr=1e-3, every=100, eval_every=100, gate_acc=0.97, input_shape=(1, 28, 28)),
    "cnn": dict(epochs=20, bs=128, lr=0.05, wd=5e-4, every=100, eval_every=100, gate_acc=0.80, input_shape=(3, 32, 32)),
}
BIG_MLP_WEIGHTS = n_weights(1024)
_DATA: Dict[str, tuple] = {}


# ---------------------------------------------------------------------------
# 모델, 데이터
# ---------------------------------------------------------------------------

class KWTA(nn.Module):
    """표본마다 활성 상위 k 비율만 남기고 나머지는 0 (활동 희소성). ReLU 뒤에 둔다."""

    def __init__(self, k_frac: float):
        super().__init__()
        self.k_frac = float(k_frac)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k_frac >= 1.0:
            return x
        k = max(1, int(round(self.k_frac * x.shape[1])))
        thr = x.topk(k, dim=1).values[:, -1:]
        return x * (x >= thr).to(x.dtype)


def build_model(model_name: str, condition: str, budget: int, kwta: float = 1.0) -> nn.Module:
    if model_name == "mnist":
        if condition == "dense_small":
            h = dense_hidden_for_budget(budget)
            m = MLP((1, 28, 28), 10, hidden=(h, h))
        else:
            m = MLP((1, 28, 28), 10, hidden=(1024, 1024))
        if kwta < 1.0:
            # net: Linear, ReLU, Linear, ReLU, Linear -> ReLU 뒤에 k-WTA
            layers = []
            for mod in m.net:
                layers.append(mod)
                if isinstance(mod, nn.ReLU):
                    layers.append(KWTA(kwta))
            m.net = nn.Sequential(*layers)
    else:
        m = SmallCNN(channels=(64, 128, 256), fc=256)
    if condition in ("dense", "dense_small"):
        return m
    return convert_to_masked(m)


def get_data(model_name: str, device):
    if model_name not in _DATA:
        if model_name == "mnist":
            (x_tr, y_tr), (x_te, y_te) = load_mnist_tensors(device=device)
            _DATA[model_name] = (x_tr, y_tr, x_te, y_te, None, None)
        else:
            (x_tr, y_tr), (x_te, y_te), (mean, std) = load_cifar10_gpu(device=device)
            _DATA[model_name] = (x_tr, y_tr, x_te, y_te, mean, std)
    return _DATA[model_name]


def layer_densities(model: nn.Module) -> Dict[str, float]:
    return {n: float(m.weight_mask.mean().item()) for n, m in masked_modules(model)}


def active_weights(model: nn.Module) -> int:
    tot = 0
    for m in model.modules():
        if hasattr(m, "weight_mask"):
            tot += int(m.weight_mask.sum().item())
        elif isinstance(m, (nn.Linear, nn.Conv2d)):
            tot += m.weight.numel()
    return tot


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def save_masks(model: nn.Module, tag: str) -> str:
    os.makedirs(MASK_DIR, exist_ok=True)
    path = os.path.join(MASK_DIR, tag + ".pt")
    torch.save({n: m.weight_mask.bool().cpu() for n, m in masked_modules(model)}, path)
    return path


# ---------------------------------------------------------------------------
# 가지치기 플러그인
# ---------------------------------------------------------------------------

@torch.no_grad()
def prune_layer_to(m: nn.Module, density: float) -> int:
    k = int(round(density * m.weight_mask.numel()))
    return _keep_topk(m, m.weight.abs(), k)


class GlobalCubic:
    """논문 1 의 pd_mag_global (기준)."""

    def __init__(self, model, d_final, total_steps, every):
        self.s = PruningScheduler(model, rule="magnitude", d_final=d_final, begin_step=int(0.1 * total_steps),
                                  end_step=int(0.7 * total_steps), every=every, scope="global")

    def step(self, g: int, ema_acc: float) -> bool:
        return self.s.step(g)


class LayerwiseCubic:
    """층별 창 [b_l, e_l] 과 최종 밀도 d_l. 창 안에서 every 스텝마다, 창 끝에서 한 번 더 깎아 정확히 맞춘다."""

    def __init__(self, model, targets: Dict[str, float], windows: Dict[str, tuple], every: int,
                 d_init: Optional[Dict[str, float]] = None):
        self.model, self.targets, self.windows, self.every = model, targets, windows, every
        self.d_init = d_init or {}

    def step(self, g: int, ema_acc: float) -> bool:
        pruned = False
        for n, m in masked_modules(self.model):
            b, e = self.windows[n]
            if g < b or g > e:
                continue
            if (g - b) % self.every != 0 and g != e:
                continue
            tgt = cubic_density(g, b, e, self.d_init.get(n, 1.0), self.targets[n])
            prune_layer_to(m, tgt)
            pruned = True
        if pruned:
            apply_masks(self.model)
        return pruned


class TwoWaves:
    def __init__(self, model, targets, total_steps, every):
        T = total_steps
        names = [n for n, _ in masked_modules(model)]
        mid = {n: math.sqrt(targets[n]) for n in names}
        self.w1 = LayerwiseCubic(model, mid, {n: (int(0.10 * T), int(0.35 * T)) for n in names}, every)
        self.w2 = LayerwiseCubic(model, targets, {n: (int(0.45 * T), int(0.70 * T)) for n in names}, every, d_init=mid)

    def step(self, g, ema_acc):
        return self.w1.step(g, ema_acc) or self.w2.step(g, ema_acc)


class ProgressGated:
    """학습 정확도 EMA 가 gate 를 넘는 순간 (늦어도 0.40 T) 시작, 0.70 T 에 끝."""

    def __init__(self, model, targets, total_steps, every, gate_acc):
        self.model, self.targets, self.T, self.every, self.gate = model, targets, total_steps, every, gate_acc
        self.start: Optional[int] = None
        self.inner: Optional[LayerwiseCubic] = None

    def step(self, g, ema_acc):
        if self.inner is None:
            if ema_acc >= self.gate or g >= int(0.40 * self.T):
                self.start = g
                names = [n for n, _ in masked_modules(self.model)]
                self.inner = LayerwiseCubic(self.model, self.targets, {n: (g, int(0.70 * self.T)) for n in names}, self.every)
            else:
                return False
        return self.inner.step(g, ema_acc)


@torch.no_grad()
def tile_stats(m: nn.Module, B: int):
    """2 차원 보기 (out, in*k*k) 를 B x B 타일로 나눈 (점수 = 살아있는 |w| 평균, 실제 칸 수, 타일 생존 여부)."""
    w2 = (m.weight * m.weight_mask).abs().reshape(m.weight.shape[0], -1)
    R, C = w2.shape
    Rp, Cp = math.ceil(R / B) * B, math.ceil(C / B) * B
    pad = F.pad(w2, (0, Cp - C, 0, Rp - R))
    cnt = F.pad(torch.ones_like(w2), (0, Cp - C, 0, Rp - R)).view(Rp // B, B, Cp // B, B).sum(dim=(1, 3))
    alive = F.pad(m.weight_mask.reshape(R, -1), (0, Cp - C, 0, Rp - R)).view(Rp // B, B, Cp // B, B).sum(dim=(1, 3)) > 0
    score = pad.view(Rp // B, B, Cp // B, B).sum(dim=(1, 3)) / cnt
    return score, cnt, alive, (R, C, Rp, Cp)


@torch.no_grad()
def prune_tiles_global(model: nn.Module, target_density: float, B: int = 16) -> int:
    """모든 층의 B x B 타일을 한 줄로 세워 점수 높은 타일부터 남긴다. 남긴 타일의 실제 칸 수가 목표 예산에 가장 가깝게.
    이미 죽은 타일은 되살리지 않는다. 끊은 연결 수를 돌려준다."""
    mods = masked_modules(model)
    infos = [tile_stats(m, B) for _, m in mods]
    total = sum(m.weight_mask.numel() for _, m in mods)
    budget = target_density * total
    scores = torch.cat([s.flatten() for s, _, _, _ in infos])
    cnts = torch.cat([c.flatten() for _, c, _, _ in infos])
    alive = torch.cat([a.flatten() for _, _, a, _ in infos])
    scores = torch.where(alive, scores, torch.full_like(scores, float("-inf")))
    order = torch.argsort(scores, descending=True)
    cum = torch.cumsum(cnts[order], dim=0)
    n_alive = int(alive.sum().item())
    k = int((cum <= budget).sum().item())
    if k < n_alive and k < order.numel() and abs(float(cum[k]) - budget) < abs(float(cum[max(k - 1, 0)]) - budget):
        k += 1
    k = min(k, n_alive)
    keep_flat = torch.zeros_like(alive)
    keep_flat[order[:k]] = True
    before = sum(int(m.weight_mask.sum().item()) for _, m in mods)
    off = 0
    for (n, m), (s, c, a, (R, C, Rp, Cp)) in zip(mods, infos):
        nt = s.numel()
        keep = keep_flat[off:off + nt].view(Rp // B, Cp // B)
        off += nt
        full = keep.repeat_interleave(B, 0).repeat_interleave(B, 1)[:R, :C]
        new = (m.weight_mask.reshape(R, C).bool() & full).to(m.weight_mask.dtype)
        m.weight_mask.copy_(new.view_as(m.weight_mask))
    apply_masks(model)
    return before - sum(int(m.weight_mask.sum().item()) for _, m in mods)


@torch.no_grad()
def prune_tiles_layerwise(model: nn.Module, per_layer: Dict[str, float], B: int = 16) -> int:
    """층마다 따로: 그 층의 B x B 타일을 점수순으로 남겨 층별 목표 밀도에 가장 가깝게 맞춘다 (E2 와 같은 층별 배분)."""
    removed = 0
    for n, m in masked_modules(model):
        s, c, a, (R, C, Rp, Cp) = tile_stats(m, B)
        budget = per_layer[n] * m.weight_mask.numel()
        sf = torch.where(a, s, torch.full_like(s, float("-inf"))).flatten()
        order = torch.argsort(sf, descending=True)
        cum = torch.cumsum(c.flatten()[order], dim=0)
        n_alive = int(a.sum().item())
        k = int((cum <= budget).sum().item())
        if k < n_alive and k < order.numel() and abs(float(cum[k]) - budget) < abs(float(cum[max(k - 1, 0)]) - budget):
            k += 1
        k = min(max(k, 1), n_alive)
        keep = torch.zeros_like(a).flatten()
        keep[order[:k]] = True
        full = keep.view(Rp // B, Cp // B).repeat_interleave(B, 0).repeat_interleave(B, 1)[:R, :C]
        before = int(m.weight_mask.sum().item())
        m.weight_mask.copy_((m.weight_mask.reshape(R, C).bool() & full).to(m.weight_mask.dtype).view_as(m.weight_mask))
        removed += before - int(m.weight_mask.sum().item())
    apply_masks(model)
    return removed


class BlockCubicLayerwise:
    """층별 목표 (E2 기준 런의 층별 최종 밀도) 를 cubic 스케줄로 따라가며 타일 단위로 깎는다."""

    def __init__(self, model, targets, total_steps, every, B=16):
        self.model, self.targets, self.b, self.e, self.every, self.B = model, targets, int(0.1 * total_steps), int(0.7 * total_steps), every, B

    def step(self, g, ema_acc):
        if g < self.b or g > self.e:
            return False
        if (g - self.b) % self.every != 0 and g != self.e:
            return False
        prune_tiles_layerwise(self.model, {n: cubic_density(g, self.b, self.e, 1.0, d) for n, d in self.targets.items()}, self.B)
        return True


class BlockCubic:
    def __init__(self, model, d_final, total_steps, every, B=16):
        self.model, self.d, self.b, self.e, self.every, self.B = model, d_final, int(0.1 * total_steps), int(0.7 * total_steps), every, B

    def step(self, g, ema_acc):
        if g < self.b or g > self.e:
            return False
        if (g - self.b) % self.every != 0 and g != self.e:
            return False
        prune_tiles_global(self.model, cubic_density(g, self.b, self.e, 1.0, self.d), self.B)
        return True


def layer_windows(names: List[str], order: str, total_steps: int) -> Dict[str, tuple]:
    L = len(names)
    T = total_steps
    out = {}
    for l, n in enumerate(names):
        if order == "sync":
            s = 0.10
        else:
            pos = l / max(1, L - 1)
            if order == "top_down":
                pos = 1.0 - pos
            s = 0.10 + 0.30 * pos
        out[n] = (int(s * T), int((s + 0.30) * T))
    return out


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------

def run(cfg: dict, log=print) -> dict:
    model_name, cond, seed = cfg["model"], cfg["condition"], int(cfg["seed"])
    P = dict(PROTO[model_name])
    epochs = int(cfg.get("epochs", P["epochs"]))
    bs, lr, every, eval_every = P["bs"], P["lr"], P["every"], P["eval_every"]
    density = float(cfg.get("density", 1.0))
    kwta = float(cfg.get("kwta", 1.0))
    set_seed(seed)
    device = torch.device("cuda")
    x_tr, y_tr, x_te, y_te, mean, std = get_data(model_name, device)
    y_te_np = y_te.cpu().numpy()
    is_cnn = model_name == "cnn"

    big_n = BIG_MLP_WEIGHTS if not is_cnn else None
    model = build_model(model_name, cond, int(round(density * BIG_MLP_WEIGHTS)) if not is_cnn else 0, kwta).to(device)
    masked = cond not in ("dense", "dense_small")
    leaf_dense = count_flops(model, P["input_shape"], device).leaf_dense if is_cnn else None
    n_train = x_tr.shape[0]
    steps_per_epoch = math.ceil(n_train / bs)
    total_steps = epochs * steps_per_epoch
    warmup = steps_per_epoch
    names = [n for n, _ in masked_modules(model)] if masked else []
    targets = cfg.get("targets")

    pruner = None
    oneshot = False
    if cond == "global":
        pruner = GlobalCubic(model, density, total_steps, every)
    elif cond in ("sync", "bottom_up", "top_down"):
        pruner = LayerwiseCubic(model, targets, layer_windows(names, cond, total_steps), every)
    elif cond == "two_waves":
        pruner = TwoWaves(model, targets, total_steps, every)
    elif cond == "progress_gated":
        pruner = ProgressGated(model, targets, total_steps, every, P["gate_acc"])
    elif cond == "block16_during":
        pruner = BlockCubic(model, density, total_steps, every, 16)
    elif cond == "block16_during_pl":
        pruner = BlockCubicLayerwise(model, targets, total_steps, every, 16)
    elif cond in ("block16_oneshot", "block16_oneshot_pl"):
        oneshot = True
    elif cond in ("dense", "dense_small"):
        pass
    else:
        raise KeyError(cond)

    def make_opt(base_lr):
        if is_cnn:
            decay, no_decay = [], []
            for n, p in model.named_parameters():
                (no_decay if p.ndim <= 1 else decay).append(p)
            return torch.optim.SGD([{"params": decay, "weight_decay": P["wd"]}, {"params": no_decay, "weight_decay": 0.0}],
                                   lr=base_lr, momentum=0.9, nesterov=True)
        return torch.optim.Adam(model.parameters(), lr=base_lr)

    opt = make_opt(lr)
    scaler = torch.amp.GradScaler("cuda") if is_cnn else None
    state = {"step": 0, "cum_flops": 0.0, "samples": 0, "ema": 0.0}
    curve: List[Dict] = []
    density_traj: List[Dict] = []
    t0 = time.time()

    def eff_flops():
        if is_cnn:
            d = layer_densities(model) if masked else {}
            return float(sum(leaf_dense[n] * d.get(n, 1.0) for n in leaf_dense))
        return 2.0 * active_weights(model)

    @torch.no_grad()
    def eval_probs():
        was = model.training
        model.eval()
        out = []
        for s in range(0, x_te.shape[0], 1000):
            xb = x_te[s:s + 1000]
            if is_cnn:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(normalize(xb, mean, std))
            else:
                logits = model(xb)
            out.append(torch.softmax(logits.float(), 1))
        model.train(was)
        return torch.cat(out)

    def record(phase):
        p = eval_probs()
        acc = float((p.argmax(1).cpu().numpy() == y_te_np).mean())
        curve.append({"step": state["step"], "samples_seen": state["samples"], "test_acc": acc, "active": active_weights(model),
                      "cum_train_flops": state["cum_flops"], "phase": phase})
        if masked:
            density_traj.append({"step": state["step"], **layer_densities(model)})
        return acc

    def train_steps(n_steps, lr_fn, phase):
        done = 0
        model.train()
        while done < n_steps:
            for idx in TensorBatches(torch.arange(n_train, device=device), bs):
                if done >= n_steps:
                    break
                for g in opt.param_groups:
                    g["lr"] = lr_fn(done)
                xb, yb = x_tr[idx], y_tr[idx]
                if is_cnn:
                    xb = normalize(augment(xb), mean, std)
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = model(xb)
                        loss = F.cross_entropy(logits, yb)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                if masked:
                    apply_masks(model)
                batch_acc = float((logits.argmax(1) == yb).float().mean().item())
                state["ema"] = 0.99 * state["ema"] + 0.01 * batch_acc if state["step"] > 0 else batch_acc
                state["cum_flops"] += 3.0 * idx.numel() * eff_flops()
                state["samples"] += idx.numel()
                state["step"] += 1
                done += 1
                g = state["step"]
                if pruner is not None and phase == "main":
                    pruner.step(g, state["ema"])
                if g % eval_every == 0:
                    acc = record(phase)
                    if g % (eval_every * 10) == 0:
                        log(f"  [{model_name} {cond} d={density:g} s{seed}] step {g}/{total_steps} acc {acc:.4f} "
                            f"active {active_weights(model):,} ema {state['ema']:.3f} ({time.time() - t0:.0f}s)")
                    model.train()

    train_steps(total_steps, lambda s: cosine_lr(s, total_steps, warmup, lr), "main")
    prune_info = {}
    if oneshot:
        acc_before = record("pre_prune")
        removed = prune_tiles_layerwise(model, targets, 16) if cond == "block16_oneshot_pl" else prune_tiles_global(model, density, 16)
        acc_after = record("post_prune")
        prune_info = {"acc_before": acc_before, "acc_after": acc_after, "removed": removed}
        ft_steps = int((epochs / 2) * steps_per_epoch)
        opt = make_opt(lr)
        train_steps(ft_steps, lambda s: cosine_lr(s, ft_steps, 0, lr), "finetune")

    final_acc = record("final")
    probs = eval_probs().cpu().numpy()
    cal = calibration_metrics(probs, y_te_np)
    out = {
        "cfg": cfg, "model": model_name, "condition": cond, "density": density, "seed": seed, "kwta": kwta,
        "final_acc": final_acc, "best_acc": max(c["test_acc"] for c in curve), "ece": cal["ece"],
        "final_active": active_weights(model), "layer_densities": layer_densities(model) if masked else None,
        "infer_flops": eff_flops(), "cum_train_flops": state["cum_flops"], "steps": state["step"], "epochs": epochs,
        "prune_start_step": getattr(pruner, "start", None), "total_steps": total_steps, "prune_info": prune_info,
        "curve": curve, "density_traj": density_traj, "time_s": time.time() - t0,
    }
    if cfg.get("mask_tag") and masked:
        out["mask_path"] = save_masks(model, cfg["mask_tag"])
    if cfg.get("out"):
        os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
        with open(cfg["out"], "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
    log(f"[done] p2 {model_name} {cond} d={density:g} s{seed}: acc {final_acc:.4f} active {out['final_active']:,} "
        f"train {state['cum_flops']:.3e} ({out['time_s']:.0f}s)")
    return out
