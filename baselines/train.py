# -*- coding: utf-8 -*-
"""지도학습 공용 학습 루프.

run(cfg) 하나로 데이터셋 / 모델 / 옵티마이저 / 스케줄러를 조립해 학습하고,
results/<run_name>/seed<seed>_<timestamp>/ 아래에
  config.yaml, results.json, model_final.pt, tb/ (TensorBoard)
를 남긴다.

results.json 에 담기는 것 (ADR-001 의 5 지표에 대응):
  - 정확도      : history[*].test_acc, best/final_test_acc
  - 이론 FLOPs  : flops_dense_per_sample, flops_effective_per_sample, train_flops_total (순전파 x3 근사)
  - 데이터 효율 : learning_curve (표본 수 대 정확도), data_efficiency (AULC, 목표 도달 표본 수)
  - 확신도 보정 : history[*].ece/nll, calibration (마지막 시점 ECE/MCE/NLL/Brier/AUROC/AURC/선택 정확도)
  - 망각 저항성 : 연속학습 러너(experiments/exp3_dual_learning) 가 따로 계산

cfg 구조 (configs/*.yaml 참고):
  run_name, seed, deterministic
  dataset: {name, batch_size, num_workers, augment}
  model:   {name, ...생성자 인자}
  optim:   {name: adamw|adam|sgd, lr, weight_decay, momentum}
  sched:   {name: cosine|none, warmup_epochs, min_lr_ratio}
  train:   {epochs, amp, grad_clip, label_smoothing, eval_every, eval_every_steps,
            target_acc, progress, max_steps_per_epoch}
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baselines import build_model                                   # noqa: E402
from utils.data_loader import get_dataset                           # noqa: E402
from utils.metrics import (accuracy, calibration_metrics, count_active_params,  # noqa: E402
                           count_flops, count_params, data_efficiency, training_flops)
from utils.seed import set_seed                                     # noqa: E402

RESULTS_ROOT = os.path.join(REPO_ROOT, "results")


# ---------------------------------------------------------------------------
# 조립
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict, params=None) -> torch.optim.Optimizer:
    name = cfg.get("name", "adamw").lower()
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    params = list(model.parameters()) if params is None else list(params)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=tuple(cfg.get("betas", (0.9, 0.999))))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd,
                               momentum=float(cfg.get("momentum", 0.9)), nesterov=bool(cfg.get("nesterov", True)))
    raise KeyError(f"unknown optimizer '{name}'")


def build_scheduler(opt: torch.optim.Optimizer, cfg: Optional[dict], steps_per_epoch: int, epochs: int):
    """스텝 단위 스케줄러. warmup(선형) 후 cosine 감쇠. name: none 이면 상수."""
    cfg = cfg or {}
    name = cfg.get("name", "cosine").lower()
    if name == "none":
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    warmup = int(round(float(cfg.get("warmup_epochs", 0)) * steps_per_epoch))
    total = max(1, epochs * steps_per_epoch)
    min_ratio = float(cfg.get("min_lr_ratio", 0.0))

    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


# ---------------------------------------------------------------------------
# 학습 / 평가
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, loader, opt, sched, scaler, device, *,
                    amp: bool, grad_clip: float, label_smoothing: float,
                    max_steps: Optional[int] = None, progress: bool = False,
                    step_hook: Optional[Callable[[int, int, nn.Module], None]] = None) -> Dict[str, float]:
    """한 에폭 학습. step_hook(step_idx, samples_in_epoch_so_far, model) 은 매 스텝 뒤 호출된다."""
    model.train()
    total_loss, correct, n, steps = 0.0, 0, 0, 0
    it = loader
    if progress:
        from tqdm import tqdm
        it = tqdm(loader, leave=False, ncols=80)
    for x, y in it:
        if max_steps is not None and steps >= max_steps:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and x.is_cuda)):
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()
        total_loss += loss.item() * y.numel()
        correct += int((logits.argmax(1) == y).sum().item())
        n += y.numel()
        steps += 1
        if step_hook is not None:
            step_hook(steps, n, model)
            model.train()
    return {"loss": total_loss / max(n, 1), "acc": correct / max(n, 1), "steps": steps, "samples": n}


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, *, amp: bool = False,
             with_calibration: bool = True) -> Dict[str, float]:
    """손실/정확도 + (옵션) 확신도 보정 지표. 손실은 라벨 스무딩 없는 NLL."""
    was_training = model.training
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and x.is_cuda)):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        labels.append(y.cpu())
    model.train(was_training)
    p = torch.cat(probs).numpy()
    t = torch.cat(labels).numpy()
    out = {"acc": float((p.argmax(1) == t).mean()),
           "loss": float(-np.log(np.clip(p[np.arange(len(t)), t], 1e-12, 1.0)).mean())}
    if with_calibration:
        cal = calibration_metrics(p, t)
        out.update({k: v for k, v in cal.items() if k != "acc"})
    return out


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def make_run_dir(run_name: str, seed: int, results_root: str = RESULTS_ROOT) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    d = os.path.join(results_root, run_name, f"seed{seed}_{ts}")
    os.makedirs(d, exist_ok=True)
    return d


def run(cfg: dict, log: Callable[[str], None] = print) -> dict:
    seed = int(cfg.get("seed", 0))
    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg.get("train", {})
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    epochs = int(tcfg.get("epochs", 10))

    dcfg = dict(cfg["dataset"])
    ds_name = dcfg.pop("name")
    train_loader, test_loader, info = get_dataset(ds_name, **dcfg)

    model = build_model(cfg["model"], info.input_shape, info.num_classes).to(device)
    opt = build_optimizer(model, cfg.get("optim", {}))
    sched = build_scheduler(opt, cfg.get("sched", {}), len(train_loader), epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    flops = count_flops(model, info.input_shape, device)
    run_name = cfg.get("run_name") or f"{cfg['model']['name']}_{ds_name}"
    run_dir = make_run_dir(run_name, seed)
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(run_dir, "tb"))

    results = {
        "run_name": run_name, "seed": seed, "dataset": ds_name, "model": cfg["model"],
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "amp": amp,
        "params_total": count_params(model), "params_active": count_active_params(model),
        "flops_dense_per_sample": flops.dense, "flops_effective_per_sample": flops.effective,
        "flops_leaf_total_per_sample": flops.leaf_total,
        "epochs": epochs, "history": [], "learning_curve": [], "run_dir": run_dir,
    }
    log(f"[run] {run_name} seed={seed} device={results['device']} amp={amp}")
    log(f"[run] params={results['params_total']:,} active={results['params_active']:,} "
        f"flops/sample dense={flops.dense:,.0f} effective={flops.effective:,.0f}")

    # 데이터 효율 곡선: eval_every_steps 스텝마다 테스트 정확도를 표본 수와 함께 기록
    eval_every_steps = tcfg.get("eval_every_steps")
    state = {"samples_seen": 0, "global_step": 0}

    def step_hook(step_idx: int, samples_in_epoch: int, m: nn.Module) -> None:
        state["global_step"] += 1
        if eval_every_steps and state["global_step"] % int(eval_every_steps) == 0:
            acc = accuracy(m, test_loader, device, amp=amp)
            seen = state["samples_seen"] + samples_in_epoch
            results["learning_curve"].append({"samples_seen": seen, "step": state["global_step"], "test_acc": acc})
            tb.add_scalar("curve/test_acc_vs_samples", acc, seen)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    best_acc = 0.0
    eval_every = int(tcfg.get("eval_every", 1))
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(
            model, train_loader, opt, sched, scaler, device,
            amp=amp, grad_clip=float(tcfg.get("grad_clip", 0.0)),
            label_smoothing=float(tcfg.get("label_smoothing", 0.0)),
            max_steps=tcfg.get("max_steps_per_epoch"), progress=bool(tcfg.get("progress", False)),
            step_hook=step_hook,
        )
        state["samples_seen"] += tr["samples"]
        rec = {"epoch": epoch, "samples_seen": state["samples_seen"], "train_loss": tr["loss"],
               "train_acc": tr["acc"], "lr": opt.param_groups[0]["lr"], "epoch_time_s": time.time() - t0}
        if epoch % eval_every == 0 or epoch == epochs:
            te = evaluate(model, test_loader, device, amp=amp)
            rec.update({"test_loss": te["loss"], "test_acc": te["acc"], "ece": te["ece"], "nll": te["nll"]})
            best_acc = max(best_acc, te["acc"])
            results["learning_curve"].append({"samples_seen": state["samples_seen"], "step": state["global_step"],
                                              "test_acc": te["acc"]})
        results["history"].append(rec)
        for k, v in rec.items():
            if k != "epoch" and v is not None:
                tb.add_scalar(k, v, epoch)
        msg = (f"[ep {epoch:3d}/{epochs}] train loss {tr['loss']:.4f} acc {tr['acc']:.4f}"
               + (f" | test loss {rec['test_loss']:.4f} acc {rec['test_acc']:.4f} ece {rec['ece']:.3f}"
                  if "test_acc" in rec else "")
               + f" | lr {rec['lr']:.2e} | {rec['epoch_time_s']:.1f}s")
        log(msg)

    results["total_train_time_s"] = time.time() - t_start
    results["best_test_acc"] = best_acc
    results["final_test_acc"] = results["history"][-1].get("test_acc")
    results["peak_gpu_mem_mb"] = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
    results["train_flops_total"] = training_flops(flops.effective, state["samples_seen"])
    results["calibration"] = evaluate(model, test_loader, device, amp=amp)
    curve = sorted(results["learning_curve"], key=lambda r: r["samples_seen"])
    results["data_efficiency"] = data_efficiency([r["samples_seen"] for r in curve], [r["test_acc"] for r in curve],
                                                 target_acc=tcfg.get("target_acc"))
    tb.close()

    torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(run_dir, "model_final.pt"))
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    cal = results["calibration"]
    log(f"[done] final acc {results['final_test_acc']:.4f} best {best_acc:.4f} ece {cal['ece']:.3f} "
        f"auroc {cal['auroc'] if cal['auroc'] is None else round(cal['auroc'], 3)} "
        f"time {results['total_train_time_s']:.0f}s peak mem {results['peak_gpu_mem_mb']:.0f}MB -> {run_dir}")
    return results
