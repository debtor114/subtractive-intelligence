# -*- coding: utf-8 -*-
"""실험 1: 억제 기반 희소 어텐션 (사후 마스킹) 대 시상 사전 라우팅.

  python experiments/exp1_prerouting_attention/run.py --dataset mnist --mode pre --keep 0.25 --seed 0
  python experiments/exp1_prerouting_attention/run.py --dataset cifar10 --mode post --keep 0.25 --epochs 30

MNIST 는 GPU 텐서 경로(증강 없음), CIFAR-10 은 DataLoader(크롭+플립) 를 쓴다.
결과: results/exp1/<dataset>/<mode>_k<keep>/seed<seed>.json
  정확도, 총 FLOPs(dense/측정), 어텐션 FLOPs(해석적: dense / 희소성 활용), 보정, 학습곡선, 시간
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baselines.train import build_scheduler, evaluate                      # noqa: E402
from baselines.vit import ViT                                              # noqa: E402
from core.thalamic_router import analytic_attention_flops, collect_aux_loss, swap_attention  # noqa: E402
from utils.data_loader import get_cifar10                                  # noqa: E402
from utils.metrics import calibration_metrics, count_flops, count_params, data_efficiency  # noqa: E402
from utils.seed import set_seed                                            # noqa: E402
from utils.tensor_data import TensorBatches, load_mnist_tensors            # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "exp1")

PRESETS = {
    "mnist": dict(input_shape=(1, 28, 28), patch_size=4, dim=128, depth=4, num_heads=4, dropout=0.1, emb_dropout=0.1,
                  epochs=10, lr=1e-3, wd=0.05, warmup=1, label_smoothing=0.0, batch_size=128),
    "cifar10": dict(input_shape=(3, 32, 32), patch_size=4, dim=192, depth=6, num_heads=3, dropout=0.1, emb_dropout=0.1,
                    epochs=30, lr=1e-3, wd=0.05, warmup=3, label_smoothing=0.1, batch_size=128),
}


class _TensorLoader:
    """(x, y) 텐서를 DataLoader 처럼 순회."""

    def __init__(self, x, y, bs, shuffle):
        self.x, self.y, self.bs, self.shuffle = x, y, bs, shuffle
        self.batch_size = bs

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.bs)

    def __iter__(self):
        for idx in TensorBatches(torch.arange(self.x.shape[0], device=self.x.device), self.bs, self.shuffle):
            yield self.x[idx], self.y[idx]


def run_one(cfg: dict, log=print) -> dict:
    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device("cuda")
    ds = cfg["dataset"]
    P = dict(PRESETS[ds])
    P.update({k: v for k, v in cfg.items() if k in P})
    epochs = int(P["epochs"])
    bs = int(P["batch_size"])

    if ds == "mnist":
        (x_tr, y_tr), (x_te, y_te) = load_mnist_tensors(device=device)
        train_loader = _TensorLoader(x_tr, y_tr, bs, True)
        test_loader = _TensorLoader(x_te, y_te, 1000, False)
    else:
        train_loader, test_loader, _ = get_cifar10(batch_size=bs, num_workers=0, augment=True)

    model = ViT(P["input_shape"], 10, patch_size=P["patch_size"], dim=P["dim"], depth=P["depth"],
                num_heads=P["num_heads"], dropout=P["dropout"], emb_dropout=P["emb_dropout"]).to(device)
    mode, keep = cfg["mode"], float(cfg.get("keep", 1.0))
    router_dim = int(cfg.get("router_dim", 8))
    swap_attention(model, mode=mode, keep_ratio=keep, router_dim=router_dim, aux_weight=float(cfg.get("aux", 1.0)))

    n_tokens = (P["input_shape"][1] // P["patch_size"]) * (P["input_shape"][2] // P["patch_size"]) + 1
    head_dim = P["dim"] // P["num_heads"]
    attn_flops = analytic_attention_flops(n_tokens, head_dim, P["num_heads"], mode, keep, router_dim)
    attn_flops_full = analytic_attention_flops(n_tokens, head_dim, P["num_heads"], "full")
    flops = count_flops(model, P["input_shape"], device)

    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=P["wd"])
    sched = build_scheduler(opt, {"name": "cosine", "warmup_epochs": P["warmup"], "min_lr_ratio": 0.01},
                            len(train_loader), epochs)
    scaler = torch.amp.GradScaler("cuda")
    amp = True
    curve, history = [], []
    seen = 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tl, tc, tn, ta = 0.0, 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=P["label_smoothing"])
                aux = collect_aux_loss(model).to(loss.dtype)
                total = loss + aux
            opt.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * y.numel()
            ta += float(aux.item()) * y.numel()
            tc += int((logits.argmax(1) == y).sum().item())
            tn += y.numel()
        seen += tn
        te = evaluate(model, test_loader, device, amp=amp)
        rec = {"epoch": ep, "samples_seen": seen, "train_loss": tl / tn, "train_acc": tc / tn, "aux": ta / tn,
               "test_acc": te["acc"], "test_loss": te["loss"], "ece": te["ece"], "time_s": time.time() - t0}
        history.append(rec)
        curve.append({"samples_seen": seen, "test_acc": te["acc"]})
        log(f"  [{ds} {mode} k={keep} s{seed}] ep {ep}/{epochs} train {tc / tn:.4f} aux {ta / tn:.4f} "
            f"test {te['acc']:.4f} ece {te['ece']:.3f} ({time.time() - t0:.0f}s)")

    cal = evaluate(model, test_loader, device, amp=amp)
    de = data_efficiency([c["samples_seen"] for c in curve], [c["test_acc"] for c in curve])
    out = {
        "cfg": cfg, "dataset": ds, "mode": mode, "keep": keep, "router_dim": router_dim, "seed": seed,
        "n_tokens": n_tokens, "k": attn_flops["k"], "params": count_params(model),
        "flops_total_measured": flops.dense,
        "attn_flops_per_block_dense": attn_flops["dense"], "attn_flops_per_block_exploited": attn_flops["exploited"],
        "attn_flops_full_per_block": attn_flops_full["dense"], "depth": P["depth"],
        "final_acc": history[-1]["test_acc"], "best_acc": max(h["test_acc"] for h in history),
        "calibration": cal, "data_efficiency": de, "history": history, "time_s": time.time() - t0,
    }
    # 총 FLOPs (해석적 보정): 측정치는 full 기준 어텐션을 포함하므로, 어텐션 부분을 모드별 해석값으로 치환
    out["flops_total_exploited"] = flops.dense - P["depth"] * attn_flops_full["dense"] + P["depth"] * attn_flops["exploited"]
    d = os.path.join(RESULTS_DIR, ds, f"{mode}_k{keep:g}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log(f"[done] exp1 {ds} {mode} k={keep} seed {seed}: acc {out['final_acc']:.4f} best {out['best_acc']:.4f} "
        f"ece {cal['ece']:.3f} attn/blk {attn_flops['exploited']:.3e} (full {attn_flops_full['dense']:.3e}) "
        f"total {out['flops_total_exploited']:.3e} (full-measured {flops.dense:.3e}) ({out['time_s']:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--mode", default="full", choices=["full", "post", "pre"])
    ap.add_argument("--keep", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = {"dataset": a.dataset, "mode": a.mode, "keep": a.keep, "seed": a.seed}
    if a.epochs:
        cfg["epochs"] = a.epochs
    for p in a.set:
        k, v = p.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    run_one(cfg)


if __name__ == "__main__":
    main()
