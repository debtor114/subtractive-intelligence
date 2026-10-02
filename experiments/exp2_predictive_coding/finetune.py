# -*- coding: utf-8 -*-
"""실험 2 보강: 스킵을 켠 채로 ViT 를 미세조정 (학습 시 적응). 사후 적용과 비교.

  python experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_mnist/seed0_*/model_final.pt" \
      --dataset mnist --score predicted --skip 0.5 --epochs 2

절차: 예측기 학습 -> (ViT + 예측기) 를 스킵 비율 s 로 CE + 예측기 MSE 로 미세조정 -> 여러 스킵 비율에서 평가.
대조: score=random 으로 같은 미세조정.
결과: results/exp2/<dataset>/finetune_<score>_s<skip>_seed<seed>.json
"""
from __future__ import annotations

import argparse
import glob
import json
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

from baselines import build_model                                              # noqa: E402
from core.predictive_coding import PCForward, _block_io, analytic_flops, fit_predictors  # noqa: E402
from experiments.exp2_predictive_coding.run import _TensorLoader, eval_forward  # noqa: E402
from utils.data_loader import get_dataset                                      # noqa: E402
from utils.metrics import count_flops                                          # noqa: E402
from utils.seed import set_seed                                                # noqa: E402
from utils.tensor_data import load_mnist_tensors                               # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "exp2")


class TrainablePC(PCForward):
    """PCForward 와 같은 계산이지만 no_grad 없이 (미세조정용)."""

    def __call__(self, x):
        return self.forward(x)


def run_one(cfg: dict, log=print) -> dict:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = torch.device("cuda")
    ckpt = torch.load(cfg["ckpt"], map_location=device, weights_only=False)
    mcfg, ds = ckpt["config"]["model"], cfg["dataset"]
    if ds == "mnist":
        (x_tr, y_tr), (x_te, y_te) = load_mnist_tensors(device=device)
        train_loader, test_loader = _TensorLoader(x_tr, y_tr, 128, True), _TensorLoader(x_te, y_te, 1000, False)
        input_shape = (1, 28, 28)
    else:
        train_loader, test_loader, _ = get_dataset("cifar10", batch_size=128, num_workers=0, augment=True)
        input_shape = (3, 32, 32)
    vit = build_model(mcfg, input_shape, 10).to(device)
    vit.load_state_dict(ckpt["model"])
    depth, dim = mcfg["depth"], mcfg["dim"]
    n_tokens = (input_shape[1] // mcfg["patch_size"]) * (input_shape[2] // mcfg["patch_size"]) + 1
    rank = int(cfg.get("rank", 16))
    score, skip, epochs = cfg["score"], float(cfg["skip"]), int(cfg.get("epochs", 2))

    t0 = time.time()
    vit.eval()
    preds = fit_predictors(vit, train_loader, device, rank=rank, epochs=int(cfg.get("pred_epochs", 2)), log=log)
    base = eval_forward(vit, test_loader, device)
    before = eval_forward(PCForward(vit, preds, [skip] * depth, score=score, substitute="identity"), test_loader, device)
    log(f"  [ft {ds} {score} s={skip}] before finetune: full {base['acc']:.4f} skipped {before['acc']:.4f}")

    fwd = TrainablePC(vit, preds, [skip] * depth, score=score, substitute="identity")
    params = list(vit.parameters()) + list(preds.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-4)), weight_decay=0.05)
    steps = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for ep in range(epochs):
        vit.train()
        preds.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = fwd(x)
            ce = F.cross_entropy(logits, y)
            # 예측기는 계속 잔차를 배운다 (고정 순전파 기준)
            with torch.no_grad():
                vit.eval()
                ins, deltas = _block_io(vit, x)
                vit.train()
            mse = sum(F.mse_loss(p(i.detach()), d.detach()) for p, i, d in zip(preds, ins, deltas))
            loss = ce + float(cfg.get("mse_weight", 1.0)) * mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tl += ce.item() * y.numel()
            tc += int((logits.argmax(1) == y).sum().item())
            tn += y.numel()
        vit.eval()
        preds.eval()
        te = eval_forward(PCForward(vit, preds, [skip] * depth, score=score, substitute="identity"), test_loader, device)
        log(f"  [ft {ds} {score} s={skip}] ep {ep + 1}/{epochs} train {tc / tn:.4f} skipped-test {te['acc']:.4f}")

    vit.eval()
    preds.eval()
    measured = count_flops(vit, input_shape, device).dense
    per_block = analytic_flops(dim, n_tokens, 1, [0.0], rank, 0.0, 0.0)["full"]
    fixed = max(0.0, measured - depth * per_block)
    rows = []
    for s in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for sc in ("predicted", "random"):
            f = PCForward(vit, preds, [s] * depth, score=sc, substitute="identity") if s > 0 else vit
            r = eval_forward(f, test_loader, device)
            fl = analytic_flops(dim, n_tokens, depth, [s] * depth, rank, fixed, 0.0)
            rows.append({"eval_score": sc, "skip_frac": s, **r, "flops_ratio": fl["ratio"]})
            if s == 0:
                break
    out = {"cfg": cfg, "dataset": ds, "seed": seed, "train_score": score, "train_skip": skip, "epochs": epochs,
           "baseline_full": base, "before_finetune_skipped": before, "rows": rows, "time_s": time.time() - t0}
    d = os.path.join(RESULTS_DIR, ds)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"finetune_{score}_s{skip:g}_seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log(f"[done] exp2-finetune {ds} {score} s={skip} seed {seed}: "
        + " ".join(f"{r['eval_score'][:4]}@{r['skip_frac']:.1f}={r['acc']:.4f}" for r in rows) + f" ({out['time_s']:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--score", default="predicted", choices=["predicted", "random"])
    ap.add_argument("--skip", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    ckpts = sorted(glob.glob(a.ckpt))
    if not ckpts:
        raise FileNotFoundError(a.ckpt)
    cfg = {"ckpt": ckpts[-1], "dataset": a.dataset, "score": a.score, "skip": a.skip, "epochs": a.epochs, "seed": a.seed}
    for p in a.set:
        k, v = p.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    run_one(cfg)


if __name__ == "__main__":
    main()
