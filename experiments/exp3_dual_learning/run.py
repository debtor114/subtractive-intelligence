# -*- coding: utf-8 -*-
"""실험 3 러너: Split MNIST (Class-IL) / Permuted MNIST.

  python experiments/exp3_dual_learning/run.py --dataset split_mnist --method cls --seed 0 \
      --set downscale=0.1 prune_frac=0.05 --tag sleep_full

결과: results/exp3/<dataset>/<method>[_<tag>]/seed<seed>.json
  R          : (T, T) 정확도 행렬. R[i][j] = 태스크 i 까지 학습한 뒤 태스크 j 의 테스트 정확도 (j > i 도 기록)
  continual  : avg_acc / forgetting / bwt / retention
  calibration: 마지막 시점, 전체 테스트셋 (10 클래스) 위 ECE 등
  unknown    : 각 시점 t 에서 '아직 안 배운 클래스' 를 확신도로 가려내는 AUROC (seen 이 양성), 신호별
  cost       : 파라미터, 학습 FLOPs, 추론 FLOPs(dense/effective)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baselines.mlp import MLP                                          # noqa: E402
from experiments.exp3_dual_learning.methods import METHODS             # noqa: E402
from utils.metrics import _auroc, calibration_metrics, continual_metrics  # noqa: E402
from utils.seed import set_seed                                        # noqa: E402
from utils.tensor_data import (TensorBatches, apply_perm, load_mnist_tensors,  # noqa: E402
                               make_permutations, make_split_tasks)

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "exp3")


@torch.no_grad()
def predict_all(method, x: torch.Tensor, perm=None, bs: int = 2000):
    outs = []
    for s in range(0, x.shape[0], bs):
        outs.append(method.predict(apply_perm(x[s:s + bs], perm)).float())
    return torch.cat(outs)


@torch.no_grad()
def confidences_all(method, x: torch.Tensor, perm=None, bs: int = 2000):
    acc = {}
    for s in range(0, x.shape[0], bs):
        c = method.confidences(apply_perm(x[s:s + bs], perm))
        for k, v in c.items():
            acc.setdefault(k, []).append(v.float())
    return {k: torch.cat(v).cpu().numpy() for k, v in acc.items()}


def run_one(cfg: dict, log=print) -> dict:
    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (x_tr, y_tr), (x_te, y_te) = load_mnist_tensors(device=device)
    n_tasks = int(cfg["n_tasks"])
    dataset = cfg["dataset"]

    if dataset == "split_mnist":
        tasks = make_split_tasks(y_tr, y_te, n_tasks)
        perms = [None] * n_tasks
    elif dataset == "permuted_mnist":
        perms = make_permutations(n_tasks, seed=1000 + seed, device=device)
        all_tr = torch.arange(x_tr.shape[0], device=device)
        all_te = torch.arange(x_te.shape[0], device=device)
        tasks = [{"task_id": t, "classes": tuple(range(10)), "train_idx": all_tr, "test_idx": all_te}
                 for t in range(n_tasks)]
    else:
        raise KeyError(dataset)

    hidden = tuple(cfg.get("hidden", (256, 256)))
    model_fn = lambda: MLP((1, 28, 28), 10, hidden=hidden)  # noqa: E731
    method = METHODS[cfg["method"]](model_fn, device, cfg, (1, 28, 28))

    T = n_tasks
    R = np.zeros((T, T))
    R_alt = {}
    unknown = []      # per t: {signal: auroc}
    seen_auroc = []   # per t: {signal: auroc correct-vs-incorrect on seen classes}
    epochs = int(cfg.get("epochs", 3))
    bs = int(cfg.get("batch_size", 128))
    t_start = time.time()
    for t, task in enumerate(tasks):
        method.begin_task(t)
        tr_idx, perm = task["train_idx"], perms[t]
        step = 0
        for ep in range(epochs):
            for idx in TensorBatches(tr_idx, bs):
                xb, yb = apply_perm(x_tr[idx], perm), y_tr[idx]
                method.observe(xb, yb, t)
                step += 1
        method.end_task(t, apply_perm(x_tr[tr_idx], perm), y_tr[tr_idx])

        # 평가: 모든 태스크 (미래 태스크 포함) 에 대해
        for j, tj in enumerate(tasks):
            te_idx = tj["test_idx"]
            logits = predict_all(method, x_te[te_idx], perms[j])
            R[t, j] = float((logits.argmax(1) == y_te[te_idx]).float().mean().item())
            # 대안 예측기 (예: 해마/피질 중 확신도 높은 쪽) 도 같이 기록
            if hasattr(method, "predict_alt"):
                for name, fn in method.predict_alt().items():
                    if name not in R_alt:
                        R_alt[name] = np.zeros((T, T))
                    outs = [fn(apply_perm(x_te[te_idx][s:s + 2000], perms[j])).float()
                            for s in range(0, te_idx.numel(), 2000)]
                    la = torch.cat(outs)
                    R_alt[name][t, j] = float((la.argmax(1) == y_te[te_idx]).float().mean().item())

        # 미지 클래스 탐지 (Split 만 의미 있음): 배운 클래스 vs 아직 안 배운 클래스
        if dataset == "split_mnist" and t < T - 1:
            seen_cls = set()
            for k in range(t + 1):
                seen_cls |= set(tasks[k]["classes"])
            seen_mask = torch.isin(y_te, torch.as_tensor(sorted(seen_cls), device=device)).cpu().numpy()
            confs = confidences_all(method, x_te)
            preds = predict_all(method, x_te).argmax(1).cpu().numpy()
            correct = (preds == y_te.cpu().numpy())
            unknown.append({k: _auroc(v, seen_mask) for k, v in confs.items()})
            seen_auroc.append({k: _auroc(v[seen_mask], correct[seen_mask]) for k, v in confs.items()})
        log(f"[{cfg['method']}{'_' + cfg['tag'] if cfg.get('tag') else ''} s{seed}] task {t}: "
            + " ".join(f"{R[t, j]:.3f}" for j in range(t + 1))
            + f" | avg {R[t, :t + 1].mean():.4f}")

    cm = continual_metrics(R)
    # 마지막 시점 전체 테스트셋 보정 (Permuted 는 태스크 0 순열 = 원본 기준)
    probs = torch.softmax(predict_all(method, x_te, perms[0] if dataset == "permuted_mnist" else None), 1).cpu().numpy()
    cal = calibration_metrics(probs, y_te.cpu().numpy())
    out = {
        "cfg": cfg, "dataset": dataset, "method": cfg["method"], "tag": cfg.get("tag", ""), "seed": seed,
        "R": R.tolist(), "continual": cm, "calibration": cal,
        "R_alt": {k: v.tolist() for k, v in R_alt.items()},
        "continual_alt": {k: continual_metrics(v) for k, v in R_alt.items()},
        "unknown_auroc": unknown, "seen_correct_auroc": seen_auroc,
        "cost": method.summary(), "time_s": time.time() - t_start,
    }
    name = cfg["method"] + (f"_{cfg['tag']}" if cfg.get("tag") else "")
    d = os.path.join(RESULTS_DIR, dataset, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    alt_msg = " ".join(f"{k}:{continual_metrics(v)['avg_acc']:.4f}" for k, v in R_alt.items())
    log(f"[done] {dataset}/{name} seed {seed}: avg_acc {cm['avg_acc']:.4f} ({alt_msg}) forgetting {cm['forgetting']:.4f} "
        f"retention {cm.get('retention', 0):.3f} ece {cal['ece']:.3f} train_flops {out['cost']['train_flops']:.3e} "
        f"({out['time_s']:.0f}s)")
    return out


def parse_set(pairs):
    d = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        d[k] = yaml.safe_load(v)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="split_mnist", choices=["split_mnist", "permuted_mnist"])
    ap.add_argument("--method", default="cls", choices=sorted(METHODS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_tasks", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = {"dataset": a.dataset, "method": a.method, "seed": a.seed, "n_tasks": a.n_tasks,
           "epochs": a.epochs, "tag": a.tag}
    cfg.update(parse_set(a.set))
    run_one(cfg)


if __name__ == "__main__":
    main()
