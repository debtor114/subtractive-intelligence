# -*- coding: utf-8 -*-
"""실험 2: 예측 오차 기반 토큰 스킵. 학습된 ViT 체크포인트 위에서 사후 적용한다.

  python experiments/exp2_predictive_coding/run.py --ckpt results/baseline_vit_mnist/seed0_*/model_final.pt --dataset mnist
  python experiments/exp2_predictive_coding/run.py --ckpt results/baseline_vit_cifar10/seed0_*/model_final.pt --dataset cifar10

절차: 예측기 학습(고정 ViT) -> 스킵 비율 s 를 0~0.7 로 바꾸며 정확도 측정
      (점수: predicted / oracle / random, 대체: identity / predicted) -> 해석적 FLOPs.
결과: results/exp2/<dataset>/seed<seed>.json
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
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baselines import build_model                                              # noqa: E402
from core.predictive_coding import PCForward, analytic_flops, fit_predictors   # noqa: E402
from utils.data_loader import get_dataset                                      # noqa: E402
from utils.metrics import calibration_metrics, count_flops                     # noqa: E402
from utils.seed import set_seed                                                # noqa: E402
from utils.tensor_data import TensorBatches, load_mnist_tensors                # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "exp2")


class _TensorLoader:
    def __init__(self, x, y, bs, shuffle):
        self.x, self.y, self.bs, self.shuffle = x, y, bs, shuffle

    def __len__(self):
        return (self.x.shape[0] + self.bs - 1) // self.bs

    def __iter__(self):
        for idx in TensorBatches(torch.arange(self.x.shape[0], device=self.x.device), self.bs, self.shuffle):
            yield self.x[idx], self.y[idx]


@torch.no_grad()
def eval_forward(fwd, loader, device):
    probs, labels, comp = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = fwd(x)
        probs.append(torch.softmax(logits.float(), 1).cpu())
        labels.append(y.cpu())
        if getattr(fwd, "last_computed_frac", None) is not None:
            comp.append(fwd.last_computed_frac)
    p, t = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    cal = calibration_metrics(p, t)
    return {"acc": cal["acc"], "ece": cal["ece"], "nll": cal["nll"], "auroc": cal["auroc"],
            "computed_frac": float(np.mean(comp)) if comp else 1.0}


def run_one(cfg: dict, log=print) -> dict:
    set_seed(int(cfg.get("seed", 0)))
    device = torch.device("cuda")
    ckpt = torch.load(cfg["ckpt"], map_location=device, weights_only=False)
    mcfg, ds = ckpt["config"]["model"], cfg["dataset"]
    if ds == "mnist":
        (x_tr, y_tr), (x_te, y_te) = load_mnist_tensors(device=device)
        train_loader, test_loader = _TensorLoader(x_tr, y_tr, 256, True), _TensorLoader(x_te, y_te, 1000, False)
        input_shape = (1, 28, 28)
    else:
        train_loader, test_loader, info = get_dataset("cifar10", batch_size=256, num_workers=0, augment=False)
        input_shape = (3, 32, 32)
    vit = build_model(mcfg, input_shape, 10).to(device)
    vit.load_state_dict(ckpt["model"])
    vit.eval()
    depth, dim = mcfg["depth"], mcfg["dim"]
    n_tokens = (input_shape[1] // mcfg["patch_size"]) * (input_shape[2] // mcfg["patch_size"]) + 1
    rank = int(cfg.get("rank", 16))

    t0 = time.time()
    base = eval_forward(vit, test_loader, device)
    log(f"  [exp2 {ds}] baseline acc {base['acc']:.4f} ece {base['ece']:.3f}")
    preds = fit_predictors(vit, train_loader, device, rank=rank, epochs=int(cfg.get("pred_epochs", 3)), log=log)

    # 해석적 FLOPs 의 고정 부분 (패치 임베딩 + 헤드) 는 측정치에서 역산
    measured = count_flops(vit, input_shape, device).dense
    per_block = analytic_flops(dim, n_tokens, 1, [0.0], rank, 0.0, 0.0)["full"]
    fixed = max(0.0, measured - depth * per_block)

    rows = []
    fracs = [float(f) for f in cfg.get("fracs", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])]
    for score in ("predicted", "random", "oracle"):
        for sub in (("identity", "predicted") if score == "predicted" else ("identity",)):
            for s in fracs:
                skip = [s] * depth
                fwd = PCForward(vit, preds, skip, score=score, substitute=sub)
                r = eval_forward(fwd, test_loader, device)
                fl = analytic_flops(dim, n_tokens, depth, skip, rank, fixed, 0.0)
                rows.append({"score": score, "substitute": sub, "skip_frac": s, **r,
                             "flops_ratio": fl["ratio"], "flops_skip": fl["skip"], "flops_full": fl["full"]})
                log(f"  [exp2 {ds}] {score:9s} {sub:9s} skip {s:.1f}: acc {r['acc']:.4f} ece {r['ece']:.3f} "
                    f"flops x{fl['ratio']:.3f}")
    # 층별 차등: 뒤쪽 블록만 스킵 (예측 오차가 작아지는 쪽) 참고 변형
    for s in (0.3, 0.5, 0.7):
        skip = [0.0] * (depth // 2) + [s] * (depth - depth // 2)
        fwd = PCForward(vit, preds, skip, score="predicted", substitute="identity")
        r = eval_forward(fwd, test_loader, device)
        fl = analytic_flops(dim, n_tokens, depth, skip, rank, fixed, 0.0)
        rows.append({"score": "predicted", "substitute": "identity_late_only", "skip_frac": s, **r,
                     "flops_ratio": fl["ratio"], "flops_skip": fl["skip"], "flops_full": fl["full"]})

    out = {"cfg": cfg, "dataset": ds, "seed": int(cfg.get("seed", 0)), "ckpt": cfg["ckpt"], "model": mcfg,
           "n_tokens": n_tokens, "baseline": base, "predictor_rel_mse": getattr(preds, "rel_mse", None),
           "measured_flops": measured, "rows": rows, "time_s": time.time() - t0}
    d = os.path.join(RESULTS_DIR, ds)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"seed{out['seed']}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log(f"[done] exp2 {ds} seed {out['seed']} ({out['time_s']:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    ckpts = sorted(glob.glob(a.ckpt))
    if not ckpts:
        raise FileNotFoundError(a.ckpt)
    cfg = {"ckpt": ckpts[-1], "dataset": a.dataset, "seed": a.seed}
    for p in a.set:
        k, v = p.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    run_one(cfg)


if __name__ == "__main__":
    main()
