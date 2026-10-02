# -*- coding: utf-8 -*-
"""실험 1+2 통합 (사전 필터링): 시상 라우터 사전 결정 스킵 + 예측 잔차 대체 + 점진 스킵 미세조정 (+ 뒤쪽 블록 사전 라우팅 어텐션).

  python experiments/exp12_prefilter/run.py --ckpt "results/baseline_vit_mnist/seed0_*/model_final.pt" --dataset mnist \
      --mode thalamic --late_attn full --smax 0.7 --epochs 4 --seed 0

절차: 체크포인트 로드 -> 예측기/라우터 워밍업(고정 ViT, 1 에폭) -> 미세조정 E 에폭 (스킵 비율 0 -> smax 로 60% 구간 선형 상승,
      CE + 예측기 MSE + 라우터 MSE) -> 여러 스킵 비율과 세 선택 기준(thalamic / layerwise / random)으로 평가.
결과: results/exp12/<dataset>/<mode>_<late_attn>_s<smax>_seed<seed>.json
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

from baselines import build_model                                                  # noqa: E402
from core.thalamic_router import collect_aux_loss                                  # noqa: E402
from core.thalamic_skip import ThalamicSkip, analytic_flops, swap_late_attention    # noqa: E402
from experiments.exp2_predictive_coding.run import _TensorLoader                   # noqa: E402
from utils.data_loader import get_dataset                                          # noqa: E402
from utils.metrics import calibration_metrics, count_flops                         # noqa: E402
from utils.seed import set_seed                                                    # noqa: E402
from utils.tensor_data import load_mnist_tensors                                   # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "exp12")


@torch.no_grad()
def evaluate(model: ThalamicSkip, loader, device, skip_fracs, mode):
    model.eval()
    probs, labels, comp = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x, skip_fracs, mode)
        probs.append(torch.softmax(logits.float(), 1).cpu())
        labels.append(y.cpu())
        comp.append(model.last_computed_frac)
    p, t = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    cal = calibration_metrics(p, t)
    return {"acc": cal["acc"], "ece": cal["ece"], "nll": cal["nll"], "computed_frac": float(np.mean(comp))}


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
    n_front = int(cfg.get("n_front", depth // 2))
    n_late = depth - n_front
    late_attn = cfg.get("late_attn", "full")
    if late_attn == "pre":
        swap_late_attention(vit, n_front, keep_ratio=float(cfg.get("attn_keep", 0.25)))
    model = ThalamicSkip(vit, n_front, rank=int(cfg.get("rank", 16)), router_hidden=int(cfg.get("router_hidden", 64))).to(device)
    mode, smax, epochs = cfg["mode"], float(cfg["smax"]), int(cfg.get("epochs", 4))
    substitute = str(cfg.get("substitute", "predicted"))
    model.substitute = substitute
    n_tokens = (input_shape[1] // mcfg["patch_size"]) * (input_shape[2] // mcfg["patch_size"]) + 1
    measured = count_flops(vit, input_shape, device).dense
    fixed = max(0.0, measured - depth * analytic_flops(dim, n_tokens, 1, 0, [0.0], 16, 64, 0.0)["full"])
    t0 = time.time()

    # 0. 기준 (미세조정 전): 스킵 없음
    # 스킵 0 이면 모드와 무관하게 전 블록을 계산하므로, drop_late 도 원본 (뒤쪽 블록 포함) 정확도를 기준으로 기록한다
    base = evaluate(model, test_loader, device, [0.0] * n_late, "thalamic" if mode == "drop_late" else mode)
    log(f"  [exp12 {ds} {mode}/{late_attn} s{smax}] before: full {base['acc']:.4f}")

    # 1. 워밍업: ViT 고정, 예측기 + 라우터만 1 에폭 (drop_late 대조군은 라우터/예측기를 쓰지 않으므로 생략)
    if mode != "drop_late":
        aux_params = list(model.predictors.parameters()) + list(model.router.parameters())
        opt_aux = torch.optim.Adam(aux_params, lr=1e-3)
        model.train()
        for p in vit.parameters():
            p.requires_grad_(False)
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            _, ar, ap = model(x, [0.0] * n_late, mode, with_aux=True)
            loss = ar + ap
            opt_aux.zero_grad(set_to_none=True)
            loss.backward()
            opt_aux.step()
        for p in vit.parameters():
            p.requires_grad_(True)
    warm = evaluate(model, test_loader, device, [smax] * n_late, mode)
    log(f"  [exp12 {ds} {mode}/{late_attn} s{smax}] after warmup (no finetune), skip {smax}: {warm['acc']:.4f}")

    # 2. 점진 스킵 미세조정
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg.get("lr", 1e-4)), weight_decay=0.05)
    steps = epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ramp = int(0.6 * steps)
    step = 0
    hist = []
    for ep in range(epochs):
        model.train()
        tc, tn = 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            s_now = smax * min(1.0, step / max(1, ramp))
            logits, ar, ap = model(x, [s_now] * n_late, mode, with_aux=True)
            loss = F.cross_entropy(logits, y) + ar + ap + (collect_aux_loss(vit).to(logits.dtype) if late_attn == "pre" else 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            tc += int((logits.argmax(1) == y).sum().item())
            tn += y.numel()
        te = evaluate(model, test_loader, device, [smax] * n_late, mode)
        hist.append({"epoch": ep + 1, "train_acc": tc / tn, "test_acc_at_smax": te["acc"], "s_end": s_now})
        log(f"  [exp12 {ds} {mode}/{late_attn} s{smax}] ep {ep + 1}/{epochs} s={s_now:.2f} train {tc / tn:.4f} test@smax {te['acc']:.4f}")

    # 3. 평가: 스킵 비율 x 선택 기준
    rows = []
    if mode == "drop_late":
        # 뒤쪽 블록 제거: 앞쪽 블록만의 비용 (라우터/예측기 없음)
        r = evaluate(model, test_loader, device, [1.0] * n_late, mode)
        full = analytic_flops(dim, n_tokens, depth, n_front, [0.0] * n_late, 16, 64, fixed)["full"]
        front_only = analytic_flops(dim, n_tokens, n_front, n_front, [], 0, 0, fixed)["skip"]
        rows.append({"skip_frac": 1.0, "eval_mode": "drop_late", **r, "flops_ratio": front_only / full, "flops": front_only})
    for s in ([] if mode == "drop_late" else [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]):
        for ev_mode in (["thalamic", "layerwise", "random"] if s > 0 else [mode]):
            r = evaluate(model, test_loader, device, [s] * n_late, ev_mode)
            fl = analytic_flops(dim, n_tokens, depth, n_front, [s] * n_late, model.rank, model.router.hidden, fixed,
                                late_attn=late_attn, attn_keep=float(cfg.get("attn_keep", 0.25)))
            rows.append({"skip_frac": s, "eval_mode": ev_mode, **r, "flops_ratio": fl["ratio"], "flops": fl["skip"]})
    out = {"cfg": cfg, "dataset": ds, "seed": seed, "mode": mode, "late_attn": late_attn, "smax": smax, "epochs": epochs,
           "substitute": substitute,
           "n_front": n_front, "n_tokens": n_tokens, "baseline_full": base, "after_warmup_at_smax": warm,
           "history": hist, "rows": rows, "flops_full": analytic_flops(dim, n_tokens, depth, n_front, [0.0] * n_late, 16, 64, fixed)["full"],
           "time_s": time.time() - t0}
    d = os.path.join(RESULTS_DIR, ds)
    os.makedirs(d, exist_ok=True)
    name = f"{mode}_{late_attn}_s{smax:g}{'_sub-identity' if substitute == 'identity' else ''}_seed{seed}.json"
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    summ = " ".join(f"{r['eval_mode'][:4]}@{r['skip_frac']:.1f}={r['acc']:.4f}" for r in rows if r["eval_mode"] == mode)
    log(f"[done] exp12 {ds} {mode}/{late_attn} s{smax} seed {seed}: full {base['acc']:.4f} | {summ} ({out['time_s']:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--mode", default="thalamic", choices=["thalamic", "layerwise", "random", "drop_late"])
    ap.add_argument("--late_attn", default="full", choices=["full", "pre"])
    ap.add_argument("--smax", type=float, default=0.7)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--substitute", default="predicted", choices=["predicted", "identity"])
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    ckpts = sorted(glob.glob(a.ckpt))
    if not ckpts:
        raise FileNotFoundError(a.ckpt)
    cfg = {"ckpt": ckpts[-1], "dataset": a.dataset, "mode": a.mode, "late_attn": a.late_attn, "smax": a.smax,
           "epochs": a.epochs, "seed": a.seed, "substitute": a.substitute}
    for p in a.set:
        k, v = p.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    run_one(cfg)


if __name__ == "__main__":
    main()
