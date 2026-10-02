# -*- coding: utf-8 -*-
"""실험 4: STDP 만으로 MNIST 학습 (역전파 없음) + 가중치 그래프 궤적 + 역전파 MLP 대조.

  python experiments/exp4_stdp_temporal/run.py --n_e 400 --train 60000 --seed 0

기록 (results/exp4/n<n_e>/seed<seed>.json):
- 학습곡선: 표본 수 대 정확도 (매 eval_every 장마다 라벨 배정 + 테스트 부분집합 평가)
- 최종: 테스트 10k 정확도 (라벨 배정은 학습 이미지 assign_n 장, 가소성 꺼서)
- 가중치 통계 궤적 (가지치기 관찰): 임계 이하 비율, Gini, theta
- 이론 연산량: 이미지당 추론 SOP, 학습 갱신 수 (사건 구동)
- 대조: 같은 폭의 MLP (784-n_e-10) 를 같은 표본 수만 보고 역전파로 학습한 곡선 (Adam)
"""
from __future__ import annotations

import argparse
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

from baselines.mlp import MLP                                                  # noqa: E402
from experiments.exp4_stdp_temporal.snn import (DiehlCookSNN, OpCounts, SNNConfig,  # noqa: E402
                                               assign_labels, predict)
from utils.metrics import data_efficiency                                     # noqa: E402
from utils.seed import set_seed                                               # noqa: E402
from utils.tensor_data import load_mnist_tensors                              # noqa: E402

RESULTS_DIR = os.path.join(REPO_ROOT, "results", "exp4")


@torch.no_grad()
def collect_counts(snn: DiehlCookSNN, x: torch.Tensor, bs: int, ops: OpCounts):
    out = []
    for s in range(0, x.shape[0], bs):
        out.append(snn.run_batch(x[s:s + bs], learn=False, ops=ops))
    return torch.cat(out)


def snn_accuracy(snn, x_assign, y_assign, x_eval, y_eval, bs, ops):
    counts_a = collect_counts(snn, x_assign, bs, ops)
    assign = assign_labels(counts_a, y_assign)
    counts_e = collect_counts(snn, x_eval, bs, ops)
    pred = predict(counts_e, assign)
    return float((pred == y_eval).float().mean().item()), assign, counts_e


def backprop_reference(x_tr, y_tr, x_te, y_te, n_hidden, n_train, eval_points, seed, device, log):
    """같은 표본 수를 한 번 훑는 역전파 MLP (Adam, bs 32). eval_points 표본 시점마다 테스트 정확도."""
    set_seed(seed)
    model = MLP((1, 28, 28), 10, hidden=(n_hidden,)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 32
    curve = []
    seen = 0
    next_eval = 0
    order = torch.randperm(x_tr.shape[0], device=device)[:n_train]
    model.train()
    for s in range(0, n_train, bs):
        idx = order[s:s + bs]
        loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        seen += idx.numel()
        if next_eval < len(eval_points) and seen >= eval_points[next_eval]:
            model.eval()
            with torch.no_grad():
                acc = float((model(x_te).argmax(1) == y_te).float().mean().item())
            model.train()
            curve.append({"samples_seen": seen, "test_acc": acc})
            next_eval += 1
    model.eval()
    with torch.no_grad():
        acc = float((model(x_te).argmax(1) == y_te).float().mean().item())
    n_w = 784 * n_hidden + n_hidden * 10
    return {"final_acc": acc, "curve": curve, "infer_flops": 2.0 * n_w, "train_flops_total": 3.0 * 2.0 * n_w * n_train,
            "params": n_w + n_hidden + 10}


def run_one(cfg: dict, log=print) -> dict:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = torch.device("cuda")
    (x_tr_n, y_tr), (x_te_n, y_te) = load_mnist_tensors(device=device, normalize=False)
    x_tr = x_tr_n.reshape(x_tr_n.shape[0], -1)
    x_te = x_te_n.reshape(x_te_n.shape[0], -1)

    scfg = SNNConfig(n_e=int(cfg.get("n_e", 400)), time=int(cfg.get("time", 250)),
                     intensity=float(cfg.get("intensity", 128.0)), batch_size=int(cfg.get("batch_size", 32)),
                     inh=float(cfg.get("inh", 17.5)), theta_plus=float(cfg.get("theta_plus", 0.05)),
                     nu_pre=float(cfg.get("nu_pre", 1e-4)), nu_post=float(cfg.get("nu_post", 1e-2)),
                     reduction=str(cfg.get("reduction", "sum")))
    snn = DiehlCookSNN(scfg, device, seed=seed)
    n_train = int(cfg.get("train", 60000))
    eval_every = int(cfg.get("eval_every", 5000))
    bs = scfg.batch_size
    eval_every = max(bs, (eval_every // bs) * bs)   # 배치 크기의 배수로 맞춘다 (아니면 평가 시점을 건너뛴다)
    assign_n = int(cfg.get("assign_n", 10000))
    eval_sub = int(cfg.get("eval_sub", 2000))

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    order = torch.randperm(x_tr.shape[0], device=device, generator=g)[:n_train]
    assign_idx = torch.randperm(x_tr.shape[0], device=device, generator=g)[:assign_n]
    eval_idx = torch.randperm(x_te.shape[0], device=device, generator=g)[:eval_sub]
    quick_assign_idx = assign_idx[:2000]

    curve, wstats, spikes_log = [], [], []
    t0 = time.time()
    seen = 0
    wstats.append({"samples_seen": 0, **snn.weight_stats()})
    for s in range(0, n_train, bs):
        idx = order[s:s + bs]
        counts = snn.run_batch(x_tr[idx], learn=True, ops=snn.ops_train)
        seen += idx.numel()
        if s == 0 or (seen // bs) % 50 == 0 or seen % eval_every == 0:
            spikes_log.append({"samples_seen": seen, "mean_spikes_per_image": float(counts.sum(1).mean().item()),
                               "silent_frac": float((counts.sum(1) == 0).float().mean().item())})
        if seen % eval_every == 0 or seen >= n_train:
            acc, _, _ = snn_accuracy(snn, x_tr[quick_assign_idx], y_tr[quick_assign_idx],
                                     x_te[eval_idx], y_te[eval_idx], 128, OpCounts())
            curve.append({"samples_seen": seen, "test_acc": acc})
            wstats.append({"samples_seen": seen, **snn.weight_stats()})
            log(f"  [stdp n_e={scfg.n_e} s{seed}] seen {seen} acc(sub) {acc:.4f} spikes/img "
                f"{spikes_log[-1]['mean_spikes_per_image']:.1f} silent {spikes_log[-1]['silent_frac']:.2f} "
                f"w<1% {wstats[-1]['frac_below_1pct']:.3f} gini {wstats[-1]['gini']:.3f} ({time.time() - t0:.0f}s)")
    train_time = time.time() - t0

    # 최종 평가: 라벨 배정 assign_n 장 (가소성 off), 테스트 10k
    ops_eval = OpCounts()
    final_acc, assign, counts_te = snn_accuracy(snn, x_tr[assign_idx], y_tr[assign_idx], x_te, y_te, 128, ops_eval)
    # 확신도: 클래스 점수의 정규화 (모르겠다 판정 참고용)
    inf_ops = ops_eval.inference_ops_per_image(scfg.n_e)
    learn_ops = snn.ops_train.learning_ops_per_image(scfg.n_input, scfg.n_e)
    train_inf_ops = snn.ops_train.inference_ops_per_image(scfg.n_e)

    eval_points = [c["samples_seen"] for c in curve]
    ref = backprop_reference(x_tr_n, y_tr, x_te_n, y_te, scfg.n_e, n_train, eval_points, seed, device, log)
    de_snn = data_efficiency([c["samples_seen"] for c in curve], [c["test_acc"] for c in curve], target_acc=0.8)
    de_ref = data_efficiency([c["samples_seen"] for c in ref["curve"]], [c["test_acc"] for c in ref["curve"]], target_acc=0.8)

    out = {
        "cfg": cfg, "snn_cfg": scfg.__dict__, "seed": seed, "n_train": n_train,
        "final_acc": final_acc, "curve": curve, "weight_stats": wstats, "spikes": spikes_log,
        "neurons_per_class": torch.bincount(assign, minlength=10).tolist(),
        "ops": {"inference_per_image": inf_ops, "learning_updates_per_image": learn_ops,
                "train_inference_per_image": train_inf_ops,
                "train_total_ops": (train_inf_ops["total"] + learn_ops) * n_train,
                "dense_gpu_flops_per_image_inference": 2.0 * scfg.n_input * scfg.n_e * scfg.time},
        "params": scfg.n_input * scfg.n_e + scfg.n_e,
        "backprop_ref": ref, "data_efficiency_80": {"snn": de_snn, "backprop": de_ref},
        "train_time_s": train_time, "time_s": time.time() - t0,
    }
    d = os.path.join(RESULTS_DIR, f"n{scfg.n_e}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    torch.save({"W": snn.W.cpu(), "theta": snn.theta.cpu(), "assign": assign.cpu()}, os.path.join(d, f"weights_seed{seed}.pt"))
    log(f"[done] exp4 n_e={scfg.n_e} seed {seed}: STDP acc {final_acc:.4f} | backprop MLP acc {ref['final_acc']:.4f} | "
        f"SNN infer SOP/img {inf_ops['total']:.3e} (MLP {ref['infer_flops']:.3e} FLOPs) | to80 snn {de_snn['samples_to_target']} "
        f"bp {de_ref['samples_to_target']} ({out['time_s']:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_e", type=int, default=400)
    ap.add_argument("--train", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = {"n_e": a.n_e, "train": a.train, "seed": a.seed}
    for p in a.set:
        k, v = p.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    run_one(cfg)


if __name__ == "__main__":
    main()
