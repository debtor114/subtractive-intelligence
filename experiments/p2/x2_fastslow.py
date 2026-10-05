# -*- coding: utf-8 -*-
"""X2: 해마+피질 '구조' 를 원리로 번역 -> 한 가중치 안의 빠른/느린 성분 (노트 11 절, 번역 미시험).

W = W_slow + W_fast. slow 는 작은 학습률 (Adam), fast 는 큰 학습률 (SGD) 로 배우고 매 스텝 (1 - decay) 로 감쇠한다.
예측은 slow + fast 의 합. 망 두 개도, 증류도, 리플레이 버퍼도 없다.
  fastslow        : 감쇠만
  fastslow_merge  : 태스크가 끝날 때 fast 를 slow 에 더하고 fast 를 0 으로 (수면 전사의 최소 번역)
실험 3 의 러너 (Split / Permuted MNIST, MLP 784-256-256-10, 3 에폭) 를 그대로 쓰고, 결과는 results/p2/x2/ 에 둔다.
기준선 (finetune, er) 은 results/exp3/ 의 논문 1 결과를 재사용한다.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.exp3_dual_learning import run as exp3run                      # noqa: E402
from experiments.exp3_dual_learning.methods import ContinualMethod, METHODS      # noqa: E402

RES_P2 = os.path.join(REPO_ROOT, "results", "p2")


class FastSlowLinear(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.slow = nn.Linear(i, o)
        self.fast = nn.Parameter(torch.zeros(o, i))

    def forward(self, x):
        return F.linear(x, self.slow.weight + self.fast, self.slow.bias)


class FastSlowMLP(nn.Module):
    def __init__(self, hidden=(256, 256)):
        super().__init__()
        dims = [784] + list(hidden)
        self.layers = nn.ModuleList([FastSlowLinear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)] + [FastSlowLinear(dims[-1], 10)])

    def forward(self, x):
        h = torch.flatten(x, 1)
        for i, l in enumerate(self.layers):
            h = l(h)
            if i < len(self.layers) - 1:
                h = torch.relu(h)
        return h


class FastSlow(ContinualMethod):
    name = "fastslow"

    def __init__(self, model_fn, device, cfg, input_shape):
        super().__init__(model_fn, device, cfg, input_shape)
        self.model = FastSlowMLP(tuple(cfg.get("hidden", (256, 256)))).to(device)
        slow = [p for n, p in self.model.named_parameters() if "fast" not in n]
        fast = [p for n, p in self.model.named_parameters() if "fast" in n]
        self.opt_slow = torch.optim.Adam(slow, lr=float(cfg.get("lr_slow", 2e-4)))
        self.opt_fast = torch.optim.SGD(fast, lr=float(cfg.get("lr_fast", 1e-2)))
        self.decay = float(cfg.get("decay", 0.02))
        self.merge = bool(cfg.get("merge", False))
        self.fast_params = fast

    def observe(self, x, y, t):
        self.model.train()
        loss = F.cross_entropy(self.model(x), y)
        self.opt_slow.zero_grad(set_to_none=True)
        self.opt_fast.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_slow.step()
        self.opt_fast.step()
        with torch.no_grad():
            for p in self.fast_params:
                p.mul_(1.0 - self.decay)
        self._account(x.shape[0], self.model)
        return float(loss.item())

    def end_task(self, t, task_x, task_y):
        if self.merge:
            with torch.no_grad():
                for l in self.model.layers:
                    l.slow.weight.add_(l.fast)
                    l.fast.zero_()

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        return self.model(x)

    def predictor(self):
        return self.model


METHODS["fastslow"] = FastSlow
exp3run.RESULTS_DIR = os.path.join(RES_P2, "x2")


def main():
    jobs = []
    for dataset in ("split_mnist", "permuted_mnist"):
        for seed in (0, 1, 2):
            jobs.append({"dataset": dataset, "method": "fastslow", "seed": seed, "n_tasks": 5 if dataset == "split_mnist" else 10,
                         "epochs": 3, "tag": "decay", "merge": False})
            jobs.append({"dataset": dataset, "method": "fastslow", "seed": seed, "n_tasks": 5 if dataset == "split_mnist" else 10,
                         "epochs": 3, "tag": "merge", "merge": True})
    for cfg in jobs:
        out = os.path.join(exp3run.RESULTS_DIR, cfg["dataset"], f"fastslow_{cfg['tag']}", f"seed{cfg['seed']}.json")
        if os.path.exists(out):
            continue
        t = time.time()
        try:
            exp3run.run_one(cfg, log=lambda s: print(s, flush=True))
        except Exception as e:  # noqa: BLE001
            print(f"[error] x2 {cfg}: {type(e).__name__}: {e}", flush=True)
        print(f"  ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
