# -*- coding: utf-8 -*-
"""X2 대조군: 빠른/느린 가중치의 느린 성분 학습률 (2e-4) 과 같은 학습률로 보통 미세조정 (fast 성분 없음).
fast/slow 의 이득이 '학습률이 낮아서' 인지 '빠른 성분이 있어서' 인지 가른다. 결과는 results/p2/x2/<dataset>/finetune_lr2e4/."""
from __future__ import annotations

import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.exp3_dual_learning import run as exp3run   # noqa: E402

RES_P2 = os.path.join(REPO_ROOT, "results", "p2")
exp3run.RESULTS_DIR = os.path.join(RES_P2, "x2")


def main():
    for dataset in ("split_mnist", "permuted_mnist"):
        for seed in (0, 1, 2):
            cfg = {"dataset": dataset, "method": "finetune", "seed": seed, "n_tasks": 5 if dataset == "split_mnist" else 10,
                   "epochs": 3, "tag": "lr2e4", "lr": 2e-4}
            out = os.path.join(exp3run.RESULTS_DIR, dataset, "finetune_lr2e4", f"seed{seed}.json")
            if os.path.exists(out):
                continue
            t = time.time()
            try:
                exp3run.run_one(cfg, log=lambda s: print(s, flush=True))
            except Exception as e:  # noqa: BLE001
                print(f"[error] x2 control {cfg}: {type(e).__name__}: {e}", flush=True)
            print(f"  ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
