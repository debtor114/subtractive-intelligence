# -*- coding: utf-8 -*-
"""핵심 실험 CIFAR-10 스윕. 한 프로세스에서 순차 실행, --skip_existing 으로 재개.

  python experiments/core_prune_during_learning/run_all_cifar.py --densities 0.1 0.03 0.01 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.core_prune_during_learning.run_cifar import run_one  # noqa: E402

ARMS = ["dense_small", "pd_mag_global", "rigl", "ttp", "pd_drivenorm_erk", "pd_mag_erk"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densities", nargs="+", type=float, default=[0.1, 0.03, 0.01])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--skip_dense_big", action="store_true")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    a = ap.parse_args()
    arms = a.arms or ARMS
    jobs = []
    if not a.skip_dense_big:
        jobs += [("dense_big", 1.0, s) for s in a.seeds]
    # 시드 0 을 전 밀도/방식 먼저 돌려 결과가 일찍 보이게 하고, 그다음 시드 1, 2
    for seed in a.seeds:
        for d in a.densities:
            for arm in arms:
                jobs.append((arm, d, seed))
    for arm, d, seed in jobs:
        folder = "core_resnet" if a.model == "resnet18" else "core_cifar"
        out_path = os.path.join(REPO_ROOT, "results", folder, f"d{d:g}", arm, f"seed{seed}.json")
        if a.skip_existing and os.path.exists(out_path):
            continue
        try:
            run_one({"arm": arm, "density": d, "seed": seed, "epochs": a.epochs, "model": a.model})
        except Exception:
            print(f"[error] cifar {arm} d={d} seed {seed}")
            traceback.print_exc()
        finally:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
