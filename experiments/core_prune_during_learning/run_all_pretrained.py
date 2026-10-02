# -*- coding: utf-8 -*-
"""실험 A 스윕 (사전 학습 ResNet-18 -> CIFAR-10 적응하며 깎기). 한 프로세스, --skip_existing 으로 재개.

  python experiments/core_prune_during_learning/run_all_pretrained.py --densities 0.05 0.02 0.005 --seeds 0 1
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

from experiments.core_prune_during_learning.run_pretrained import run_one  # noqa: E402

ARMS = ["pt_pd", "pt_pd_erk", "pt_oneshot", "pt_rigl", "scratch_pd", "scratch_small"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densities", nargs="+", type=float, default=[0.05, 0.02, 0.005])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()
    arms = a.arms or ARMS
    jobs = [("pt_dense", 1.0, s) for s in a.seeds]
    for seed in a.seeds:
        for d in a.densities:
            for arm in arms:
                jobs.append((arm, d, seed))
    for arm, d, seed in jobs:
        out_path = os.path.join(REPO_ROOT, "results", "core_pretrained", f"d{d:g}", arm, f"seed{seed}.json")
        if a.skip_existing and os.path.exists(out_path):
            continue
        try:
            run_one({"arm": arm, "density": d, "seed": seed, "epochs": a.epochs})
        except Exception:
            print(f"[error] pretrained {arm} d={d} seed {seed}")
            traceback.print_exc()
        finally:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
