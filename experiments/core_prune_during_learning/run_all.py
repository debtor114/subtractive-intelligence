# -*- coding: utf-8 -*-
"""핵심 실험 스윕: 밀도 x 방식 x 시드.

  python experiments/core_prune_during_learning/run_all.py --densities 0.2 0.1 0.05 0.02 0.01 --seeds 0 1 2
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

from experiments.core_prune_during_learning.run import run_one  # noqa: E402

ARMS = ["dense_small", "static_sparse", "set", "rigl",
        "pd_mag_layer", "pd_mag_global", "pd_act_layer", "pd_actmag_layer", "pd_random_layer", "ttp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densities", nargs="+", type=float, default=[0.2, 0.1, 0.05, 0.02, 0.01])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--skip_dense_big", action="store_true")
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()
    arms = a.arms or ARMS
    if not a.skip_dense_big:
        for seed in a.seeds:
            try:
                run_one({"arm": "dense_big", "density": 1.0, "seed": seed, "epochs": a.epochs})
            except Exception:
                traceback.print_exc()
    for d in a.densities:
        for arm in arms:
            for seed in a.seeds:
                out_path = os.path.join(REPO_ROOT, "results", "core", f"d{d:g}", arm, f"seed{seed}.json")
                if a.skip_existing and os.path.exists(out_path):
                    continue
                try:
                    run_one({"arm": arm, "density": d, "seed": seed, "epochs": a.epochs})
                except Exception:
                    print(f"[error] {arm} d={d} seed {seed}")
                    traceback.print_exc()
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
