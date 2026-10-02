# -*- coding: utf-8 -*-
"""실험 1 스윕.

  python experiments/exp1_prerouting_attention/run_all.py --dataset mnist --seeds 0 1 2
  python experiments/exp1_prerouting_attention/run_all.py --dataset cifar10 --seeds 0 --variants full post_0.25 pre_0.25
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

from experiments.exp1_prerouting_attention.run import run_one  # noqa: E402

DEFAULT_VARIANTS = ["full", "post_0.5", "post_0.25", "post_0.1", "pre_0.5", "pre_0.25", "pre_0.1"]


def parse_variant(v: str):
    if v == "full":
        return "full", 1.0
    mode, keep = v.split("_")
    return mode, float(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()
    for v in a.variants:
        mode, keep = parse_variant(v)
        for seed in a.seeds:
            out_path = os.path.join(REPO_ROOT, "results", "exp1", a.dataset, f"{mode}_k{keep:g}", f"seed{seed}.json")
            if a.skip_existing and os.path.exists(out_path):
                continue
            cfg = {"dataset": a.dataset, "mode": mode, "keep": keep, "seed": seed}
            if a.epochs:
                cfg["epochs"] = a.epochs
            try:
                run_one(cfg)
            except Exception:
                print(f"[error] exp1 {a.dataset} {v} seed {seed}")
                traceback.print_exc()
            finally:
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
