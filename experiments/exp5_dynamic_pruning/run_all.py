# -*- coding: utf-8 -*-
"""실험 5 스윕. 한 프로세스에서 순차 실행, --skip_existing 으로 재개.

  python experiments/exp5_dynamic_pruning/run_all.py --densities 0.05 0.005 0.2 --seeds 0 --arms dyn_local dyn_random kwta_in --skip_existing
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.exp5_dynamic_pruning.run import ARMS, run_one   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--densities", nargs="+", type=float, default=[0.05, 0.005, 0.2])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    extra = {}
    for p in a.set:
        k, v = p.split("=", 1)
        import yaml
        extra[k] = yaml.safe_load(v)
    t0 = time.time()
    for d in a.densities:
        for arm in a.arms:
            for seed in a.seeds:
                out_path = os.path.join(REPO_ROOT, "results", "exp5_dynamic", f"d{d:g}", arm, f"seed{seed}.json")
                if a.skip_existing and os.path.exists(out_path):
                    print(f"[skip] exists {out_path}")
                    continue
                try:
                    run_one({"arm": arm, "density": d, "seed": seed, "epochs": a.epochs, **extra})
                except Exception as e:  # 한 런의 실패가 스윕을 막지 않게
                    print(f"[error] {arm} d={d} seed={seed}: {e}")
                    traceback.print_exc()
                torch.cuda.empty_cache()
    print(f"EXP5_SWEEP_DONE ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
