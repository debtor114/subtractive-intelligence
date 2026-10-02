# -*- coding: utf-8 -*-
"""실험 3 전체 스윕. 방법 x 변형 x 시드 x 데이터셋을 한 프로세스에서 돈다.

  python experiments/exp3_dual_learning/run_all.py --datasets split_mnist permuted_mnist --seeds 0 1 2
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

from experiments.exp3_dual_learning.run import run_one  # noqa: E402

# (method, tag, overrides). 버퍼 500 은 ER / CLS 공통. 수면 500 스텝.
VARIANTS = [
    ("finetune", "", {}),
    ("joint", "", {}),
    ("er", "", {"buffer": 500, "replay_bs": 128}),
    ("er", "x2ep", {"buffer": 500, "replay_bs": 128, "epochs": 6}),            # 연산량 대략 맞춘 ER
    ("cls", "plain", {"buffer": 500, "sleep_steps": 500, "kd_alpha": 1.0}),
    ("cls", "nokd", {"buffer": 500, "sleep_steps": 500, "kd_alpha": 0.0}),    # fast 의 증류 기여 분리
    ("cls", "down", {"buffer": 500, "sleep_steps": 500, "downscale": 0.1}),
    ("cls", "prune", {"buffer": 500, "sleep_steps": 500, "prune_frac": 0.05}),
    ("cls", "pruneact", {"buffer": 500, "sleep_steps": 500, "prune_frac": 0.05, "prune_rule": "activity"}),
    ("cls", "full", {"buffer": 500, "sleep_steps": 500, "downscale": 0.1, "prune_frac": 0.05}),
    # --- 재설계 (2026-10-01 오후): 균형 리플레이 + 희소 해마 + 증류 없음 + 크기 가지치기만 ---
    ("er_balanced", "", {"buffer": 500}),
    ("cls2", "dense", {"buffer": 500, "sleep_steps": 500, "k_frac": 0.0}),
    ("cls2", "kwta10", {"buffer": 500, "sleep_steps": 500, "k_frac": 0.10}),
    ("cls2", "kwta5", {"buffer": 500, "sleep_steps": 500, "k_frac": 0.05}),
    ("cls2", "kwta10_prune", {"buffer": 500, "sleep_steps": 500, "k_frac": 0.10, "prune_frac": 0.05}),
    ("cls2", "kwta10_lr", {"buffer": 500, "sleep_steps": 300, "k_frac": 0.10, "lr_slow": 1e-3}),   # 수면 과적합 완화 탐색
    # --- 논문 2 토대 (2026-10-01 저녁): 수면 감쇠 스케줄 / 분산 증류 ---
    ("cls2", "decay_boundary", {"buffer": 500, "sleep_steps": 500, "k_frac": 0.10, "decay_mode": "boundary", "downscale": 0.1}),
    ("cls2", "decay_periodic", {"buffer": 500, "sleep_steps": 500, "k_frac": 0.10, "decay_mode": "periodic", "downscale": 0.1, "decay_every": 50}),
    ("cls2", "decay_periodic_small", {"buffer": 500, "sleep_steps": 500, "k_frac": 0.10, "decay_mode": "periodic", "downscale": 0.03, "decay_every": 50}),
    ("cls2", "decay_continuous", {"buffer": 500, "sleep_steps": 500, "k_frac": 0.10, "decay_mode": "continuous", "sleep_wd": 1e-4}),
    ("cls2", "kd_global", {"buffer": 500, "sleep_steps": 500, "k_frac": 0.10, "hidden_fast": 256, "kd_mode": "global", "kd_alpha": 1.0}),
    ("cls2", "kd_local", {"buffer": 500, "sleep_steps": 500, "k_frac": 0.10, "hidden_fast": 256, "kd_mode": "local", "kd_alpha": 1.0}),
    ("cls2", "kd_local_logits", {"buffer": 500, "sleep_steps": 500, "k_frac": 0.10, "hidden_fast": 256, "kd_mode": "local_logits", "kd_alpha": 1.0}),
    ("cls2", "kd_none_h256", {"buffer": 500, "sleep_steps": 500, "k_frac": 0.10, "hidden_fast": 256}),   # 같은 폭의 무증류 기준
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["split_mnist", "permuted_mnist"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None, help="tag filter, e.g. cls_full er")
    ap.add_argument("--skip_existing", action="store_true")
    a = ap.parse_args()
    for ds in a.datasets:
        n_tasks = 5 if ds == "split_mnist" else 10
        for method, tag, over in VARIANTS:
            name = method + (f"_{tag}" if tag else "")
            if a.only and name not in a.only:
                continue
            for seed in a.seeds:
                cfg = {"dataset": ds, "method": method, "seed": seed, "n_tasks": n_tasks,
                       "epochs": a.epochs, "tag": tag}
                cfg.update(over)
                out_path = os.path.join(REPO_ROOT, "results", "exp3", ds, name, f"seed{seed}.json")
                if a.skip_existing and os.path.exists(out_path):
                    continue
                try:
                    run_one(cfg)
                except Exception:
                    print(f"[error] {ds}/{name} seed {seed}")
                    traceback.print_exc()
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()   # WDDM 에서는 GPU 캐시가 커밋 메모리로 잡힌다. 실행 사이에 비운다.


if __name__ == "__main__":
    main()
