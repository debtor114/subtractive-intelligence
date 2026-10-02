# -*- coding: utf-8 -*-
"""기준선 학습 CLI.

  python scripts/train_baseline.py --config configs/mlp_mnist.yaml
  python scripts/train_baseline.py --config configs/vit_cifar10.yaml --set train.epochs=5 seed=1

--set 은 점 표기 경로로 YAML 값을 덮어쓴다. 값은 YAML 로 파싱한다 (5, 0.1, true, "[1, 2]" 등).
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baselines.train import run  # noqa: E402


def apply_overrides(cfg: dict, pairs) -> dict:
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"override must look like a.b=value, got '{pair}'")
        key, raw = pair.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=5 seed=1")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args.set)
    run(cfg)


if __name__ == "__main__":
    main()
