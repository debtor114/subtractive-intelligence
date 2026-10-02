# -*- coding: utf-8 -*-
"""환경 / 파이프라인 연기 테스트.

1. torch + CUDA 확인
2. MLP, ViT 순전파/역전파 + FLOPs 집계 (난수 입력)
3. 연속학습 로더 (Split / Permuted MNIST) 모양 확인
4. MLP MNIST 를 1 에폭 x 50 스텝만 실제로 학습 (전 경로 통과 확인)
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from baselines import build_model  # noqa: E402
from baselines.train import run  # noqa: E402
from utils.data_loader import get_continual_dataset  # noqa: E402
from utils.metrics import count_active_params, count_flops, count_params  # noqa: E402


def check_model(name: str, cfg: dict, input_shape, device) -> None:
    model = build_model(dict(name=name, **cfg), input_shape, 10).to(device)
    x = torch.randn(4, *input_shape, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    rep = count_flops(model, input_shape, device)
    print(f"  {name:4s} params={count_params(model):,} active={count_active_params(model):,} "
          f"flops dense={rep.dense:,.0f} effective={rep.effective:,.0f} leaf={rep.leaf_total:,.0f} "
          f"loss={loss.item():.3f}")
    assert rep.leaf_total <= rep.dense + 1, "leaf flops exceed dense flops"
    assert abs(rep.effective - rep.dense) < 1e-6, "no pruning yet, effective must equal dense"


def main() -> None:
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}")

    print("[1] models")
    check_model("mlp", dict(hidden=[256, 256]), (1, 28, 28), device)
    check_model("vit", dict(patch_size=4, dim=128, depth=4, num_heads=4), (1, 28, 28), device)
    check_model("vit", dict(patch_size=4, dim=192, depth=6, num_heads=3), (3, 32, 32), device)

    print("[2] continual loaders")
    tasks, info = get_continual_dataset("split_mnist", n_tasks=5, batch_size=64)
    x, y = next(iter(tasks[0].train_loader))
    print(f"  split_mnist tasks={len(tasks)} classes[0]={tasks[0].classes} batch={tuple(x.shape)} "
          f"labels={sorted(set(y.tolist()))} num_classes={info.num_classes}")
    tasks, info = get_continual_dataset("permuted_mnist", n_tasks=3, batch_size=64)
    x0, _ = next(iter(tasks[0].test_loader))
    x1, _ = next(iter(tasks[1].test_loader))
    print(f"  permuted_mnist tasks={len(tasks)} batch={tuple(x1.shape)} "
          f"task0_vs_task1_same={bool(torch.equal(x0, x1))}")

    print("[3] short real training run (mlp / mnist, 50 steps)")
    cfg = {
        "run_name": "smoke_mlp_mnist", "seed": 0,
        "dataset": {"name": "mnist", "batch_size": 128, "num_workers": 0},
        "model": {"name": "mlp", "hidden": [256, 256]},
        "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
        "sched": {"name": "cosine", "warmup_epochs": 0},
        "train": {"epochs": 1, "amp": True, "grad_clip": 1.0, "max_steps_per_epoch": 50},
    }
    res = run(cfg)
    assert res["final_test_acc"] > 0.5, "50 steps of MLP on MNIST should already exceed 50%"
    print("smoke test passed")


if __name__ == "__main__":
    main()
