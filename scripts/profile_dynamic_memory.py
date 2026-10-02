# -*- coding: utf-8 -*-
"""DynamicConv2d 연결 단위 경로의 메모리/시간 프로파일 (팟에서 26GB 를 먹은 원인 추적용).

  python scripts/profile_dynamic_memory.py --arm dyn_local --density 0.05 --train_bs 128 --eval_bs 250
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from baselines.resnet import resnet18                                           # noqa: E402
from core.dynamic_layers import DynamicConv2d, convert_to_dynamic, set_dynamic_density   # noqa: E402


def gb(x):
    return x / 1024 ** 3


def report(tag):
    torch.cuda.synchronize()
    print(f"{tag}: alloc now {gb(torch.cuda.memory_allocated()):.2f} GB, peak alloc {gb(torch.cuda.max_memory_allocated()):.2f} GB, "
          f"peak reserved {gb(torch.cuda.max_memory_reserved()):.2f} GB")
    torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="dyn_local")
    ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--train_bs", type=int, default=128)
    ap.add_argument("--eval_bs", type=int, default=250)
    ap.add_argument("--layer_hooks", action="store_true", help="층별 피크 메모리도 찍는다")
    a = ap.parse_args()
    dev = torch.device("cuda")
    model = convert_to_dynamic(resnet18(1.0), a.arm).to(dev)
    set_dynamic_density(model, a.density)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(a.train_bs, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (a.train_bs,), device=dev)
    if a.layer_hooks:
        def pre(m, inp):
            torch.cuda.synchronize(); m._m0 = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
        def post(m, inp, out):
            torch.cuda.synchronize()
            print(f"    {m._name}: in {gb(m._m0):.2f} -> peak {gb(torch.cuda.max_memory_allocated()):.2f} GB (transient {gb(torch.cuda.max_memory_allocated() - m._m0):.2f})")
        for n, m in model.named_modules():
            if isinstance(m, DynamicConv2d):
                m._name = n; m.register_forward_pre_hook(pre); m.register_forward_hook(post)
    report("after model build")
    model.train()
    for i in range(3):
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        print(f"  train step {i}: {time.time() - t0:.2f} s")
        if i == 0:
            report("after first train step (B=%d)" % a.train_bs)
    report("after 3 train steps")
    model.eval()
    xe = torch.randn(a.eval_bs, 3, 32, 32, device=dev)
    with torch.no_grad():
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            model(xe)
        torch.cuda.synchronize()
        print(f"  eval forward B={a.eval_bs}: {time.time() - t0:.2f} s")
    report("after eval forward")
    print(f"reserved now {gb(torch.cuda.memory_reserved()):.2f} GB")


if __name__ == "__main__":
    main()
