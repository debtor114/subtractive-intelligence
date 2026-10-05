# -*- coding: utf-8 -*-
"""E1-real + 타일 점유율. 학습으로 만든 마스크 (E2 기준/sync, E4 블록16) 와 층별 밀도가 같은 무작위 마스크를 비교한다.

타일 점유율: 층마다 16x16, 32x32 타일 중 완전히 빈 타일 비율, 죽은 행/열 비율, 무작위 이론값 (1-d)^(타일 칸 수).
실제 지연: MNIST MLP 전체 모델 (배치 1/64/1024, fp32/fp16, dense / csr / bsr16(블록 마스크만)), CNN 은 fc 층만, CPU 는 배치 1 스레드 1.
출력: results/p2/tile_occupancy.csv, results/p2/e1_real.csv
"""
from __future__ import annotations

import csv
import glob
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES_P2 = os.path.join(REPO_ROOT, "results", "p2")
MASK_DIR = os.path.join(RES_P2, "masks")
dev = torch.device("cuda")


def bench_cuda(fn, warm=20, rep=100):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.median(times))


def bench_cpu(fn, warm=10, rep=50):
    for _ in range(warm):
        fn()
    times = []
    for _ in range(rep):
        t = time.perf_counter(); fn(); times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def tile_occupancy(mask2d: torch.Tensor, B: int):
    R, C = mask2d.shape
    Rp, Cp = math.ceil(R / B) * B, math.ceil(C / B) * B
    pad = F.pad(mask2d.float(), (0, Cp - C, 0, Rp - R)).view(Rp // B, B, Cp // B, B).sum(dim=(1, 3))
    return float((pad == 0).float().mean().item())


def random_like(masks: dict, gen) -> dict:
    out = {}
    for n, m in masks.items():
        d = m.float().mean().item()
        out[n] = torch.rand(m.shape, generator=gen) < d
    return out


def occupancy_rows(tag: str, masks: dict):
    rows = []
    for n, m in masks.items():
        m2 = m.reshape(m.shape[0], -1)
        d = float(m2.float().mean().item())
        dead_rows = float((m2.sum(1) == 0).float().mean().item())
        dead_cols = float((m2.sum(0) == 0).float().mean().item())
        for B in (16, 32):
            rows.append(dict(mask_set=tag, layer=n, shape=f"{m2.shape[0]}x{m2.shape[1]}", density=d, B=B,
                             empty_tile_frac=tile_occupancy(m2, B), random_theory=(1.0 - d) ** (B * B),
                             dead_rows=dead_rows, dead_cols=dead_cols))
    return rows


def mlp_forward_dense(ws, bs_, x):
    h = x
    for i, (w, b) in enumerate(zip(ws, bs_)):
        h = F.linear(h, w, b)
        if i < len(ws) - 1:
            h = torch.relu(h)
    return h


def mlp_forward_sparse(wsp, bs_, x):
    h = x.t().contiguous()           # (in, N)
    for i, (w, b) in enumerate(zip(wsp, bs_)):
        h = (w @ h) + b[:, None]
        if i < len(wsp) - 1:
            h = torch.relu(h)
    return h.t()


def mlp_latency(tag, masks: dict, is_block: bool, rows):
    names = list(masks.keys())
    gen = torch.Generator(device="cpu").manual_seed(0)
    ws32 = [(torch.randn(masks[n].shape, generator=gen) * masks[n]).to(dev) for n in names]
    bs_ = [torch.randn(masks[n].shape[0], generator=gen).to(dev) for n in names]
    for dtype, dn in ((torch.float32, "fp32"), (torch.float16, "fp16")):
        ws = [w.to(dtype) for w in ws32]
        bb = [b.to(dtype) for b in bs_]
        fmts = {"csr": [w.to_sparse_csr() for w in ws]}
        if is_block:
            try:
                fmts["bsr16"] = [w.to_sparse_bsr((16, 16)) for w in ws]
            except Exception as e:  # noqa: BLE001
                fmts["bsr16"] = e
        for N in (1, 64, 1024):
            x = torch.randn(N, ws[0].shape[1], device=dev).to(dtype)
            d_ms = bench_cuda(lambda: mlp_forward_dense(ws, bb, x))
            rows.append(dict(model="mnist_mlp", mask_set=tag, device="cuda", dtype=dn, batch=N, format="dense", ms=d_ms, speedup=1.0, status="ok", error=""))
            for fn, obj in fmts.items():
                if isinstance(obj, Exception):
                    rows.append(dict(model="mnist_mlp", mask_set=tag, device="cuda", dtype=dn, batch=N, format=fn, ms=None, speedup=None, status="unsupported", error=str(obj)[:160]))
                    continue
                try:
                    ms = bench_cuda(lambda: mlp_forward_sparse(obj, bb, x))
                    rows.append(dict(model="mnist_mlp", mask_set=tag, device="cuda", dtype=dn, batch=N, format=fn, ms=ms, speedup=d_ms / ms, status="ok", error=""))
                except Exception as e:  # noqa: BLE001
                    rows.append(dict(model="mnist_mlp", mask_set=tag, device="cuda", dtype=dn, batch=N, format=fn, ms=None, speedup=None, status="unsupported", error=f"{type(e).__name__}: {str(e)[:160]}"))
    # CPU, 배치 1, 스레드 1, fp32
    torch.set_num_threads(1)
    wc = [w.cpu() for w in ws32]
    bc = [b.cpu() for b in bs_]
    x = torch.randn(1, wc[0].shape[1])
    d_ms = bench_cpu(lambda: mlp_forward_dense(wc, bc, x))
    rows.append(dict(model="mnist_mlp", mask_set=tag, device="cpu1", dtype="fp32", batch=1, format="dense", ms=d_ms, speedup=1.0, status="ok", error=""))
    try:
        wcs = [w.to_sparse_csr() for w in wc]
        ms = bench_cpu(lambda: mlp_forward_sparse(wcs, bc, x))
        rows.append(dict(model="mnist_mlp", mask_set=tag, device="cpu1", dtype="fp32", batch=1, format="csr", ms=ms, speedup=d_ms / ms, status="ok", error=""))
    except Exception as e:  # noqa: BLE001
        rows.append(dict(model="mnist_mlp", mask_set=tag, device="cpu1", dtype="fp32", batch=1, format="csr", ms=None, speedup=None, status="unsupported", error=str(e)[:160]))


def cnn_fc_latency(tag, masks: dict, rows):
    gen = torch.Generator(device="cpu").manual_seed(0)
    for n in ("classifier.1", "classifier.4"):
        if n not in masks:
            continue
        m = masks[n]
        w32 = (torch.randn(m.shape, generator=gen) * m).to(dev)
        for dtype, dn in ((torch.float32, "fp32"), (torch.float16, "fp16")):
            w = w32.to(dtype)
            try:
                wc = w.to_sparse_csr()
            except Exception as e:  # noqa: BLE001
                wc = e
            for N in (1, 64, 1024):
                x = torch.randn(w.shape[1], N, device=dev).to(dtype)
                d_ms = bench_cuda(lambda: torch.mm(w, x))
                rows.append(dict(model=f"cnn_{n}", mask_set=tag, device="cuda", dtype=dn, batch=N, format="dense", ms=d_ms, speedup=1.0, status="ok", error=""))
                if isinstance(wc, Exception):
                    rows.append(dict(model=f"cnn_{n}", mask_set=tag, device="cuda", dtype=dn, batch=N, format="csr", ms=None, speedup=None, status="unsupported", error=str(wc)[:160]))
                else:
                    try:
                        ms = bench_cuda(lambda: wc @ x)
                        rows.append(dict(model=f"cnn_{n}", mask_set=tag, device="cuda", dtype=dn, batch=N, format="csr", ms=ms, speedup=d_ms / ms, status="ok", error=""))
                    except Exception as e:  # noqa: BLE001
                        rows.append(dict(model=f"cnn_{n}", mask_set=tag, device="cuda", dtype=dn, batch=N, format="csr", ms=None, speedup=None, status="unsupported", error=f"{type(e).__name__}: {str(e)[:160]}"))
    rows.append(dict(model="cnn_conv", mask_set=tag, device="cuda", dtype="-", batch="-", format="csr", ms=None, speedup=None, status="unsupported",
                     error="PyTorch has no sparse conv2d kernel; conv layers measured as dense only"))


def main():
    gen = torch.Generator(device="cpu").manual_seed(0)
    sets = []
    for model in ("mnist", "cnn"):
        for d in (("0.01", "0.005") if model == "mnist" else ("0.03", "0.01")):
            learned = "global" if model == "mnist" else "sync"
            for kind, tag in (("learned", f"e2_{model}_d{d}_{learned}_s0"), ("block16", f"e4_{model}_d{d}_block16_during_s0")):
                p = os.path.join(MASK_DIR, tag + ".pt")
                if os.path.exists(p):
                    sets.append((model, d, kind, torch.load(p)))
                else:
                    print(f"[missing] {p}", flush=True)
    occ, lat = [], []
    for model, d, kind, masks in sets:
        occ += occupancy_rows(f"{model}_d{d}_{kind}", masks)
        if kind == "learned":
            occ += occupancy_rows(f"{model}_d{d}_random", random_like(masks, gen))
    for model, d, kind, masks in sets:
        tag = f"{model}_d{d}_{kind}"
        if model == "mnist":
            mlp_latency(tag, masks, kind == "block16", lat)
            if kind == "learned":
                mlp_latency(f"{model}_d{d}_random", random_like(masks, gen), False, lat)
        else:
            cnn_fc_latency(tag, masks, lat)
        print(f"  latency {tag} done", flush=True)
    for name, rows in (("tile_occupancy.csv", occ), ("e1_real.csv", lat)):
        with open(os.path.join(RES_P2, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"[done] e1_real: {len(occ)} occupancy rows, {len(lat)} latency rows", flush=True)


if __name__ == "__main__":
    main()
