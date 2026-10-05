# -*- coding: utf-8 -*-
"""E1-micro: GPU 가 0 을 실제로 건너뛰는가. 희소 행렬 곱 (W: size x size, X: size x N) 의 실측 시간.

형식: dense (0 이 든 빽빽한 행렬 torch.mm) / csr / bsr16, bsr32 (블록 마스크만) / 2:4 (fp16, 밀도 0.5, 비구조적만)
측정: CUDA 이벤트, 예열 20 회, 반복 100 회 (한 번에 20 ms 넘으면 20 회), 중앙값. 형식별 저장 바이트도 기록.
출력: results/p2/e1_micro.csv.  지원 안 되는 조합은 status=unsupported 와 오류 메시지.
  python experiments/p2/e1_micro.py            # 전체
  QUICK=1 python experiments/p2/e1_micro.py    # 코드 점검용 축소 격자
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO_ROOT, "results", "p2", "e1_micro.csv")
QUICK = os.environ.get("QUICK") == "1"
SIZES = [1024, 4096] if not QUICK else [1024]
NS = [1, 16, 128, 1024] if not QUICK else [1, 128]
DENS = [1.0, 0.5, 0.1, 0.05, 0.02, 0.01, 0.005] if not QUICK else [0.5, 0.01]
DTYPES = [torch.float32, torch.float16]
dev = torch.device("cuda")


def bench(fn, warm=20, rep=100):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); torch.cuda.synchronize()
    if s.elapsed_time(e) > 20.0:
        rep = 20
    times = []
    for _ in range(rep):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return float(np.median(times))


def make_mask(size, density, shape, gen):
    if density >= 1.0:
        return torch.ones(size, size, device=dev, dtype=torch.bool)
    if shape == "unstructured":
        return torch.rand(size, size, device=dev, generator=gen) < density
    t = torch.rand(size // 16, size // 16, device=dev, generator=gen) < density
    return t.repeat_interleave(16, 0).repeat_interleave(16, 1)


def nbytes(t):
    if t.layout == torch.sparse_csr:
        return sum(a.numel() * a.element_size() for a in (t.values(), t.col_indices(), t.crow_indices()))
    if t.layout == torch.sparse_bsr:
        return sum(a.numel() * a.element_size() for a in (t.values(), t.col_indices(), t.crow_indices()))
    tot = 0
    for a in ("packed", "meta", "packed_t", "meta_t"):
        x = getattr(t, a, None)
        if x is not None:
            tot += x.numel() * x.element_size()
    return tot if tot else t.numel() * t.element_size()


def main():
    gen = torch.Generator(device=dev).manual_seed(0)
    rows = []
    header = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda}
    print(header, flush=True)
    t0 = time.time()
    for size in SIZES:
        for shape in ("unstructured", "block16"):
            for density in DENS:
                mask = make_mask(size, density, shape, gen)
                W32 = torch.randn(size, size, device=dev, generator=gen) * mask
                for dtype in DTYPES:
                    W = W32.to(dtype)
                    dname = "fp32" if dtype == torch.float32 else "fp16"
                    formats = {}
                    try:
                        formats["csr"] = W.to_sparse_csr()
                    except Exception as e:  # noqa: BLE001
                        formats["csr"] = e
                    if shape == "block16":
                        for bs in (16, 32):
                            try:
                                formats[f"bsr{bs}"] = W.to_sparse_bsr((bs, bs))
                            except Exception as e:  # noqa: BLE001
                                formats[f"bsr{bs}"] = e
                    if shape == "unstructured" and density == 0.5 and dtype == torch.float16:
                        try:
                            from torch.sparse import to_sparse_semi_structured
                            g = W.abs().reshape(size, size // 4, 4)
                            keep = torch.zeros_like(g, dtype=torch.bool)
                            keep.scatter_(2, g.topk(2, dim=2).indices, True)
                            W24 = (W.reshape(size, size // 4, 4) * keep).reshape(size, size).contiguous()
                            try:
                                formats["semi24"] = (to_sparse_semi_structured(W24), W24)
                            except Exception as e1:  # noqa: BLE001  cuSPARSELt 없으면 CUTLASS 백엔드로 재시도
                                from torch.sparse import SparseSemiStructuredTensor
                                SparseSemiStructuredTensor._FORCE_CUTLASS = True
                                try:
                                    formats["semi24"] = (to_sparse_semi_structured(W24), W24)
                                except Exception as e2:  # noqa: BLE001
                                    formats["semi24"] = RuntimeError(f"cuSPARSELt: {str(e1)[:60]} | CUTLASS: {str(e2)[:80]}")
                        except Exception as e:  # noqa: BLE001
                            formats["semi24"] = e
                    for N in NS:
                        X = torch.randn(size, N, device=dev, generator=gen).to(dtype)
                        dense_ms = bench(lambda: torch.mm(W, X))
                        rows.append(dict(size=size, N=N, density=density, shape=shape, format="dense", dtype=dname,
                                         ms=dense_ms, dense_ms=dense_ms, speedup=1.0, bytes=W.numel() * W.element_size(),
                                         status="ok", error=""))
                        for fname, obj in formats.items():
                            if isinstance(obj, Exception):
                                rows.append(dict(size=size, N=N, density=density, shape=shape, format=fname, dtype=dname, ms=None,
                                                 dense_ms=dense_ms, speedup=None, bytes=None, status="unsupported",
                                                 error=f"{type(obj).__name__}: {str(obj)[:160]}"))
                                continue
                            try:
                                if fname == "semi24":
                                    Ws, Wd = obj
                                    ms = bench(lambda: Ws @ X)
                                    # dense 기준은 같은 2:4 투영 행렬의 빽빽한 곱
                                    d24 = bench(lambda: torch.mm(Wd, X))
                                    rows.append(dict(size=size, N=N, density=density, shape="2of4", format="semi24", dtype=dname,
                                                     ms=ms, dense_ms=d24, speedup=d24 / ms, bytes=nbytes(Ws), status="ok", error=""))
                                else:
                                    Ws = obj
                                    ms = bench(lambda: Ws @ X)
                                    rows.append(dict(size=size, N=N, density=density, shape=shape, format=fname, dtype=dname,
                                                     ms=ms, dense_ms=dense_ms, speedup=dense_ms / ms, bytes=nbytes(Ws), status="ok", error=""))
                            except Exception as e:  # noqa: BLE001
                                rows.append(dict(size=size, N=N, density=density, shape=shape, format=fname, dtype=dname, ms=None,
                                                 dense_ms=dense_ms, speedup=None, bytes=None, status="unsupported",
                                                 error=f"{type(e).__name__}: {str(e)[:160]}"))
                    del formats
                    torch.cuda.empty_cache()
                print(f"  size {size} {shape} d={density:g} done ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["size", "N", "density", "shape", "format", "dtype", "ms", "dense_ms", "speedup", "bytes", "status", "error"])
        w.writeheader()
        w.writerows(rows)
    with open(OUT.replace(".csv", "_env.txt"), "w", encoding="utf-8") as f:
        f.write(str(header) + "\n")
    print(f"[done] e1_micro: {len(rows)} rows -> {OUT} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
