# -*- coding: utf-8 -*-
"""P4-A: 같은 메모리 상한에서 누가 더 잘 배우나 — Split CIFAR-10 단일 통과 온라인 클래스 증분 학습.

망: CNN16 (BN 없음, 3x3 합성곱 16-16 / 32-32 / 64-64, 최대 풀링 3 회, 완전연결 64, 머리 10; 매개변수 138,330 개 = 553KB fp32).
희소 10%: 첫 합성곱·머리는 그대로, 나머지 6 층 정적 무작위 10% (값 + 비트맵으로 장부).
학습기 (모두 모멘텀 없는 SGD, 옵티마이저 상태 0 바이트):
  bp      일반 역전파.
  bp_mem  블록 체크포인팅 + 층별 즉시 갱신(in-place SGD, 기울기 버퍼를 층 하나 크기로).
  np      노드 섭동 (모든 층 동시 섭동, 3 회 순전파: 깨끗 → 섭동 → 같은 잡음을 시드로 재생성하며 층별 국소 갱신).
          층마다 국소 기울기 = 출력 섭동 xi 와 입력 x 의 바깥곱 (그 층 하나만 autograd), 기울기 버퍼는 층 하나 크기.
  fg      순방향 기울기 (torch.func.jvp, 방향 K=4, 마스크 안 방향).
장부(바이트, 이상적 구현 기준 해석식; 실제 GPU 사용량은 참고로만 기록):
  매개변수 + 학습기 작업 메모리(배치 크기 x 표본당 바이트 + 매개변수 크기 버퍼) + 재현 버퍼(이미지 uint8 3,072 + 라벨 1 = 3,073 바이트/장).
  스트림 배치 b 는 {10,5,2,1} 중 '재현 배치 r=b 와 재현 버퍼 >= b 장' 이 들어가는 가장 큰 값, 아니면 재현 없이 b, 아니면 '들어가지 않음'.
발산 대비: 2 개 링 체크포인트(실험 안전장치, 장부에 넣지 않음, 모든 학습기 동일) — 발산하면 가장 오래된 것으로 되돌리고 lr 절반.
결과: results/p4/iso/<budget>/<net>/<learner>/seed<k>.json
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.masked_layers import apply_masks, convert_to_masked, masked_modules  # noqa: E402
from utils.cifar_gpu import load_cifar10_gpu, normalize                         # noqa: E402
from utils.seed import set_seed                                                 # noqa: E402

RES = os.path.join(REPO_ROOT, "results", "p4", "iso")
KB = 1024
IMG_BYTES = 3 * 32 * 32 + 1
BATCHES = (10, 5, 2, 1)
EVAL_EVERY = 1000          # 스트림 표본마다
LOSS_BLOWUP = 20.0

# 층별 텐서 크기(표본당 원소 수): (입력, 출력)  — CNN16, 32x32 입력
LAYER_IO = [(3072, 16384), (16384, 16384), (4096, 8192), (8192, 8192), (2048, 4096), (4096, 4096), (1024, 64), (64, 10)]
RELU_ELEMS = 16384 + 16384 + 8192 + 8192 + 4096 + 4096 + 64
POOL_OUT = 4096 + 2048 + 1024
BLOCK_BOUNDARIES = 3072 + 4096 + 2048 + 1024 + 64
MAX_T = 16384


class CNN16(nn.Module):
    def __init__(self, c=(16, 32, 64), fc=64, nc=10):
        super().__init__()
        self.c1 = nn.Conv2d(3, c[0], 3, padding=1)
        self.c2 = nn.Conv2d(c[0], c[0], 3, padding=1)
        self.c3 = nn.Conv2d(c[0], c[1], 3, padding=1)
        self.c4 = nn.Conv2d(c[1], c[1], 3, padding=1)
        self.c5 = nn.Conv2d(c[1], c[2], 3, padding=1)
        self.c6 = nn.Conv2d(c[2], c[2], 3, padding=1)
        self.fc = nn.Linear(c[2] * 16, fc)
        self.head = nn.Linear(fc, nc)

    def layers(self):
        return [self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.fc, self.head]

    def block(self, i, x):
        a, b = [(self.c1, self.c2), (self.c3, self.c4), (self.c5, self.c6)][i]
        return F.max_pool2d(F.relu(b(F.relu(a(x)))), 2)

    def forward(self, x):
        for i in range(3):
            x = self.block(i, x)
        return self.head(F.relu(self.fc(x.flatten(1))))


POOL_AFTER = {1, 3, 5}       # c2, c4, c6 뒤에 풀링 (층 번호 0..7)


def layer_forward(model: CNN16, k: int, h: torch.Tensor) -> torch.Tensor:
    """층 k 의 선형 연산만 (활성화·풀링 제외)."""
    m = model.layers()[k]
    if k == 6:
        h = h.flatten(1)
    w = m.weight * m.weight_mask if hasattr(m, "weight_mask") else m.weight
    return F.conv2d(h, w, m.bias, padding=1) if isinstance(m, nn.Conv2d) else F.linear(h, w, m.bias)


def post(k: int, a: torch.Tensor) -> torch.Tensor:
    if k == 7:
        return a
    a = F.relu(a)
    return F.max_pool2d(a, 2) if k in POOL_AFTER else a


def build(net: str, seed: int, device) -> CNN16:
    set_seed(seed)
    m = CNN16()
    m = convert_to_masked(m)
    if net == "sparse10":
        g = torch.Generator(device="cpu").manual_seed(50_000 + seed)
        with torch.no_grad():
            for name, mod in masked_modules(m):
                if name in ("c1", "head"):
                    continue
                n = mod.weight_mask.numel()
                keep = torch.zeros(n, dtype=torch.bool)
                keep[torch.randperm(n, generator=g)[:max(1, int(round(0.1 * n)))]] = True
                mod.weight_mask.copy_(keep.view_as(mod.weight_mask).float())
                # 희소 초기화 보정: 남은 입력 수에 맞춰 가중치 규모를 키운다 (He 초기화 기준 fan-in 10%)
                mod.weight.mul_(math.sqrt(1.0 / 0.1))
    m = m.to(device)
    apply_masks(m)
    return m


# ---------------------------------------------------------------------------
# 장부
# ---------------------------------------------------------------------------
def param_stats(model: CNN16) -> dict:
    vals, total_w, max_layer = 0, 0, 0
    for m in model.layers():
        nnz = int(m.weight_mask.sum().item()) if hasattr(m, "weight_mask") else m.weight.numel()
        layer_vals = nnz + (m.bias.numel() if m.bias is not None else 0)
        vals += layer_vals
        total_w += m.weight.numel()
        max_layer = max(max_layer, layer_vals)
    sparse = any(hasattr(m, "weight_mask") and m.weight_mask.mean().item() < 0.999 for m in model.layers())
    return {"values": vals, "bitmap_bytes": (total_w // 8) if sparse else 0, "max_layer_values": max_layer}


def ledger(learner: str, ps: dict, b_eff: int, K: int = 4) -> dict:
    """학습 1 스텝의 최대 바이트(재현 버퍼 제외). 이상적 구현 기준."""
    P = 4 * ps["values"] + ps["bitmap_bytes"]
    if learner == "bp":
        per = 4 * sum(i for i, _ in LAYER_IO) + RELU_ELEMS // 8 + POOL_OUT + 4 * 2 * MAX_T
        extra = 4 * ps["values"]                                   # 기울기 전부
    elif learner == "bp_mem":
        per = 4 * BLOCK_BOUNDARIES + 4 * 16384 + (16384 * 2) // 8 + 4096 + 4 * 2 * MAX_T
        extra = 4 * ps["max_layer_values"]                         # 층 하나 기울기
    elif learner == "np":
        per = 4 * max(i + 2 * o for i, o in LAYER_IO) + 8         # 입력 + 출력 + 잡음, 표본별 손실 2 개
        extra = 4 * ps["max_layer_values"]
    elif learner == "fg":
        per = 4 * 2 * max(i + o for i, o in LAYER_IO)             # 원값 + 접선
        extra = 4 * ps["values"] * 2                               # 매개변수 접선 + 기울기 누적
    else:
        raise KeyError(learner)
    return {"params": P, "extra": extra, "per_sample": per, "work": extra + per * b_eff, "total_wo_replay": P + extra + per * b_eff}


def plan_budget(learner: str, ps: dict, budget: int, policy: str = "max_batch") -> Optional[dict]:
    if policy == "max_buffer":                        # 스트림 배치 1, 재현 배치 1, 나머지 전부 재현 버퍼 (사전 등록 수정, 02:5x)
        L = ledger(learner, ps, 2)
        cap = (budget - L["total_wo_replay"]) // IMG_BYTES
        if cap >= 1:
            return {"b": 1, "r": 1, "replay_cap": int(cap), **L, "replay_bytes": int(cap) * IMG_BYTES}
        L = ledger(learner, ps, 1)
        return {"b": 1, "r": 0, "replay_cap": 0, **L, "replay_bytes": 0} if L["total_wo_replay"] <= budget else None
    for b in BATCHES:
        L = ledger(learner, ps, 2 * b)
        cap = (budget - L["total_wo_replay"]) // IMG_BYTES
        if cap >= b:
            return {"b": b, "r": b, "replay_cap": int(cap), **L, "replay_bytes": int(cap) * IMG_BYTES}
    for b in BATCHES:
        L = ledger(learner, ps, b)
        if L["total_wo_replay"] <= budget:
            return {"b": b, "r": 0, "replay_cap": 0, **L, "replay_bytes": 0}
    return None


FWD_EQ = {"bp": 3.0, "bp_mem": 4.0, "np": 3.0 + 1.0, "fg": None}   # 업데이트당 순전파 등가 (fg 는 3K)


# ---------------------------------------------------------------------------
# 학습기
# ---------------------------------------------------------------------------
class Base:
    def __init__(self, model: CNN16, lr: float, cfg: dict):
        self.model, self.lr, self.cfg = model, lr, cfg

    def masks(self):
        return [getattr(m, "weight_mask", None) for m in self.model.layers()]


class BP(Base):
    def step(self, x, y):
        self.model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(self.model(x), y)
        loss.backward()
        with torch.no_grad():
            for m in self.model.layers():
                for p in (m.weight, m.bias):
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-self.lr)
            apply_masks(self.model)
        return float(loss.item())


class BPMem(Base):
    """블록 체크포인팅 + 층별 즉시 갱신. 수학적으로는 BP 와 같은 기울기 — 다른 것은 최대 메모리뿐."""

    def __init__(self, model, lr, cfg):
        super().__init__(model, lr, cfg)
        self.handles = []
        for m in model.layers():
            for p in (m.weight, m.bias):
                self.handles.append(p.register_post_accumulate_grad_hook(self._hook(m, p)))

    def _hook(self, m, p):
        def h(param):
            with torch.no_grad():
                param.add_(param.grad, alpha=-self.lr)
                if param is m.weight and hasattr(m, "weight_mask"):
                    param.mul_(m.weight_mask)
            param.grad = None
        return h

    def step(self, x, y):
        from torch.utils.checkpoint import checkpoint
        h = x
        for i in range(3):
            h = checkpoint(self.model.block, i, h, use_reentrant=False)
        loss = F.cross_entropy(self.model.head(F.relu(self.model.fc(h.flatten(1)))), y)
        loss.backward()
        return float(loss.item())


class NP(Base):
    """노드 섭동: 깨끗한 순전파 → 모든 층 출력에 잡음을 넣은 순전파 → 같은 잡음을 시드로 재생성하며 층별 국소 갱신."""

    def __init__(self, model, lr, cfg):
        super().__init__(model, lr, cfg)
        self.sigma = float(cfg.get("sigma", 0.1))
        self.gen = torch.Generator(device=next(model.parameters()).device)
        self.counter = int(cfg.get("seed", 0)) * 1_000_003

    def _noise(self, shape, seed, device):
        self.gen.manual_seed(seed)
        return torch.randn(shape, generator=self.gen, device=device) * self.sigma

    @torch.no_grad()
    def step(self, x, y):
        model, dev = self.model, x.device
        self.counter += 1
        base_seed = self.counter * 97
        h = x
        for k in range(8):
            h = post(k, layer_forward(model, k, h))
        l0 = F.cross_entropy(h, y, reduction="none")
        h = x
        for k in range(8):
            a = layer_forward(model, k, h)
            a = a + self._noise(a.shape, base_seed + k, dev)
            h = post(k, a)
        l1 = F.cross_entropy(h, y, reduction="none")
        c = (l1 - l0) / (self.sigma ** 2)                                 # 표본별 방송 (B,)
        h = x
        B = x.shape[0]
        for k, m in enumerate(model.layers()):
            hin = h.flatten(1) if k == 6 else h
            a = layer_forward(model, k, h)
            xi = self._noise(a.shape, base_seed + k, dev)
            gout = xi * c.view(-1, *([1] * (xi.dim() - 1))) / B
            with torch.enable_grad():
                w = m.weight.detach().requires_grad_(True)
                bb = m.bias.detach().requires_grad_(True)
                wm = w * m.weight_mask if hasattr(m, "weight_mask") else w
                out = F.conv2d(hin, wm, bb, padding=1) if isinstance(m, nn.Conv2d) else F.linear(hin, wm, bb)
                gw, gb = torch.autograd.grad(out, (w, bb), grad_outputs=gout)
            a = a + xi                                                       # 같은 섭동 경로를 이어 간다 (갱신 전 가중치로 계산한 a)
            m.weight.add_(gw, alpha=-self.lr)
            m.bias.add_(gb, alpha=-self.lr)
            if hasattr(m, "weight_mask"):
                m.weight.mul_(m.weight_mask)
            h = post(k, a)
        return float(l0.mean().item())


class FG(Base):
    def __init__(self, model, lr, cfg):
        super().__init__(model, lr, cfg)
        self.K = int(cfg.get("dirs", 4))
        self.mask_by_name = {}
        for name, m in model.named_modules():
            if hasattr(m, "weight_mask"):
                self.mask_by_name[name + ".weight"] = m.weight_mask
        self.buffers = dict(model.named_buffers())

    def step(self, x, y):
        params = {n: p.detach() for n, p in self.model.named_parameters()}

        def loss_fn(p):
            return F.cross_entropy(functional_call(self.model, (p, self.buffers), (x,)), y)

        acc = {n: torch.zeros_like(p) for n, p in params.items()}
        loss = None
        for _ in range(self.K):
            v = {n: torch.randn_like(p) for n, p in params.items()}
            for n, mk in self.mask_by_name.items():
                v[n].mul_(mk)
            loss, d = jvp(loss_fn, (params,), (v,))
            for n in acc:
                acc[n].add_(d * v[n])
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.add_(acc[n] / self.K, alpha=-self.lr)
            apply_masks(self.model)
        return float(loss.item())


LEARNERS = {"bp": BP, "bp_mem": BPMem, "np": NP, "fg": FG}


# ---------------------------------------------------------------------------
# 데이터·평가
# ---------------------------------------------------------------------------
_DATA: Dict[str, tuple] = {}


def get_data(device):
    if "d" not in _DATA:
        (x_tr, y_tr), (x_te, y_te), (mean, std) = load_cifar10_gpu(device=device)
        g = torch.Generator(device="cpu").manual_seed(2468)
        val_idx = []
        for c in range(10):
            idx = (y_tr == c).nonzero().flatten().cpu()
            val_idx.append(idx[torch.randperm(idx.numel(), generator=g)[:500]])
        val_idx = torch.cat(val_idx)
        mask = torch.ones(y_tr.shape[0], dtype=torch.bool)
        mask[val_idx] = False
        tr_idx = mask.nonzero().flatten()
        _DATA["d"] = (x_tr[tr_idx.to(device)], y_tr[tr_idx.to(device)], x_tr[val_idx.to(device)], y_tr[val_idx.to(device)], x_te, y_te, mean, std)
    return _DATA["d"]


def stream_order(y: torch.Tensor, seed: int, n_tasks: int = 5) -> List[torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for t in range(n_tasks):
        idx = ((y == 2 * t) | (y == 2 * t + 1)).nonzero().flatten().cpu()
        out.append(idx[torch.randperm(idx.numel(), generator=g)].to(y.device))
    return out


@torch.no_grad()
def acc_on(model, x, y, classes, mean, std, bs=1000) -> float:
    sel = torch.isin(y, torch.tensor(classes, device=y.device))
    xs, ys = x[sel], y[sel]
    if ys.numel() == 0:
        return float("nan")
    correct = 0
    for s in range(0, ys.numel(), bs):
        correct += int((model(normalize(xs[s:s + bs], mean, std)).argmax(1) == ys[s:s + bs]).sum().item())
    return correct / ys.numel()


# ---------------------------------------------------------------------------
# 한 런
# ---------------------------------------------------------------------------
def run_one(cfg: dict, log=print) -> dict:
    budget, net, lname, seed, lr0 = int(cfg["budget"]), cfg["net"], cfg["learner"], int(cfg["seed"]), float(cfg["lr"])
    n_tasks = int(cfg.get("n_tasks", 5))
    eval_on = cfg.get("eval_on", "test")
    device = torch.device("cuda")
    x_tr, y_tr, x_val, y_val, x_te, y_te, mean, std = get_data(device)
    xe, ye = (x_val, y_val) if eval_on == "val" else (x_te, y_te)
    model = build(net, seed, device)
    ps = param_stats(model)
    plan = plan_budget(lname, ps, budget, cfg.get("policy", "max_batch"))
    res = {"cfg": dict(cfg), "param_stats": ps, "plan": plan}
    out = cfg.get("out")
    if plan is None:
        res["fits"] = False
        log(f"  [p4 {budget // KB}KB {net} {lname} s{seed}] does not fit (params {(4 * ps['values'] + ps['bitmap_bytes']) // KB}KB)")
        if out:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return res
    res["fits"] = True
    b, r, cap = plan["b"], plan["r"], plan["replay_cap"]
    set_seed(seed * 1000 + 13)
    lr = lr0
    learner = LEARNERS[lname](model, lr, dict(cfg, seed=seed))
    order = stream_order(y_tr, seed + 99)[:n_tasks]
    if cfg.get("max_per_task"):
        order = [o[:int(cfg["max_per_task"])] for o in order]
    buf_x = torch.empty((cap, 3, 32, 32), dtype=torch.uint8, device=device) if cap > 0 else None
    buf_y = torch.empty((cap,), dtype=torch.long, device=device) if cap > 0 else None
    n_buf, seen = 0, 0
    rng = torch.Generator(device="cpu").manual_seed(seed + 7)
    curve: List[dict] = []
    task_peak = [0.0] * n_tasks
    ring = deque(maxlen=2)
    ring.append(copy.deepcopy(model.state_dict()))
    backoffs, failed = 0, False
    torch.cuda.reset_peak_memory_stats()
    mem0 = torch.cuda.memory_allocated()
    t0 = time.time()
    steps = 0
    for t, idx in enumerate(order):
        classes_seen = list(range(2 * (t + 1)))
        for s in range(0, idx.numel(), b):
            bi = idx[s:s + b]
            xb, yb = x_tr[bi], y_tr[bi]
            if r > 0 and n_buf > 0:
                ri = torch.randint(0, n_buf, (min(r, n_buf),), generator=rng).to(device)
                xb = torch.cat([xb, buf_x[ri]])
                yb = torch.cat([yb, buf_y[ri]])
            if lname == "bp_mem":
                learner.lr = lr
            else:
                learner.lr = lr
            loss = learner.step(normalize(xb, mean, std), yb)
            steps += 1
            if (not math.isfinite(loss)) or loss > LOSS_BLOWUP:
                backoffs += 1
                if backoffs > 5:
                    failed = True
                    break
                model.load_state_dict(ring[0])
                lr *= 0.5
            # 저수지 표집으로 재현 버퍼 갱신
            if cap > 0:
                for j in range(bi.numel()):
                    seen += 1
                    if n_buf < cap:
                        buf_x[n_buf], buf_y[n_buf] = x_tr[bi[j]], y_tr[bi[j]]
                        n_buf += 1
                    else:
                        k = int(torch.randint(0, seen, (1,), generator=rng).item())
                        if k < cap:
                            buf_x[k], buf_y[k] = x_tr[bi[j]], y_tr[bi[j]]
            done_samples = sum(o.numel() for o in order[:t]) + min(s + b, idx.numel())
            if done_samples % EVAL_EVERY < b or (s + b >= idx.numel()):
                a_seen = acc_on(model, xe, ye, classes_seen, mean, std)
                per_task = [acc_on(model, xe, ye, [2 * u, 2 * u + 1], mean, std) for u in range(t + 1)]
                for u, a in enumerate(per_task):
                    task_peak[u] = max(task_peak[u], a)
                curve.append({"samples": done_samples, "task": t, "acc_seen": a_seen, "loss": loss, "lr": lr})
                ring.append(copy.deepcopy(model.state_dict()))
        if failed:
            break
    a_last = acc_on(model, xe, ye, list(range(2 * n_tasks)), mean, std)
    final_task = [acc_on(model, xe, ye, [2 * u, 2 * u + 1], mean, std) for u in range(n_tasks)]
    forgetting = float(np.mean([task_peak[u] - final_task[u] for u in range(n_tasks - 1)])) if n_tasks > 1 else 0.0
    fwd_eq = FWD_EQ[lname] if lname != "fg" else 3.0 * learner.K
    res.update({
        "a_last": a_last, "a_auc": float(np.mean([c["acc_seen"] for c in curve])) if curve else float("nan"),
        "forgetting": forgetting, "final_task_acc": final_task, "curve": curve, "steps": steps, "backoffs": backoffs, "failed": failed,
        "lr_final": lr, "fwd_eq_per_update": fwd_eq, "samples_per_update": b + r,
        "fwd_eq_total": fwd_eq * steps * (b + r), "gpu_peak_minus_base_bytes": int(torch.cuda.max_memory_allocated() - mem0),
        "elapsed_s": time.time() - t0, "eval_on": eval_on})
    log(f"  [p4 {budget // KB}KB {net} {lname} lr={lr0:g} s{seed}] b={b} r={r} buf={cap} A_last {a_last:.4f} A_auc {res['a_auc']:.4f} "
        f"forget {forgetting:.3f} bo {backoffs}{' FAILED' if failed else ''} ({res['elapsed_s']:.0f}s)")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return res


if __name__ == "__main__":
    # 장부 표: python experiments/p4/iso_memory.py ledger
    dev = torch.device("cuda")
    if len(sys.argv) > 1 and sys.argv[1] == "ledger":
        for net in ("dense", "sparse10"):
            ps = param_stats(build(net, 0, dev))
            print(net, "params", (4 * ps["values"] + ps["bitmap_bytes"]) // KB, "KB", ps)
            for budget in (256 * KB, 384 * KB, 512 * KB, 1024 * KB, 4096 * KB):
                row = []
                for l in ("bp", "bp_mem", "np", "fg"):
                    p = plan_budget(l, ps, budget)
                    row.append(f"{l}:" + ("X" if p is None else f"b{p['b']}r{p['r']}buf{p['replay_cap']}"))
                print(f"  {budget // KB}KB  " + "  ".join(row))
