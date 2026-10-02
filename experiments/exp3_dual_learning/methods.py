# -*- coding: utf-8 -*-
"""실험 3: 연속학습 방법들.

공통 인터페이스 (ContinualMethod):
  begin_task(t) -> observe(x, y, t) 반복 -> end_task(t, task_x, task_y) -> predict(x)
  train_flops : 학습에 쓴 순전파 표본 수 x 샘플당 순전파 FLOPs x 3 (역전파 2 배 근사). 리플레이/수면 포함.

방법:
- finetune : 단일 망, 스트림만 학습. 치명적 망각의 하한 대조군.
- er       : Experience Replay. 저수지 샘플링 버퍼, 매 스텝 스트림 배치 + 리플레이 배치.
- joint    : 태스크 끝마다 지금까지 본 모든 데이터로 처음부터 재학습. 상한 대조군.
- cls      : 이중 학습률 + 수면 (ARCHITECTURE.md D 절).
             fast(해마) 는 높은 lr 로 스트림만 학습. slow(피질) 는 수면 단계에서만 학습:
             현재 태스크 데이터 + 버퍼 리플레이 위에서 CE + fast 로부터의 지식 증류.
             수면 시작 때 전역 감쇠(downscale), 수면 끝에 약한 연결 가지치기(prune_frac).
             예측은 slow. 확신도 신호로 softmax max, 예측 엔트로피, fast-slow 합의도를 낸다.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.masked_layers import active_connections, apply_masks, convert_to_masked, mask_density
from core.pruning import ActivityTracker, prune_to_density
from utils.metrics import count_active_params, count_flops, count_params


class ReservoirBuffer:
    def __init__(self, capacity: int, x_shape, device):
        self.capacity = capacity
        self.x = torch.zeros((capacity,) + tuple(x_shape), device=device)
        self.y = torch.zeros(capacity, dtype=torch.long, device=device)
        self.t = torch.zeros(capacity, dtype=torch.long, device=device)
        self.n_seen = 0
        self.filled = 0

    @torch.no_grad()
    def add(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> None:
        for i in range(x.shape[0]):
            if self.filled < self.capacity:
                j = self.filled
                self.filled += 1
            else:
                j = int(torch.randint(0, self.n_seen + 1, (1,)).item())
                if j >= self.capacity:
                    self.n_seen += 1
                    continue
            self.x[j] = x[i]
            self.y[j] = y[i]
            self.t[j] = task_id
            self.n_seen += 1

    def sample(self, n: int):
        k = min(n, self.filled)
        idx = torch.randint(0, self.filled, (k,), device=self.x.device)
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return self.filled


class ContinualMethod:
    name = "base"

    def __init__(self, model_fn, device, cfg: dict, input_shape):
        self.model_fn = model_fn
        self.device = device
        self.cfg = cfg
        self.input_shape = tuple(input_shape)
        self.train_flops = 0.0
        self.fwd_flops_dense = None  # 샘플당 순전파 FLOPs (dense)

    def _account(self, n_samples: int, model: nn.Module) -> None:
        if self.fwd_flops_dense is None:
            self.fwd_flops_dense = count_flops(model, self.input_shape, self.device).dense
        self.train_flops += 3.0 * n_samples * self.fwd_flops_dense

    def begin_task(self, t: int) -> None:
        pass

    def observe(self, x: torch.Tensor, y: torch.Tensor, t: int) -> float:
        raise NotImplementedError

    def end_task(self, t: int, task_x: torch.Tensor, task_y: torch.Tensor) -> None:
        pass

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def confidences(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """확신도 신호들 (클수록 확신). 기본: softmax max, 음의 엔트로피."""
        p = torch.softmax(self.predict(x).float(), dim=1)
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1)
        return {"softmax_max": p.max(1).values, "neg_entropy": -ent}

    def predictor(self) -> nn.Module:
        raise NotImplementedError

    def summary(self) -> dict:
        m = self.predictor()
        rep = count_flops(m, self.input_shape, self.device)
        return {"params_total": count_params(m), "params_active": count_active_params(m),
                "infer_flops_dense": rep.dense, "infer_flops_effective": rep.effective,
                "train_flops": self.train_flops}


class Finetune(ContinualMethod):
    name = "finetune"

    def __init__(self, model_fn, device, cfg, input_shape):
        super().__init__(model_fn, device, cfg, input_shape)
        self.model = model_fn().to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=float(cfg.get("lr", 1e-3)))

    def observe(self, x, y, t):
        self.model.train()
        loss = F.cross_entropy(self.model(x), y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self._account(x.shape[0], self.model)
        return float(loss.item())

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        return self.model(x)

    def predictor(self):
        return self.model


class ExperienceReplay(Finetune):
    name = "er"

    def __init__(self, model_fn, device, cfg, input_shape):
        super().__init__(model_fn, device, cfg, input_shape)
        self.buffer = ReservoirBuffer(int(cfg.get("buffer", 500)), input_shape, device)
        self.replay_bs = int(cfg.get("replay_bs", 128))

    def observe(self, x, y, t):
        self.model.train()
        if len(self.buffer) > 0:
            xr, yr = self.buffer.sample(self.replay_bs)
            xa, ya = torch.cat([x, xr]), torch.cat([y, yr])
        else:
            xa, ya = x, y
        loss = F.cross_entropy(self.model(xa), ya)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self._account(xa.shape[0], self.model)
        self.buffer.add(x, y, t)
        return float(loss.item())


class Joint(ContinualMethod):
    """상한: 태스크 끝마다 지금까지의 모든 데이터로 처음부터 재학습."""
    name = "joint"

    def __init__(self, model_fn, device, cfg, input_shape):
        super().__init__(model_fn, device, cfg, input_shape)
        self.model = model_fn().to(device)
        self.xs, self.ys = [], []
        self.epochs = int(cfg.get("epochs", 3))
        self.bs = int(cfg.get("batch_size", 128))
        self.lr = float(cfg.get("lr", 1e-3))

    def observe(self, x, y, t):
        return 0.0

    def end_task(self, t, task_x, task_y):
        # 메모리: 태스크당 최대 max_per_task 표본만 fp16 으로 보관 (Permuted 10 태스크 x 60k 는 GPU 1.9GB 를 먹는다)
        cap = int(self.cfg.get("joint_max_per_task", 20000))
        if task_x.shape[0] > cap:
            keep = torch.randperm(task_x.shape[0], device=task_x.device)[:cap]
            task_x, task_y = task_x[keep], task_y[keep]
        self.xs.append(task_x.half())
        self.ys.append(task_y)
        X, Y = torch.cat(self.xs), torch.cat(self.ys)
        self.model = self.model_fn().to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        n = X.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=X.device)
            for s in range(0, n, self.bs):
                idx = perm[s:s + self.bs]
                loss = F.cross_entropy(self.model(X[idx].float()), Y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                self._account(idx.numel(), self.model)
        del X, Y

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        return self.model(x)

    def predictor(self):
        return self.model


class CLSSleep(ContinualMethod):
    name = "cls"

    def __init__(self, model_fn, device, cfg, input_shape):
        super().__init__(model_fn, device, cfg, input_shape)
        self.fast = model_fn().to(device)
        self.slow = convert_to_masked(model_fn()).to(device)
        self.opt_fast = torch.optim.Adam(self.fast.parameters(), lr=float(cfg.get("lr_fast", 3e-3)))
        self.opt_slow = torch.optim.Adam(self.slow.parameters(), lr=float(cfg.get("lr_slow", 5e-4)))
        self.buffer = ReservoirBuffer(int(cfg.get("buffer", 500)), input_shape, device)
        self.sleep_steps = int(cfg.get("sleep_steps", 500))
        self.sleep_bs = int(cfg.get("sleep_bs", 128))
        self.replay_bs = int(cfg.get("replay_bs", 128))
        self.kd_alpha = float(cfg.get("kd_alpha", 1.0))
        self.kd_T = float(cfg.get("kd_T", 2.0))
        self.downscale = float(cfg.get("downscale", 0.0))       # 수면 시작 때 slow 가중치에 곱할 (1 - downscale)
        self.prune_frac = float(cfg.get("prune_frac", 0.0))     # 수면 끝에 살아있는 연결 중 끊을 비율
        self.prune_rule = str(cfg.get("prune_rule", "magnitude"))
        self.reset_fast = bool(cfg.get("reset_fast", False))
        self.tracker = ActivityTracker(self.slow) if self.prune_rule.startswith("activity") else None
        self.density_log = [1.0]
        self.sleep_log = []

    def begin_task(self, t):
        if self.reset_fast and t > 0:
            self.fast = self.model_fn().to(self.device)
            self.opt_fast = torch.optim.Adam(self.fast.parameters(), lr=float(self.cfg.get("lr_fast", 3e-3)))

    def observe(self, x, y, t):
        self.fast.train()
        loss = F.cross_entropy(self.fast(x), y)
        self.opt_fast.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_fast.step()
        self._account(x.shape[0], self.fast)
        self.buffer.add(x, y, t)
        return float(loss.item())

    def end_task(self, t, task_x, task_y):
        """수면 단계: 전역 감쇠 -> 리플레이 + 증류로 slow 학습 -> 약한 연결 가지치기."""
        if self.downscale > 0:
            with torch.no_grad():
                for m in self.slow.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.mul_(1.0 - self.downscale)
        self.fast.eval()
        self.slow.train()
        n = task_x.shape[0]
        T = self.kd_T
        total = 0.0
        for it in range(self.sleep_steps):
            idx = torch.randint(0, n, (self.sleep_bs,), device=task_x.device)
            xb, yb = task_x[idx], task_y[idx]
            if len(self.buffer) > 0:
                xr, yr = self.buffer.sample(self.replay_bs)
                xa, ya = torch.cat([xb, xr]), torch.cat([yb, yr])
            else:
                xa, ya = xb, yb
            logits = self.slow(xa)
            loss = F.cross_entropy(logits, ya)
            if self.kd_alpha > 0:
                with torch.no_grad():
                    teacher = torch.log_softmax(self.fast(xb) / T, dim=1)
                student = torch.log_softmax(logits[: xb.shape[0]] / T, dim=1)
                loss = loss + self.kd_alpha * F.kl_div(student, teacher, log_target=True, reduction="batchmean") * T * T
            self.opt_slow.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_slow.step()
            apply_masks(self.slow)
            self._account(xa.shape[0] + xb.shape[0], self.slow)
            total += float(loss.item())
        if self.prune_frac > 0:
            cur = mask_density(self.slow)
            prune_to_density(self.slow, cur * (1.0 - self.prune_frac), self.prune_rule, self.tracker, scope="global")
        self.density_log.append(mask_density(self.slow))
        self.sleep_log.append({"task": t, "sleep_loss": total / max(self.sleep_steps, 1),
                               "density": self.density_log[-1], "active": active_connections(self.slow)})

    @torch.no_grad()
    def predict(self, x):
        self.slow.eval()
        return self.slow(x)

    @torch.no_grad()
    def predict_fast(self, x):
        self.fast.eval()
        return self.fast(x)

    @torch.no_grad()
    def confidences(self, x):
        out = super().confidences(x)
        ps = torch.softmax(self.predict(x).float(), dim=1)
        pf = torch.softmax(self.predict_fast(x).float(), dim=1)
        out["fast_slow_agreement"] = (ps * pf).sum(1)      # 두 망이 같은 답을 뽑을 확률
        return out

    def predictor(self):
        return self.slow

    def summary(self):
        s = super().summary()
        s["fast_params"] = count_params(self.fast)
        s["density_log"] = self.density_log
        s["sleep_log"] = self.sleep_log
        return s


METHODS = {"finetune": Finetune, "er": ExperienceReplay, "joint": Joint, "cls": CLSSleep}


# ---------------------------------------------------------------------------
# 재설계 (2026-10-01 오후, 사용자 설계): 희소 해마 + 균형 리플레이 + 증류 없음 + 크기 가지치기만
# ---------------------------------------------------------------------------

class KWTA(nn.Module):
    """k-Winners-Take-All: 표본마다 활성 상위 k 개만 남기고 나머지는 0 (치상회 패턴 분리 모방)."""

    def __init__(self, k_frac: float):
        super().__init__()
        self.k_frac = float(k_frac)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k_frac <= 0 or self.k_frac >= 1:
            return x
        k = max(1, int(round(self.k_frac * x.shape[-1])))
        thresh = x.topk(k, dim=-1).values[..., -1:]
        return torch.where(x >= thresh, x, torch.zeros_like(x))


class SparseMLP(nn.Module):
    """784 -> h -> h -> 10, 각 은닉층 ReLU 뒤에 k-WTA. k_frac=0 이면 보통 MLP."""

    def __init__(self, hidden: int = 512, k_frac: float = 0.1, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, num_classes)
        self.kwta = KWTA(k_frac)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.flatten(x, 1)
        h = self.kwta(torch.relu(self.fc1(h)))
        h = self.kwta(torch.relu(self.fc2(h)))
        return self.out(h)

    def hidden(self, x: torch.Tensor):
        """(h1, h2, logits)"""
        h = torch.flatten(x, 1)
        h1 = self.kwta(torch.relu(self.fc1(h)))
        h2 = self.kwta(torch.relu(self.fc2(h1)))
        return h1, h2, self.out(h2)


def _group_indices(task_y, buffer, seen_groups, task_id: int):
    """(태스크, 클래스) 그룹별 인덱스. 현재 태스크 그룹은 현재 데이터 인덱스, 과거 그룹은 버퍼 인덱스."""
    out = {}
    n = buffer.filled
    for (tg, c) in sorted(seen_groups):
        if tg == task_id:
            idx = (task_y == c).nonzero().squeeze(1)
        else:
            idx = ((buffer.t[:n] == tg) & (buffer.y[:n] == c)).nonzero().squeeze(1)
        if idx.numel() > 0:
            out[(tg, c)] = idx
    return out


def _balanced_batch(group_idx, task_x, task_y, buffer, bs: int, device, task_id: int):
    """지금까지 본 (태스크, 클래스) 그룹이 모두 같은 수가 되게 배치를 만든다. group_idx 는 _group_indices 의 결과.
    Split MNIST 에서는 클래스 균형과 같고, Permuted MNIST 에서는 태스크 균형이 된다."""
    groups = list(group_idx.keys())
    per = max(1, bs // max(len(groups), 1))
    xs, ys = [], []
    for (tg, c) in groups:
        idx = group_idx[(tg, c)]
        pick = idx[torch.randint(0, idx.numel(), (per,), device=device)]
        if tg == task_id:
            xs.append(task_x[pick])
            ys.append(task_y[pick])
        else:
            xs.append(buffer.x[pick])
            ys.append(buffer.y[pick])
    return torch.cat(xs), torch.cat(ys)


class ERBalanced(Finetune):
    """온라인 ER 의 균형 판: 매 스텝 배치를 '본 클래스 모두 같은 수' 로 만든다 (현재 클래스는 스트림 배치에서,
    과거 클래스는 버퍼에서). 균형 리플레이 효과를 구조 변경과 분리해서 보는 대조군."""
    name = "er_balanced"

    def __init__(self, model_fn, device, cfg, input_shape):
        super().__init__(model_fn, device, cfg, input_shape)
        self.buffer = ReservoirBuffer(int(cfg.get("buffer", 500)), input_shape, device)
        self.bs = int(cfg.get("batch_size", 128))
        self.seen = set()

    def observe(self, x, y, t):
        self.model.train()
        cur = set(int(c) for c in torch.unique(y).tolist())
        self.seen |= set((t, c) for c in cur)
        self._step = getattr(self, "_step", 0) + 1
        if self._step % 10 == 1 or getattr(self, "_gi_task", None) != t:
            self._gi = _group_indices(y, self.buffer, self.seen, t)
            self._gi_task = t
        else:
            # 현재 태스크 그룹의 인덱스는 이번 배치 기준으로 다시 (배치마다 라벨 위치가 다름)
            for c in cur:
                self._gi[(t, c)] = (y == c).nonzero().squeeze(1)
        xa, ya = _balanced_batch(self._gi, x, y, self.buffer, self.bs, x.device, t)
        loss = F.cross_entropy(self.model(xa), ya)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self._account(xa.shape[0], self.model)
        self.buffer.add(x, y, t)
        return float(loss.item())


class CLS2(ContinualMethod):
    """재설계 CLS.
    fast(해마): SparseMLP (k-WTA 로 활성 5~10%), 높은 lr 로 스트림만 학습. 증류 없음.
    slow(피질): 수면에서만 학습. 배치는 본 클래스 모두 같은 수 (현재 클래스 = 현재 데이터, 과거 = 버퍼).
    수면 끝 가지치기는 크기 기준만 (prune_frac). 전역 감쇠 없음.
    예측: slow (기본). 대안 'max_conf' = 표본마다 fast/slow 중 확신도 높은 쪽, 'fast' = 해마 단독 (R_alt 로 기록)."""
    name = "cls2"

    def __init__(self, model_fn, device, cfg, input_shape):
        super().__init__(model_fn, device, cfg, input_shape)
        self.fast = SparseMLP(int(cfg.get("hidden_fast", 512)), float(cfg.get("k_frac", 0.1))).to(device)
        self.slow = convert_to_masked(model_fn()).to(device)
        self.opt_fast = torch.optim.Adam(self.fast.parameters(), lr=float(cfg.get("lr_fast", 3e-3)))
        self.opt_slow = torch.optim.Adam(self.slow.parameters(), lr=float(cfg.get("lr_slow", 5e-4)))
        self.buffer = ReservoirBuffer(int(cfg.get("buffer", 500)), input_shape, device)
        self.sleep_steps = int(cfg.get("sleep_steps", 500))
        self.sleep_bs = int(cfg.get("sleep_bs", 256))
        self.prune_frac = float(cfg.get("prune_frac", 0.0))
        self.seen = set()
        self.density_log = [1.0]
        self.sleep_log = []
        # 논문 2 토대 옵션
        self.decay_mode = str(cfg.get("decay_mode", "none"))
        self.downscale = float(cfg.get("downscale", 0.0))
        self.decay_every = int(cfg.get("decay_every", 50))
        self.kd_mode = str(cfg.get("kd_mode", "none"))
        self.kd_alpha = float(cfg.get("kd_alpha", 1.0))
        self.kd_T = float(cfg.get("kd_T", 2.0))
        if self.decay_mode == "continuous":
            self.opt_slow = torch.optim.Adam(self.slow.parameters(), lr=float(cfg.get("lr_slow", 5e-4)),
                                             weight_decay=float(cfg.get("sleep_wd", 1e-4)))

    @torch.no_grad()
    def _scale_slow(self, factor: float) -> None:
        for m in self.slow.modules():
            if isinstance(m, nn.Linear):
                m.weight.mul_(factor)

    def _slow_layers(self):
        return [m for m in self.slow.net if isinstance(m, nn.Linear)]

    def _local_kd_loss(self, xb: torch.Tensor) -> torch.Tensor:
        """층별 지역 특징 증류. 각 층의 입력을 detach 해 그 층 가중치만 오차를 받게 한다.
        MSE 는 출력 뉴런별로 분해되므로 뉴런 i 의 입력 가중치는 뉴런 i 의 오차만 받는다 (분산 증류)."""
        with torch.no_grad():
            self.fast.eval()
            f1, f2, flog = self.fast.hidden(xb)
        l1, l2, l3 = self._slow_layers()
        h = torch.flatten(xb, 1)
        s1 = torch.relu(l1(h))
        loss = F.mse_loss(s1, f1)
        s2 = torch.relu(l2(s1.detach()))
        loss = loss + F.mse_loss(s2, f2)
        if self.kd_mode == "local_logits":
            T = self.kd_T
            slog = l3(s2.detach())
            loss = loss + F.kl_div(torch.log_softmax(slog / T, 1), torch.log_softmax(flog / T, 1),
                                   log_target=True, reduction="batchmean") * T * T
        return loss

    def observe(self, x, y, t):
        self.fast.train()
        self.seen |= set((t, int(c)) for c in torch.unique(y).tolist())
        loss = F.cross_entropy(self.fast(x), y)
        self.opt_fast.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_fast.step()
        self._account(x.shape[0], self.fast)
        self.buffer.add(x, y, t)
        return float(loss.item())

    def end_task(self, t, task_x, task_y):
        self.slow.train()
        total = 0.0
        gi = _group_indices(task_y, self.buffer, self.seen, t)   # 수면 중에는 버퍼가 고정이라 한 번만
        if self.decay_mode == "boundary" and self.downscale > 0:
            self._scale_slow(1.0 - self.downscale)
        n_periodic = max(1, self.sleep_steps // self.decay_every) if self.decay_mode == "periodic" else 0
        per_step_factor = (1.0 - self.downscale) ** (1.0 / n_periodic) if n_periodic else 1.0
        n_cur = int(task_x.shape[0])
        for it in range(self.sleep_steps):
            if self.decay_mode == "periodic" and self.downscale > 0 and it % self.decay_every == 0:
                self._scale_slow(per_step_factor)
            xa, ya = _balanced_batch(gi, task_x, task_y, self.buffer, self.sleep_bs, task_x.device, t)
            logits = self.slow(xa)
            loss = F.cross_entropy(logits, ya)
            if self.kd_mode == "global" and self.kd_alpha > 0:
                # 기존 실패 방식: 현재 태스크 입력에 대해 fast 로짓을 전역 KL 로 전달
                idx = torch.randint(0, n_cur, (self.sleep_bs // 2,), device=task_x.device)
                xb = task_x[idx]
                T = self.kd_T
                with torch.no_grad():
                    self.fast.eval()
                    teacher = torch.log_softmax(self.fast(xb) / T, 1)
                student = torch.log_softmax(self.slow(xb) / T, 1)
                loss = loss + self.kd_alpha * F.kl_div(student, teacher, log_target=True, reduction="batchmean") * T * T
            elif self.kd_mode in ("local", "local_logits") and self.kd_alpha > 0:
                idx = torch.randint(0, n_cur, (self.sleep_bs // 2,), device=task_x.device)
                loss = loss + self.kd_alpha * self._local_kd_loss(task_x[idx])
            self.opt_slow.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_slow.step()
            apply_masks(self.slow)
            self._account(xa.shape[0], self.slow)
            total += float(loss.item())
        if self.prune_frac > 0:
            cur_d = mask_density(self.slow)
            prune_to_density(self.slow, cur_d * (1.0 - self.prune_frac), "magnitude", None, scope="global")
        self.density_log.append(mask_density(self.slow))
        self.sleep_log.append({"task": t, "sleep_loss": total / max(self.sleep_steps, 1),
                               "density": self.density_log[-1], "active": active_connections(self.slow)})

    @torch.no_grad()
    def predict(self, x):
        self.slow.eval()
        return self.slow(x)

    @torch.no_grad()
    def predict_fast(self, x):
        self.fast.eval()
        return self.fast(x)

    @torch.no_grad()
    def predict_max_conf(self, x):
        ls, lf = self.predict(x).float(), self.predict_fast(x).float()
        cs = torch.softmax(ls, 1).max(1).values
        cf = torch.softmax(lf, 1).max(1).values
        return torch.where((cf > cs)[:, None], lf, ls)

    def predict_alt(self):
        return {"fast": self.predict_fast, "max_conf": self.predict_max_conf}

    @torch.no_grad()
    def confidences(self, x):
        out = super().confidences(x)
        ps = torch.softmax(self.predict(x).float(), dim=1)
        pf = torch.softmax(self.predict_fast(x).float(), dim=1)
        out["fast_slow_agreement"] = (ps * pf).sum(1)
        out["max_conf_softmax"] = torch.maximum(ps.max(1).values, pf.max(1).values)
        return out

    def predictor(self):
        return self.slow

    def summary(self):
        s = super().summary()
        s["fast_params"] = count_params(self.fast)
        s["density_log"] = self.density_log
        s["sleep_log"] = self.sleep_log
        return s


METHODS.update({"er_balanced": ERBalanced, "cls2": CLS2})
