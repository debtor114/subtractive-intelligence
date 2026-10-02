# -*- coding: utf-8 -*-
"""실험 1+2 통합: 시상 사전 필터링 (어느 토큰을 어느 뒤쪽 층에서 건너뛸지 사전 결정) + 예측 잔차 대체.

- 앞쪽 n_front 블록은 전부 계산한다 (특징 추출).
- 경계에서 ThalamicRouter 가 토큰마다 뒤쪽 블록 각각의 예상 변화량(log1p 잔차 노름)을 한 번에 예측한다.
  뒤쪽 블록 i 는 그 점수 상위 (1 - s_i) 비율의 토큰만 계산하고 (CLS 는 항상), 나머지는 x + predictor_i(x) 로 통과시킨다.
- 선택 기준 mode: thalamic (라우터, 사전 결정) | layerwise (블록 직전 예측기 노름, 실험 2 방식) | random (대조).
- 뒤쪽 블록의 어텐션은 그대로(full) 두거나 실험 1 의 사전 라우팅(RoutedAttention, pre) 으로 바꿀 수 있다.
- 값은 '부분 계산' 과 같지만 구현은 전 토큰을 계산한 뒤 건너뛴 자리에 대체값을 넣는다 (이론 FLOPs 는 해석적으로 센다).
  이렇게 하면 어텐션 변형과 자유롭게 조합되고 미분 가능하다 (미세조정에 그대로 사용).

FLOPs (1 샘플, 해석적): 앞쪽 블록 전부 + 뒤쪽 블록은 kv 투영(전 토큰) + 계산 토큰 x (q 투영 + 어텐션 행 + proj + MLP)
  + 라우터 1 회 + 건너뛴 토큰의 예측기 비용. 어텐션 행은 full = 4 N d, pre = 2 N r + 4 k d (헤드 합산 기준 d = dim).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .predictive_coding import ResidualPredictor
from .thalamic_router import RoutedAttention


class ThalamicRouter(nn.Module):
    """토큰 표현 -> 뒤쪽 블록 n_late 개 각각의 예상 변화량 (log1p 노름)."""

    def __init__(self, dim: int, n_late: int, hidden: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, n_late)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class ThalamicSkip(nn.Module):
    """ViT + 라우터 + 뒤쪽 블록 예측기. forward(x, skip_fracs, mode) 가 로짓과 보조 손실 재료를 돌려준다."""

    def __init__(self, vit: nn.Module, n_front: int, rank: int = 16, router_hidden: int = 64):
        super().__init__()
        self.vit = vit
        self.n_front = n_front
        depth = len(vit.blocks)
        self.n_late = depth - n_front
        dim = vit.pos_embed.shape[-1]
        self.predictors = nn.ModuleList([ResidualPredictor(dim, rank) for _ in range(self.n_late)])
        self.router = ThalamicRouter(dim, self.n_late, router_hidden)
        self.rank = rank
        self.last_computed_frac = 1.0
        self.substitute = "predicted"   # 건너뛴 토큰 대체: predicted (x + 예측 잔차) 또는 identity (x 그대로). 대조군용

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        vit = self.vit
        B = x.shape[0]
        h = vit.patch_embed(x).flatten(2).transpose(1, 2)
        h = torch.cat([vit.cls_token.expand(B, -1, -1), h], dim=1)
        return vit.pos_drop(h + vit.pos_embed)

    def forward(self, x: torch.Tensor, skip_fracs: Sequence[float], mode: str = "thalamic",
                with_aux: bool = False):
        vit = self.vit
        h = self.embed(x)
        for blk in vit.blocks[: self.n_front]:
            h = blk(h)
        if mode == "drop_late":
            # 대조군: 뒤쪽 블록을 통째로 제거 (토큰 선택도 대체도 없음). 라우터/예측기는 쓰지 않는다
            self.last_computed_frac = 0.0
            h = vit.norm(h)
            pooled = h[:, 0] if vit.pool == "cls" else h[:, 1:].mean(1)
            logits = vit.head(pooled)
            if with_aux:
                return logits, h.new_zeros(()), h.new_zeros(())
            return logits
        scores_all = self.router(h)                                   # (B, N, n_late)
        B, N, _ = h.shape
        aux_router, aux_pred = h.new_zeros(()), h.new_zeros(())
        computed = 0.0
        for i, blk in enumerate(vit.blocks[self.n_front:]):
            s = float(skip_fracs[i])
            pred = self.predictors[i](h)
            out = blk(h)
            if with_aux:
                with torch.no_grad():
                    delta = (out - h).detach()
                    target = torch.log1p(delta.norm(dim=-1))
                aux_router = aux_router + F.mse_loss(scores_all[:, :, i], target)
                aux_pred = aux_pred + F.mse_loss(pred, delta)
            n_skip = int(round(s * (N - 1)))
            if n_skip <= 0:
                h = out
                computed += N
                continue
            if mode == "thalamic":
                sc = scores_all[:, :, i].detach()
            elif mode == "layerwise":
                sc = pred.detach().norm(dim=-1)
            elif mode == "random":
                sc = torch.rand(B, N, device=h.device)
            else:
                raise KeyError(mode)
            sc = sc.clone()
            sc[:, 0] = float("inf")                                   # CLS 는 항상 계산
            keep_idx = sc.topk(N - n_skip, dim=1).indices
            keep = torch.zeros(B, N, dtype=torch.bool, device=h.device).scatter_(1, keep_idx, True)
            fill = h + pred if self.substitute == "predicted" else h
            h = torch.where(keep[:, :, None], out, fill)
            computed += N - n_skip
        self.last_computed_frac = computed / (self.n_late * N)
        h = vit.norm(h)
        pooled = h[:, 0] if vit.pool == "cls" else h[:, 1:].mean(1)
        logits = vit.head(pooled)
        if with_aux:
            return logits, aux_router, aux_pred
        return logits


def swap_late_attention(vit: nn.Module, n_front: int, keep_ratio: float = 0.25, router_dim: int = 8) -> None:
    """뒤쪽 블록의 어텐션을 실험 1 의 사전 라우팅으로 교체 (가중치 복사)."""
    for blk in vit.blocks[n_front:]:
        old = blk.attn
        new = RoutedAttention(old.qkv.in_features, old.num_heads, mode="pre", keep_ratio=keep_ratio,
                              router_dim=router_dim, qkv_bias=old.qkv.bias is not None,
                              attn_drop=old.attn_drop.p, proj_drop=old.proj_drop.p, aux_weight=1.0).to(old.qkv.weight.device)
        with torch.no_grad():
            new.qkv.weight.copy_(old.qkv.weight)
            if old.qkv.bias is not None:
                new.qkv.bias.copy_(old.qkv.bias)
            new.proj.weight.copy_(old.proj.weight)
            new.proj.bias.copy_(old.proj.bias)
        blk.attn = new


def analytic_flops(dim: int, n_tokens: int, depth: int, n_front: int, skip_fracs: Sequence[float],
                   rank: int, router_hidden: int, fixed: float, late_attn: str = "full",
                   attn_keep: float = 0.25, attn_router_dim: int = 8, mlp_ratio: float = 4.0) -> Dict[str, float]:
    d, N = dim, n_tokens
    kv = 2 * 2 * d * d
    q = 2 * d * d
    proj = 2 * d * d
    mlp = 2 * 2 * d * int(d * mlp_ratio)
    attn_full_row = 2 * 2 * N * d
    k = max(1, int(math.ceil(attn_keep * N)))
    attn_pre_row = 2 * N * attn_router_dim + 2 * 2 * k * d
    full_block = N * (kv + q + attn_full_row + proj + mlp)
    total_full = fixed + depth * full_block
    total = fixed + n_front * full_block
    total += 2 * N * (dim * router_hidden + router_hidden * (depth - n_front))      # 라우터 1 회
    for i in range(depth - n_front):
        s = skip_fracs[i]
        m = N - int(round(s * (N - 1)))
        row = attn_pre_row if late_attn == "pre" else attn_full_row
        total += N * kv + m * (q + row + proj + mlp)
        total += (N - m) * 2 * 2 * dim * rank                                           # 건너뛴 토큰의 예측기
    return {"full": total_full, "skip": total, "ratio": total / total_full}
