# -*- coding: utf-8 -*-
"""시상 사전 라우팅 어텐션 (실험 1).

세 모드를 한 모듈로 제공한다. 모두 baselines.vit.Attention 과 같은 자리에 들어간다.

- full : 표준 어텐션. 헤드당 FLOPs = 2 N^2 D (QK^T) + 2 N^2 D (AV)
- post : 점수를 전부 계산한 뒤 쿼리마다 상위 k 만 남기고 나머지를 -inf 로 억제 (측면 억제의 사후 버전).
         이론 FLOPs 는 full 과 같다 (QK^T 를 이미 다 계산). AV 의 희소성을 활용해도 2 N^2 D + 2 N k D.
- pre  : 저차원 라우터 (D -> r, r << D) 로 먼저 후보 k 개를 고르고, 그 k 개에 대해서만 전차원 어텐션을 계산.
         헤드당 FLOPs = 2 N^2 r (라우팅) + 2 N k D (QK) + 2 N k D (AV).
         라우터는 학습 중에만 계산하는 전체 어텐션 분포를 교사로 KL 로 학습한다 (aux_loss).
         선택 자체는 미분 불가이므로, 모델은 학습 중에도 라우터가 고른 k 개로만 어텐션한다.

FLOPs 집계 주의: 선택된 키에 대한 내적은 einsum 으로 써서 bmm 으로 내려가게 했다. FlopCounterMode 는
elementwise 곱+합은 세지 않지만 bmm 은 센다. analytic_attention_flops() 로 해석적 값도 같이 낸다.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, *, mode: str = "full", keep_ratio: float = 1.0,
                 router_dim: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 aux_weight: float = 1.0):
        super().__init__()
        assert mode in ("full", "post", "pre")
        assert dim % num_heads == 0
        self.mode = mode
        self.keep_ratio = float(keep_ratio)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.router_dim = int(router_dim)
        self.aux_weight = float(aux_weight)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if mode == "pre":
            # 헤드별 저차원 투영 (D -> r). 파라미터 크기 H*D*r*2
            self.router_q = nn.Parameter(torch.randn(num_heads, self.head_dim, self.router_dim) / math.sqrt(self.head_dim))
            self.router_k = nn.Parameter(torch.randn(num_heads, self.head_dim, self.router_dim) / math.sqrt(self.head_dim))
        self.aux_loss = torch.zeros(())
        self.last_keep = None  # 마지막 순전파의 k
        self.store_attn = False
        self.last_attn = None

    def k_keep(self, n: int) -> int:
        return max(1, min(n, int(math.ceil(self.keep_ratio * n))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                  # (B, H, N, D)
        k_keep = self.k_keep(N)
        self.last_keep = k_keep

        if self.mode == "full" or (self.mode == "post" and k_keep >= N):
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            if self.store_attn:
                self.last_attn = attn.detach()
            out = self.attn_drop(attn) @ v
            self.aux_loss = torch.zeros((), device=x.device)

        elif self.mode == "post":
            scores = (q @ k.transpose(-2, -1)) * self.scale                 # (B, H, N, N)  전부 계산
            top = scores.topk(k_keep, dim=-1).indices
            mask = torch.full_like(scores, float("-inf")).scatter_(-1, top, 0.0)
            attn = (scores + mask).softmax(dim=-1)                          # 억제된 자리는 정확히 0
            if self.store_attn:
                self.last_attn = attn.detach()
            out = self.attn_drop(attn) @ v
            self.aux_loss = torch.zeros((), device=x.device)

        else:  # pre
            qr = torch.einsum("bhnd,hdr->bhnr", q, self.router_q)          # (B, H, N, r)
            kr = torch.einsum("bhnd,hdr->bhnr", k, self.router_k)
            route = torch.einsum("bhnr,bhmr->bhnm", qr, kr) / math.sqrt(self.router_dim)   # (B, H, N, N) 저차원
            if k_keep >= N:
                idx = torch.arange(N, device=x.device).view(1, 1, 1, N).expand(B, H, N, N)
            else:
                idx = route.topk(k_keep, dim=-1).indices                    # (B, H, N, k)
            # 고급 인덱싱으로 뽑는다. gather+expand 는 역전파에서 (B,H,N,N,D) 크기 버퍼를 만들어 메모리를 낭비한다.
            b_ix = torch.arange(B, device=x.device).view(B, 1, 1, 1)
            h_ix = torch.arange(H, device=x.device).view(1, H, 1, 1)
            k_sel = k[b_ix, h_ix, idx]                                      # (B, H, N, k, D)
            v_sel = v[b_ix, h_ix, idx]
            scores = torch.einsum("bhnd,bhnkd->bhnk", q, k_sel) * self.scale
            attn = scores.softmax(dim=-1)
            out = torch.einsum("bhnk,bhnkd->bhnd", self.attn_drop(attn), v_sel)
            if self.store_attn:
                full = torch.zeros(B, H, N, N, device=x.device, dtype=attn.dtype).scatter_(-1, idx, attn.detach())
                self.last_attn = full
            # 라우터 학습: 학습 중에만 전체 어텐션 분포를 교사로 삼는다 (추론 비용엔 포함되지 않음)
            if self.training and self.aux_weight > 0:
                with torch.no_grad():
                    teacher = ((q @ k.transpose(-2, -1)) * self.scale).float().softmax(dim=-1)
                student = torch.log_softmax(route.float(), dim=-1)
                self.aux_loss = self.aux_weight * F.kl_div(student, teacher, reduction="batchmean") / (H * N)
            else:
                self.aux_loss = torch.zeros((), device=x.device)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


def analytic_attention_flops(n_tokens: int, head_dim: int, num_heads: int, mode: str,
                             keep_ratio: float = 1.0, router_dim: int = 8) -> Dict[str, float]:
    """헤드 전체 합, 1 샘플 1 블록 기준. 'dense' 는 마스킹을 활용하지 않을 때, 'exploited' 는 희소성을 활용할 때."""
    N, D, H = n_tokens, head_dim, num_heads
    k = max(1, min(N, int(math.ceil(keep_ratio * N))))
    full = H * (2 * N * N * D + 2 * N * N * D)
    if mode == "full":
        return {"dense": full, "exploited": full, "k": N}
    if mode == "post":
        return {"dense": full, "exploited": H * (2 * N * N * D + 2 * N * k * D), "k": k}
    pre = H * (2 * N * N * router_dim + 2 * N * k * D + 2 * N * k * D)
    return {"dense": pre, "exploited": pre, "k": k}


def swap_attention(vit: nn.Module, *, mode: str, keep_ratio: float, router_dim: int = 8,
                   aux_weight: float = 1.0) -> nn.Module:
    """baselines.vit.ViT 의 각 블록 어텐션을 RoutedAttention 으로 교체 (qkv/proj 가중치 복사)."""
    for blk in vit.blocks:
        old = blk.attn
        new = RoutedAttention(old.qkv.in_features, old.num_heads, mode=mode, keep_ratio=keep_ratio,
                              router_dim=router_dim, qkv_bias=old.qkv.bias is not None,
                              attn_drop=old.attn_drop.p, proj_drop=old.proj_drop.p, aux_weight=aux_weight)
        new = new.to(old.qkv.weight.device)
        with torch.no_grad():
            new.qkv.weight.copy_(old.qkv.weight)
            if old.qkv.bias is not None:
                new.qkv.bias.copy_(old.qkv.bias)
            new.proj.weight.copy_(old.proj.weight)
            new.proj.bias.copy_(old.proj.bias)
        blk.attn = new
    return vit


def collect_aux_loss(model: nn.Module) -> torch.Tensor:
    total = None
    for m in model.modules():
        if isinstance(m, RoutedAttention):
            total = m.aux_loss if total is None else total + m.aux_loss
    return torch.zeros(()) if total is None else total
