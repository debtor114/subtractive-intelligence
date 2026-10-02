# -*- coding: utf-8 -*-
"""소형 ViT 기준선 (CIFAR-10 / MNIST).

어텐션을 nn.MultiheadAttention 대신 직접 구현한 이유:
- 실험 1(시상 사전 라우팅)에서 점수 계산 '전에' 후보를 줄이는 변형으로 교체해야 한다.
- 분석용으로 어텐션 행렬을 꺼내 볼 수 있어야 한다 (store_attn).

기본 프리셋 (configs/ 참고):
- CIFAR-10 : patch 4, dim 192, depth 6, heads 3  -> 65 토큰, 약 2.7M 파라미터
- MNIST    : patch 4, dim 128, depth 4, heads 4  -> 50 토큰
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.store_attn = False   # True 면 마지막 어텐션 행렬을 last_attn 에 보관 (분석용)
        self.last_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # (B, H, N, D)
        attn = (q @ k.transpose(-2, -1)) * self.scale         # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        if self.store_attn:
            self.last_attn = attn.detach()
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, input_shape: Sequence[int] = (3, 32, 32), num_classes: int = 10,
                 patch_size: int = 4, dim: int = 192, depth: int = 6, num_heads: int = 3,
                 mlp_ratio: float = 4.0, dropout: float = 0.0, attn_dropout: float = 0.0,
                 emb_dropout: float = 0.0, pool: str = "cls"):
        super().__init__()
        C, H, W = input_shape
        assert H % patch_size == 0 and W % patch_size == 0, "image size must be divisible by patch"
        assert pool in ("cls", "mean")
        self.pool = pool
        n_patches = (H // patch_size) * (W // patch_size)

        self.patch_embed = nn.Conv2d(C, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.pos_drop = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, dropout, attn_dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)    # (B, N, dim)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        pooled = feats[:, 0] if self.pool == "cls" else feats[:, 1:].mean(dim=1)
        return self.head(pooled)

    def set_store_attn(self, flag: bool) -> None:
        for blk in self.blocks:
            blk.attn.store_attn = flag
