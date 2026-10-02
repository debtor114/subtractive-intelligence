# -*- coding: utf-8 -*-
"""층간 예측 오차 기반 토큰 스킵 (실험 2).

아이디어 (ARCHITECTURE.md C 절의 정적 이미지판): 블록 i 의 입력 x 에서 그 블록이 토큰을 얼마나 바꿀지
(잔차 delta = block(x) - x) 를 값싼 저랭크 예측기 g_i 로 미리 추정하고, 예측된 변화가 작은 토큰은
블록 계산을 건너뛴다. 건너뛴 토큰은 x 그대로(identity) 두거나 x + g_i(x) (예측 잔차) 로 대체한다.

- ResidualPredictor : LayerNorm -> Linear(dim, r) -> GELU -> Linear(r, dim). 비용 ~ 2*2*dim*r FLOPs/토큰 (r=16 이면 블록의 1% 미만)
- fit_predictors    : 고정된 ViT 위에서 (x_in, delta) 를 모아 MSE 로 학습
- PCForward         : 블록별 스킵 비율 s 로 순전파. 점수 = ||g_i(x)|| (ours) | ||delta|| (오라클) | 난수 (대조)
                      CLS 토큰은 항상 계산. 스킵된 토큰도 키/값으로는 남는다 (A-ViT 와 같은 처리).

FLOPs (해석적, 1 샘플): 토큰당 블록 비용 c_tok = 2*(3d^2 + d^2 + 8d^2) + 어텐션 쿼리 행 (4 N d).
  계산된 토큰 수 m 이면 블록 비용 = m * c_tok (키/값 투영은 모든 토큰에 필요하므로 정확히는
  qkv 의 k,v 부분 2*2d^2 은 전 토큰에 든다 -> 이를 반영). 예측기 비용은 모든 토큰에 든다.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualPredictor(nn.Module):
    def __init__(self, dim: int, rank: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        self.rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(self.norm(x))))


@torch.no_grad()
def _block_io(vit: nn.Module, x: torch.Tensor):
    """각 블록의 입력과 잔차(출력-입력) 를 돌려준다."""
    B = x.shape[0]
    h = vit.patch_embed(x).flatten(2).transpose(1, 2)
    h = torch.cat([vit.cls_token.expand(B, -1, -1), h], dim=1)
    h = vit.pos_drop(h + vit.pos_embed)
    ins, deltas = [], []
    for blk in vit.blocks:
        out = blk(h)
        ins.append(h)
        deltas.append(out - h)
        h = out
    return ins, deltas


def fit_predictors(vit: nn.Module, loader, device, rank: int = 16, epochs: int = 3, lr: float = 1e-3,
                   log=print) -> nn.ModuleList:
    vit.eval()
    preds = nn.ModuleList([ResidualPredictor(vit.pos_embed.shape[-1], rank) for _ in vit.blocks]).to(device)
    opt = torch.optim.Adam(preds.parameters(), lr=lr)
    for ep in range(epochs):
        tot, n = 0.0, 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                ins, deltas = _block_io(vit, x)
            loss = 0.0
            for p, xin, d in zip(preds, ins, deltas):
                loss = loss + F.mse_loss(p(xin.float()), d.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            n += 1
        log(f"  [predictors] epoch {ep + 1}/{epochs} mse {tot / max(n, 1):.5f}")
    # 설명력: 마지막 배치 기준 상대 오차
    with torch.no_grad():
        rel = []
        for p, xin, d in zip(preds, ins, deltas):
            rel.append(float((F.mse_loss(p(xin.float()), d.float()) / d.float().pow(2).mean()).item()))
    log(f"  [predictors] relative mse per block: {[round(r, 3) for r in rel]}")
    preds.rel_mse = rel
    return preds


class PCForward:
    """스킵 순전파. skip_frac: 블록별 스킵 비율 (CLS 제외 토큰 중). score: predicted | oracle | random."""

    def __init__(self, vit: nn.Module, preds: Optional[nn.ModuleList], skip_frac, score: str = "predicted",
                 substitute: str = "identity"):
        self.vit, self.preds = vit, preds
        self.skip_frac = list(skip_frac)
        self.score, self.substitute = score, substitute
        self.last_computed_frac = None

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """gradient 가 흐르는 버전 (미세조정용). 추론은 __call__ 을 쓴다."""
        vit = self.vit
        B = x.shape[0]
        h = vit.patch_embed(x).flatten(2).transpose(1, 2)
        h = torch.cat([vit.cls_token.expand(B, -1, -1), h], dim=1)
        h = h + vit.pos_embed
        N = h.shape[1]
        computed = 0.0
        for i, blk in enumerate(vit.blocks):
            s = self.skip_frac[i]
            n_skip = int(round(s * (N - 1)))
            if n_skip <= 0:
                h = blk(h)
                computed += N
                continue
            pred = self.preds[i](h) if self.preds is not None else None
            if self.score == "predicted":
                score = pred.norm(dim=-1)
            elif self.score == "oracle":
                score = (blk(h) - h).norm(dim=-1)          # 참고용 상한 (실제로는 비용 절감 없음)
            else:
                score = torch.rand(B, N, device=h.device)
            score[:, 0] = float("inf")                      # CLS 는 항상 계산
            keep_idx = score.topk(N - n_skip, dim=1).indices      # (B, m)
            m = keep_idx.shape[1]
            b_ix = torch.arange(B, device=h.device).unsqueeze(1)
            # 어텐션: 키/값은 전 토큰, 쿼리는 keep 토큰만
            hn = blk.norm1(h)
            attn_out = _attention_subset(blk.attn, hn, keep_idx)         # (B, m, d)
            h_keep = h[b_ix, keep_idx] + attn_out
            h_keep = h_keep + blk.mlp(blk.norm2(h_keep))
            new_h = h.clone() if self.substitute == "identity" else (h + pred)
            new_h[b_ix, keep_idx] = h_keep
            h = new_h
            computed += m
        self.last_computed_frac = computed / (len(vit.blocks) * N)
        h = vit.norm(h)
        pooled = h[:, 0] if vit.pool == "cls" else h[:, 1:].mean(1)
        return vit.head(pooled)


def _attention_subset(attn: nn.Module, x: torch.Tensor, q_idx: torch.Tensor) -> torch.Tensor:
    """baselines.vit.Attention 과 같은 계산을, 쿼리는 q_idx 토큰만 골라서 한다."""
    B, N, C = x.shape
    H, D = attn.num_heads, attn.head_dim
    qkv = attn.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]                                   # (B, H, N, D)
    b_ix = torch.arange(B, device=x.device).view(B, 1, 1)
    h_ix = torch.arange(H, device=x.device).view(1, H, 1)
    q_sel = q[b_ix, h_ix, q_idx.unsqueeze(1)]                          # (B, H, m, D)
    a = (q_sel @ k.transpose(-2, -1)) * attn.scale
    a = a.softmax(dim=-1)
    out = (a @ v).transpose(1, 2).reshape(B, q_idx.shape[1], C)
    return attn.proj(out)


def block_flops_per_token(dim: int, n_tokens: int, mlp_ratio: float = 4.0) -> Dict[str, float]:
    """토큰당 블록 FLOPs 분해. q 투영+어텐션 행+proj+MLP 는 계산 토큰에만, k/v 투영은 전 토큰에 든다."""
    d = dim
    kv = 2 * 2 * d * d                       # k, v 투영
    q = 2 * d * d
    attn_row = 2 * 2 * n_tokens * d          # QK 행 + AV 행
    proj = 2 * d * d
    mlp = 2 * 2 * d * int(d * mlp_ratio)
    return {"kv_all_tokens": kv, "per_computed_token": q + attn_row + proj + mlp}


def analytic_flops(dim: int, n_tokens: int, depth: int, skip_frac, predictor_rank: int, patch_flops: float,
                   head_flops: float, mlp_ratio: float = 4.0) -> Dict[str, float]:
    parts = block_flops_per_token(dim, n_tokens, mlp_ratio)
    total_full = patch_flops + head_flops
    total_skip = patch_flops + head_flops
    for i in range(depth):
        s = skip_frac[i]
        m = n_tokens - int(round(s * (n_tokens - 1)))
        total_full += n_tokens * (parts["kv_all_tokens"] + parts["per_computed_token"])
        pred_cost = 2 * 2 * dim * predictor_rank * n_tokens if s > 0 else 0.0
        total_skip += n_tokens * parts["kv_all_tokens"] + m * parts["per_computed_token"] + pred_cost
    return {"full": total_full, "skip": total_skip, "ratio": total_skip / total_full}
