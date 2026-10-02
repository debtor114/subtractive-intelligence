# -*- coding: utf-8 -*-
"""입력 의존적 (추론 중) 가지치기 층. 로드맵 후보 1 (docs/notes/roadmap_candidates_1001.md) 의 최소 실험용.

같은 망이 입력마다 다른 서브망을 쓴다. 세 가지 동적 모드와 밀집 모드:
  dense      : 보통 conv (밀도 1.0 이면 어느 모드든 이 경로).
  dyn_local  : 연결 단위, 국소 규칙. 출력 뉴런 o 마다 점수 |w[o,j]| * a[b,c(j)] (a = 입력 채널의 공간 평균 |x|) 상위 k 연결만 통과.
               표본 b 마다 다른 가중치 마스크 -> unfold + bmm 으로 표본별 가중치 곱 (메모리 B x params).
  dyn_random : 연결 단위, 표본마다 출력 뉴런별 무작위 k 연결 (선택 기준의 대조군).
  kwta_in    : 뉴런(채널) 단위. 입력 채널 중 a[b,c] 상위 비율만 통과 (구조적 희소 - GPU 에서 실제 속도가 나는 유일한 형태).
density 는 스케줄러가 바꾼다 (cubic, 학습 중 가지치기와 같은 일정). FLOPs 비율은 fraction() 이 돌려준다:
  dyn_*: k/J (연결 비율), kwta_in: k/C_in (입력 채널 비율). 두 경우 모두 이 conv 의 FLOPs = dense x fraction.

점수는 detach 하므로 마스크는 미분되지 않고, 경사는 표본별로 남은 연결의 가중치에만 흐른다 (DropConnect 와 같은 꼴).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

DYN_MODES = ("dense", "dyn_local", "dyn_random", "kwta_in")


class DynamicConv2d(nn.Conv2d):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.mode: str = "dense"
        self.density: float = 1.0
        self.capture: bool = False          # True 면 마지막 forward 의 마스크를 captured 에 남긴다 (분석용)
        self.captured: Optional[torch.Tensor] = None

    # --- 밀도 -> 실제 통과 비율 -------------------------------------------------
    def n_conn(self) -> int:
        return self.in_channels // self.groups * self.kernel_size[0] * self.kernel_size[1]

    def k_conn(self) -> int:
        return max(1, int(round(self.density * self.n_conn())))

    def k_chan(self) -> int:
        return max(1, int(round(self.density * self.in_channels)))

    def fraction(self) -> float:
        """이 층 FLOPs 의 dense 대비 비율."""
        if self.mode == "dense" or self.density >= 1.0:
            return 1.0
        if self.mode == "kwta_in":
            return self.k_chan() / self.in_channels
        return self.k_conn() / self.n_conn()

    def expected_active(self) -> float:
        """표본당 기대 활성 연결 수."""
        return self.weight.numel() * self.fraction()

    # --- forward ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "dense" or self.density >= 1.0:
            if self.capture:
                self.captured = None
            return super().forward(x)
        B, C_in, H, W = x.shape
        if self.mode == "kwta_in":
            a = x.detach().abs().mean(dim=(2, 3))                                   # (B, C_in)
            idx = a.topk(self.k_chan(), dim=1).indices
            g = torch.zeros_like(a).scatter_(1, idx, 1.0)
            if self.capture:
                self.captured = g.detach()
            return super().forward(x * g.view(B, C_in, 1, 1).to(x.dtype))
        # 연결 단위: 표본별 가중치 (B x params) 를 만들므로 학습 중에는 체크포인트로 감싸 역전파 때 다시 계산한다.
        # 안 그러면 층마다 마스크와 표본별 가중치가 역전파용으로 남아 ResNet-18 B=128 에서 26GB 를 먹었다 (2026-10-01 팟).
        # 비재진입 체크포인트는 RNG 상태를 보존하므로 dyn_random 의 무작위 마스크도 재계산 때 같다.
        # 국소 규칙의 임계값 (뉴런별 k 번째 점수) 은 체크포인트 밖에서 한 번만 구한다. kthvalue 가 가장 비싼 연산이라
        # 재계산까지 두 번 돌리면 스텝 시간이 거의 두 배였다 (팟 측정 1.5 s -> 목표 0.8 s).
        thr = self._local_threshold(x) if self.mode == "dyn_local" else None
        if torch.is_grad_enabled() and self.training:
            return torch.utils.checkpoint.checkpoint(self._dyn_forward, x, thr, use_reentrant=False)
        return self._dyn_forward(x, thr)

    def _local_score(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in = x.shape[0], x.shape[1]
        Wt = self.weight
        C_out, J = Wt.shape[0], self.n_conn()
        a = x.detach().abs().mean(dim=(2, 3)).float()                               # (B, C_in)
        return (Wt.detach().abs().float().view(1, C_out, C_in, -1) * a.view(B, 1, C_in, 1)).view(B, C_out, J)

    @torch.no_grad()
    def _local_threshold(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        J, k = self.n_conn(), self.k_conn()
        if k >= J:
            return None
        score = self._local_score(x)
        return score.kthvalue(J - k + 1, dim=-1, keepdim=True).values                # (B, C_out, 1) 출력 뉴런별 k 번째로 큰 점수

    def _dyn_forward(self, x: torch.Tensor, thr: Optional[torch.Tensor]) -> torch.Tensor:
        B, C_in, H, W = x.shape
        Wt = self.weight
        C_out, J = Wt.shape[0], self.n_conn()
        k = self.k_conn()
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else Wt.dtype
        # 마스크는 임계값 비교로 만든다. topk 는 k 개의 int64 인덱스 (B x C_out x k x 8 바이트) 를 만들어 밀도가 1 근처인
        # 스케줄 초반에 층 하나가 2~5GB 를 먹었다 (2026-10-01 팟, 평가 배치 250 에서 4.35GiB 할당 실패). kthvalue 는 (B, C_out) 만 돌려준다.
        if self.mode == "dyn_local":
            if thr is None:
                M = torch.ones(B, C_out, J, device=x.device, dtype=dt)
            else:
                score = self._local_score(x)
                M = ((score >= thr) & (score > 0)).to(dt)                            # 점수 0 (죽은 입력 채널) 은 뽑지 않는다
                del score
        elif self.mode == "dyn_random":
            # 뉴런별 정확히 k 개 대신 같은 기대값의 베르누이 표본 (topk 없이 싸다). 기대 활성 비율은 k/J 로 같다.
            M = (torch.rand(B, C_out, J, device=x.device) < (k / J)).to(dt)
        else:
            raise KeyError(self.mode)
        if self.capture:
            self.captured = M.detach()
        Wm = Wt.to(dt).view(1, C_out, J) * M                                        # (B, C_out, J)
        cols = F.unfold(x.to(dt), self.kernel_size, self.dilation, self.padding, self.stride)   # (B, J, L)
        y = torch.bmm(Wm, cols)                                                     # (B, C_out, L)
        H_out = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        y = y.view(B, C_out, H_out, W_out)
        if self.bias is not None:
            y = y + self.bias.to(dt).view(1, -1, 1, 1)
        return y


def _replace(module: nn.Module, name: str, new: nn.Module) -> None:
    parts = name.split(".")
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def convert_to_dynamic(model: nn.Module, mode: str, skip: Iterable[str] = ("conv1", "fc")) -> nn.Module:
    """model 의 nn.Conv2d (skip 에 정확히 일치하는 이름 제외) 를 DynamicConv2d 로 바꾼다. 가중치는 그대로 복사."""
    assert mode in DYN_MODES, mode
    skip = set(skip)
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d) and not isinstance(m, DynamicConv2d)
               and n not in skip]
    for n, m in targets:
        new = DynamicConv2d(m.in_channels, m.out_channels, m.kernel_size, stride=m.stride, padding=m.padding,
                            dilation=m.dilation, groups=m.groups, bias=m.bias is not None, padding_mode=m.padding_mode)
        new.weight.data.copy_(m.weight.data)
        if m.bias is not None:
            new.bias.data.copy_(m.bias.data)
        new.mode = mode
        _replace(model, n, new)
    return model


def dynamic_modules(model: nn.Module) -> List[Tuple[str, DynamicConv2d]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, DynamicConv2d)]


def set_dynamic_density(model: nn.Module, density: float) -> None:
    for _, m in dynamic_modules(model):
        m.density = float(density)


@torch.no_grad()
def layer_fractions(model: nn.Module) -> Dict[str, float]:
    """리프별 FLOPs 비율 (count_flops 의 leaf_dense 와 곱해 쓴다). 동적 층은 fraction(), 나머지는 1."""
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, DynamicConv2d):
            out[name] = m.fraction()
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            out[name] = 1.0
    return out


@torch.no_grad()
def expected_active_weights(model: nn.Module) -> float:
    tot = 0.0
    for m in model.modules():
        if isinstance(m, DynamicConv2d):
            tot += m.expected_active()
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            tot += m.weight.numel()
    return tot


@torch.no_grad()
def mask_statistics(model: nn.Module, layer_names: Iterable[str], x: torch.Tensor, y: torch.Tensor) -> Dict[str, Dict]:
    """표본 x (정규화된 입력) 에 대해 각 층의 표본별 마스크를 잡아 클래스 안/밖 자카드 유사도와 합집합 커버리지를 구한다.
    해석: J_same >> J_diff 이면 입력(클래스) 마다 다른 서브망이 생긴 것. 무작위 마스크면 둘 다 약 밀도 값.
    coverage: 표본 중 하나라도 쓴 연결(채널)의 비율 = 동적 망이 실제로 쓰는 총 용량."""
    mods = dict(model.named_modules())
    targets = [n for n in layer_names if n in mods and isinstance(mods[n], DynamicConv2d)]
    for n in targets:
        mods[n].capture = True
    was = model.training
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        model(x)
    model.train(was)
    out = {}
    same = (y.view(-1, 1) == y.view(1, -1))
    eye = torch.eye(y.numel(), dtype=torch.bool, device=y.device)
    for n in targets:
        m = mods[n]
        m.capture = False
        M = m.captured
        if M is None:
            continue
        B = M.shape[0]
        Mf = M.reshape(B, -1).to(torch.float16)
        sizes = Mf.float().sum(1)                                   # (B,)
        # 교집합 크기는 fp32 로 조각내어 누적 (fp16 내적은 65504 를 넘어 inf 가 됨 - 연기 테스트에서 확인)
        inter = torch.zeros(B, B, device=Mf.device)
        for s0 in range(0, Mf.shape[1], 32768):
            c = Mf[:, s0:s0 + 32768].float()
            inter += c @ c.t()
        union = sizes.view(-1, 1) + sizes.view(1, -1) - inter
        jac = inter / union.clamp_min(1.0)
        j_same = jac[same & ~eye].mean().item() if (same & ~eye).any() else float("nan")
        j_diff = jac[~same].mean().item() if (~same).any() else float("nan")
        coverage = (Mf.float().sum(0) > 0).float().mean().item()
        out[n] = {"unit": "channel" if m.mode == "kwta_in" else "connection", "n_units": int(Mf.shape[1]),
                  "per_sample_fraction": float(sizes.mean().item() / Mf.shape[1]), "union_coverage": coverage,
                  "jaccard_same_class": j_same, "jaccard_diff_class": j_diff, "n_samples": int(B)}
        m.captured = None
    return out
