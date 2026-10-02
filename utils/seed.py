# -*- coding: utf-8 -*-
"""재현성: 시드 고정.

논문 실험은 시드 3~5개 평균으로 보고한다. deterministic=True 는 cuDNN 자동 튜닝을 끄므로
속도가 떨어진다. 기본은 False (benchmark 모드) 로 두고, 재현 검증 때만 켠다.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
