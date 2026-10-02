# -*- coding: utf-8 -*-
"""뇌 원리 모듈 (2 단계 이후 채운다).

예정 파일:
- masked_layers.py    : weight_mask 버퍼를 가진 Linear / Conv (가지치기 대상). utils.metrics 가 이 버퍼를 읽는다.
- pruning.py          : 학습 중 가지치기 규칙. 크기 기반, 활동 의존(헤비안/STDP 형), 스케줄러
- inhibitory.py       : 측면 억제 / k-WTA 희소 활성화
- thalamic_router.py  : 어텐션 점수 계산 전에 후보를 줄이는 사전 라우팅 (실험 1)
- predictive_coding.py: 층간 예측 오차 기반 토큰 스킵 (실험 2)
- hippocampus.py      : 빠른 기억 + 리플레이 + 수면 단계 (실험 3)
- metacognition.py    : 확신도 산출 및 재처리 트리거
"""
