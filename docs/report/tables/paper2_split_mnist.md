| group | method | seeds | avg acc (final) | delta vs ref (pp) | forgetting | retention | ECE | train FLOPs |
|---|---|---|---|---|---|---|---|---|
| sleep decay schedule | CLS2 + sparse fast (k-WTA 10%) (reference) | 3 | 0.8703 +- 0.0059 | - | 0.1528 +- 0.0075 | 0.846 +- 0.008 | 0.080 +- 0.005 | 3.27e+12 |
| sleep decay schedule | CLS2 + sleep decay: one-shot 10% at boundary | 3 | 0.8735 +- 0.0015 | +0.32 | 0.1481 +- 0.0024 | 0.851 +- 0.002 | 0.077 +- 0.003 | 3.27e+12 |
| sleep decay schedule | CLS2 + sleep decay: periodic gradual, total 10% | 3 | 0.8729 +- 0.0020 | +0.27 | 0.1498 +- 0.0030 | 0.850 +- 0.003 | 0.072 +- 0.002 | 3.27e+12 |
| sleep decay schedule | CLS2 + sleep decay: periodic gradual, total 3% | 3 | 0.8750 +- 0.0087 | +0.47 | 0.1479 +- 0.0108 | 0.851 +- 0.011 | 0.074 +- 0.008 | 3.27e+12 |
| sleep decay schedule | CLS2 + sleep decay: continuous weight decay 1e-4 | 3 | 0.8710 +- 0.0022 | +0.08 | 0.1523 +- 0.0043 | 0.847 +- 0.004 | 0.078 +- 0.001 | 3.27e+12 |
| distillation locality (fast width 256) | CLS2 (fast width 256), no distillation (reference) | 3 | 0.8547 +- 0.0032 | - | 0.1718 +- 0.0041 | 0.827 +- 0.004 | 0.093 +- 0.002 | 1.31e+12 |
| distillation locality (fast width 256) | CLS2 (fast 256) + global logit distillation | 3 | 0.7928 +- 0.0044 | -6.19 | 0.2509 +- 0.0049 | 0.748 +- 0.005 | 0.145 +- 0.005 | 1.31e+12 |
| distillation locality (fast width 256) | CLS2 (fast 256) + local per-layer feature distillation | 3 | 0.8131 +- 0.0085 | -4.16 | 0.2209 +- 0.0108 | 0.778 +- 0.011 | 0.122 +- 0.012 | 1.31e+12 |
| distillation locality (fast width 256) | CLS2 (fast 256) + local features + logits | 3 | 0.7394 +- 0.0025 | -11.53 | 0.3062 +- 0.0066 | 0.691 +- 0.007 | 0.196 +- 0.013 | 1.31e+12 |

논문 2 토대. 수면 감쇠: 경계 1 회 하향(0.1) / 수면 중 50 스텝마다 점진(총 10% 또는 3%) / 수면 중 가중치 감쇠 1e-4 를 같은 설정의 CLS2(k-WTA 10%, 감쇠 없음)와 비교. 증류: 같은 폭(256)의 무증류 기준 대비 전역 로짓 증류 / 층별 국소 특징 증류 / 국소+로짓.
