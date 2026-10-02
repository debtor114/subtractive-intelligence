| group | method | seeds | avg acc (final) | delta vs ref (pp) | forgetting | retention | ECE | train FLOPs |
|---|---|---|---|---|---|---|---|---|
| sleep decay variants | CLS2 + sparse fast (k-WTA 10%) (reference) | 3 | 0.8903 +- 0.0020 | - | 0.0662 +- 0.0024 | 0.931 +- 0.003 | 0.109 +- 0.022 | 1.18e+13 |
| sleep decay variants | CLS2 + sleep decay: one-shot 10% at boundary | 3 | 0.8503 +- 0.0042 | -4.00 | 0.1125 +- 0.0042 | 0.882 +- 0.004 | 0.136 +- 0.004 | 1.18e+13 |
| sleep decay variants | CLS2 + sleep decay: periodic gradual, total 10% | 3 | 0.8550 +- 0.0035 | -3.53 | 0.1075 +- 0.0028 | 0.888 +- 0.003 | 0.118 +- 0.008 | 1.18e+13 |
| sleep decay variants | CLS2 + sleep decay: periodic gradual, total 3% | 3 | 0.8873 +- 0.0047 | -0.30 | 0.0718 +- 0.0047 | 0.925 +- 0.005 | 0.098 +- 0.011 | 1.18e+13 |
| sleep decay variants | CLS2 + sleep decay: continuous weight decay 1e-4 | 3 | 0.8850 +- 0.0058 | -0.53 | 0.0743 +- 0.0056 | 0.922 +- 0.006 | 0.094 +- 0.009 | 1.18e+13 |
| distillation locality (fast width 256) | CLS2 (fast width 256), no distillation (reference) | 3 | 0.8939 +- 0.0023 | - | 0.0636 +- 0.0019 | 0.934 +- 0.002 | 0.107 +- 0.010 | 4.75e+12 |
| distillation locality (fast width 256) | CLS2 (fast 256) + global logit distillation | 3 | 0.8782 +- 0.0064 | -1.58 | 0.0941 +- 0.0070 | 0.902 +- 0.007 | 0.120 +- 0.011 | 4.75e+12 |
| distillation locality (fast width 256) | CLS2 (fast 256) + local per-layer feature distillation | 3 | 0.8516 +- 0.0047 | -4.23 | 0.1067 +- 0.0049 | 0.888 +- 0.005 | 0.178 +- 0.021 | 4.75e+12 |
| distillation locality (fast width 256) | CLS2 (fast 256) + local features + logits | 3 | 0.8355 +- 0.0079 | -5.85 | 0.1266 +- 0.0076 | 0.867 +- 0.008 | 0.172 +- 0.017 | 4.75e+12 |

논문 2 토대. 수면 감쇠: 경계 1 회 하향(0.1) / 수면 중 50 스텝마다 점진(총 10% 또는 3%) / 수면 중 가중치 감쇠 1e-4 를 같은 설정의 CLS2(k-WTA 10%, 감쇠 없음)와 비교. 증류: 같은 폭(256)의 무증류 기준 대비 전역 로짓 증류 / 층별 국소 특징 증류 / 국소+로짓.
