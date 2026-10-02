| density | arm | seeds | active weights (per sample) | test acc | ECE | infer FLOPs | cum. train FLOPs | stage-4 Jaccard same class | Jaccard diff class | union coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | static: dense small (additive) | 2 | 2,239,484 | 0.9093 +- 0.0000 | 0.035 +- 0.002 | 2.25e+08 | 6.74e+14 | - | - | - |
| 0.2 | static: prune-during, magnitude (global) | 2 | 2,232,870 | 0.9189 +- 0.0013 | 0.038 +- 0.000 | 3.45e+08 | 1.65e+15 | - | - | - |
| 0.2 | dynamic, channel-level k-WTA input gating | 2 | 2,233,536 | 0.7339 +- 0.0072 | 0.021 +- 0.000 | 2.27e+08 | 1.36e+15 | 0.192 +- 0.010 | 0.127 +- 0.002 | 1.000 +- 0.000 |
| 0.2 | dynamic, connection-level, random selection | 2 | 2,238,912 | 0.7614 +- 0.0023 | 0.024 +- 0.003 | 2.25e+08 | 1.36e+15 | 0.111 +- 0.000 | 0.111 +- 0.000 | 1.000 +- 0.000 |
| 0.2 | dynamic, connection-level, local rule (abs(w) x input magnitude) | 2 | 2,238,912 | 0.9183 +- 0.0010 | 0.038 +- 0.000 | 2.25e+08 | 1.36e+15 | 0.613 +- 0.005 | 0.462 +- 0.008 | 0.429 +- 0.003 |
| 0.05 | static: dense small (additive) | 3 | 562,229 | 0.8866 +- 0.0018 | 0.030 +- 0.002 | 5.58e+07 | 1.67e+14 | - | - | - |
| 0.05 | static: train-then-prune + finetune | 3 | 558,218 | 0.9227 +- 0.0006 | 0.035 +- 0.001 | 2.71e+08 | 3.74e+15 | - | - | - |
| 0.05 | static: RigL | 3 | 558,213 | 0.9097 +- 0.0016 | 0.034 +- 0.001 | 1.39e+08 | 4.17e+14 | - | - | - |
| 0.05 | static: prune-during, magnitude (ERK) | 3 | 558,213 | 0.9203 +- 0.0009 | 0.034 +- 0.001 | 1.39e+08 | 1.43e+15 | - | - | - |
| 0.05 | static: prune-during, magnitude (global) | 3 | 558,218 | 0.9196 +- 0.0023 | 0.034 +- 0.001 | 1.85e+08 | 1.33e+15 | - | - | - |
| 0.05 | dynamic, channel-level k-WTA input gating | 2 | 569,536 | 0.4486 +- 0.0036 | 0.016 +- 0.003 | 5.74e+07 | 9.83e+14 | 0.083 +- 0.002 | 0.052 +- 0.001 | 0.916 +- 0.008 |
| 0.05 | dynamic, connection-level, random selection | 2 | 564,160 | 0.5229 +- 0.0005 | 0.029 +- 0.001 | 5.91e+07 | 9.86e+14 | 0.026 +- 0.000 | 0.026 +- 0.000 | 1.000 +- 0.000 |
| 0.05 | dynamic, connection-level, local rule (abs(w) x input magnitude) | 2 | 564,160 | 0.8983 +- 0.0011 | 0.039 +- 0.003 | 5.91e+07 | 9.86e+14 | 0.439 +- 0.003 | 0.274 +- 0.002 | 0.207 +- 0.003 |
| 0.005 | static: dense small (additive) | 3 | 55,872 | 0.8131 +- 0.0026 | 0.021 +- 0.001 | 6.12e+06 | 1.84e+13 | - | - | - |
| 0.005 | static: train-then-prune + finetune | 3 | 55,822 | 0.8364 +- 0.0015 | 0.016 +- 0.001 | 5.27e+07 | 3.41e+15 | - | - | - |
| 0.005 | static: RigL | 3 | 55,820 | 0.8486 +- 0.0030 | 0.026 +- 0.001 | 1.38e+07 | 4.15e+13 | - | - | - |
| 0.005 | static: prune-during, magnitude (ERK) | 3 | 55,820 | 0.8783 +- 0.0022 | 0.024 +- 0.001 | 1.38e+07 | 1.18e+15 | - | - | - |
| 0.005 | static: prune-during, magnitude (global) | 3 | 55,822 | 0.8896 +- 0.0003 | 0.027 +- 0.001 | 3.40e+07 | 1.09e+15 | - | - | - |
| 0.005 | dynamic, channel-level k-WTA input gating | 2 | 69,952 | 0.3404 +- 0.0296 | 0.021 +- 0.001 | 1.34e+07 | 8.80e+14 | 0.058 +- 0.007 | 0.036 +- 0.002 | 0.285 +- 0.021 |
| 0.005 | dynamic, connection-level, random selection | 2 | 63,424 | 0.1824 +- 0.0026 | 0.040 +- 0.002 | 9.32e+06 | 8.75e+14 | 0.002 +- 0.000 | 0.003 +- 0.000 | 0.633 +- 0.000 |
| 0.005 | dynamic, connection-level, local rule (abs(w) x input magnitude) | 2 | 63,424 | 0.7768 +- 0.0015 | 0.019 +- 0.001 | 9.32e+06 | 8.75e+14 | 0.333 +- 0.009 | 0.246 +- 0.006 | 0.033 +- 0.001 |

CIFAR-10 ResNet-18. 동적 팔은 표본마다 다른 서브망을 쓰며 표본당 기대 활성 연결이 density 에 맞춰짐 (첫 conv, fc 는 밀집). 정적 팔은 results/core_resnet 의 같은 밀도 결과. 자카드는 4 단계 블록 둘째 conv 의 표본별 마스크 유사도 (같은 클래스 쌍 / 다른 클래스 쌍, 무작위면 둘 다 약 density). union coverage 는 테스트 표본 200 개 중 하나라도 쓴 연결 비율.
