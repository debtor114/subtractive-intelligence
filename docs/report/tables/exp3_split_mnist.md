| method | seeds | avg acc (final) | forgetting | retention | ECE (all classes) | train FLOPs | active params (predictor) | unknown-class AUROC (softmax) | unknown-class AUROC (fast/slow agreement) |
|---|---|---|---|---|---|---|---|---|---|
| fine-tune (no protection) | 3 | 0.1969 +- 0.0004 | 0.9950 +- 0.0001 | 0.000 +- 0.000 | 0.731 +- 0.005 | 2.90e+11 | 269,322 | 0.649 +- 0.022 | - |
| experience replay (buffer 500) | 3 | 0.8515 +- 0.0043 | 0.1747 +- 0.0062 | 0.824 +- 0.006 | 0.087 +- 0.006 | 5.82e+11 | 269,322 | 0.845 +- 0.007 | - |
| experience replay, 2x epochs | 3 | 0.8431 +- 0.0104 | 0.1886 +- 0.0132 | 0.811 +- 0.013 | 0.105 +- 0.008 | 1.16e+12 | 269,322 | 0.842 +- 0.021 | - |
| ER, class-balanced batches (buffer 500) | 3 | 0.8784 +- 0.0068 | 0.1302 +- 0.0087 | 0.868 +- 0.009 | 0.069 +- 0.005 | 2.87e+11 | 269,322 | 0.855 +- 0.017 | - |
| CLS + sleep (replay + distill) | 3 | 0.7871 +- 0.0164 | 0.2542 +- 0.0213 | 0.744 +- 0.021 | 0.123 +- 0.028 | 1.84e+12 | 269,322 | 0.774 +- 0.015 | 0.511 +- 0.012 |
| CLS + sleep (no distill) | 3 | 0.8497 +- 0.0059 | 0.1791 +- 0.0069 | 0.820 +- 0.007 | 0.090 +- 0.003 | 1.84e+12 | 269,322 | 0.834 +- 0.004 | 0.508 +- 0.002 |
| CLS + sleep + downscale 0.1 | 3 | 0.7744 +- 0.0211 | 0.2700 +- 0.0273 | 0.728 +- 0.028 | 0.133 +- 0.027 | 1.84e+12 | 269,322 | 0.776 +- 0.013 | 0.512 +- 0.012 |
| CLS + sleep + prune 5%/sleep (magnitude) | 3 | 0.7920 +- 0.0113 | 0.2485 +- 0.0158 | 0.750 +- 0.016 | 0.118 +- 0.025 | 1.84e+12 | 208,514 | 0.776 +- 0.017 | 0.511 +- 0.012 |
| CLS + sleep + prune 5%/sleep (activity) | 3 | 0.7803 +- 0.0156 | 0.2582 +- 0.0191 | 0.739 +- 0.020 | 0.107 +- 0.022 | 1.84e+12 | 208,514 | 0.744 +- 0.016 | 0.521 +- 0.012 |
| CLS + sleep + downscale + prune | 3 | 0.7875 +- 0.0198 | 0.2540 +- 0.0247 | 0.744 +- 0.025 | 0.121 +- 0.024 | 1.84e+12 | 208,514 | 0.771 +- 0.022 | 0.510 +- 0.013 |
| CLS2: balanced sleep replay, no distill, dense fast [predict: fast] | 3 | 0.1969 +- 0.0002 | 0.9964 +- 0.0004 | 0.000 +- 0.000 | - | - | - | - | - |
| CLS2: balanced sleep replay, no distill, dense fast [predict: max_conf] | 3 | 0.6794 +- 0.0256 | 0.3947 +- 0.0325 | 0.604 +- 0.032 | - | - | - | - | - |
| CLS2: balanced sleep replay, no distill, dense fast | 3 | 0.8703 +- 0.0059 | 0.1528 +- 0.0075 | 0.846 +- 0.008 | 0.080 +- 0.005 | 3.27e+12 | 269,322 | 0.857 +- 0.017 | 0.505 +- 0.001 |
| CLS2 + sparse fast (k-WTA 10%) [predict: fast] | 3 | 0.1969 +- 0.0002 | 0.9973 +- 0.0004 | 0.000 +- 0.000 | - | - | - | - | - |
| CLS2 + sparse fast (k-WTA 10%) [predict: max_conf] | 3 | 0.6820 +- 0.0116 | 0.3924 +- 0.0144 | 0.607 +- 0.015 | - | - | - | - | - |
| CLS2 + sparse fast (k-WTA 10%) | 3 | 0.8703 +- 0.0059 | 0.1528 +- 0.0075 | 0.846 +- 0.008 | 0.080 +- 0.005 | 3.27e+12 | 269,322 | 0.857 +- 0.017 | 0.511 +- 0.002 |
| CLS2 + sparse fast (k-WTA 5%) [predict: fast] | 3 | 0.1969 +- 0.0006 | 0.9965 +- 0.0009 | 0.000 +- 0.000 | - | - | - | - | - |
| CLS2 + sparse fast (k-WTA 5%) [predict: max_conf] | 3 | 0.6595 +- 0.0190 | 0.4206 +- 0.0230 | 0.578 +- 0.023 | - | - | - | - | - |
| CLS2 + sparse fast (k-WTA 5%) | 3 | 0.8703 +- 0.0059 | 0.1528 +- 0.0075 | 0.846 +- 0.008 | 0.080 +- 0.005 | 3.27e+12 | 269,322 | 0.857 +- 0.017 | 0.512 +- 0.004 |
| CLS2 + sparse fast 10% + magnitude prune 5%/sleep [predict: fast] | 3 | 0.1969 +- 0.0002 | 0.9973 +- 0.0004 | 0.000 +- 0.000 | - | - | - | - | - |
| CLS2 + sparse fast 10% + magnitude prune 5%/sleep [predict: max_conf] | 3 | 0.6728 +- 0.0133 | 0.4036 +- 0.0168 | 0.595 +- 0.017 | - | - | - | - | - |
| CLS2 + sparse fast 10% + magnitude prune 5%/sleep | 3 | 0.8689 +- 0.0030 | 0.1548 +- 0.0039 | 0.844 +- 0.004 | 0.079 +- 0.003 | 3.27e+12 | 208,514 | 0.857 +- 0.013 | 0.510 +- 0.001 |
| CLS2 + sparse fast 10%, sleep 300 steps lr 1e-3 [predict: fast] | 3 | 0.1968 +- 0.0004 | 0.9957 +- 0.0021 | 0.000 +- 0.000 | - | - | - | - | - |
| CLS2 + sparse fast 10%, sleep 300 steps lr 1e-3 [predict: max_conf] | 3 | 0.6570 +- 0.0569 | 0.4214 +- 0.0716 | 0.577 +- 0.072 | - | - | - | - | - |
| CLS2 + sparse fast 10%, sleep 300 steps lr 1e-3 | 3 | 0.8679 +- 0.0105 | 0.1510 +- 0.0145 | 0.848 +- 0.015 | 0.080 +- 0.007 | 2.25e+12 | 269,322 | 0.857 +- 0.006 | 0.512 +- 0.006 |
| CLS2 + sleep decay: one-shot 10% at boundary [predict: fast] | 3 | 0.1972 +- 0.0002 | 0.9965 +- 0.0006 | 0.000 +- 0.000 | - | - | - | - | - |
| CLS2 + sleep decay: one-shot 10% at boundary [predict: max_conf] | 3 | 0.6579 +- 0.0100 | 0.4225 +- 0.0129 | 0.576 +- 0.013 | - | - | - | - | - |
| CLS2 + sleep decay: one-shot 10% at boundary | 3 | 0.8735 +- 0.0015 | 0.1481 +- 0.0024 | 0.851 +- 0.002 | 0.077 +- 0.003 | 3.27e+12 | 269,322 | 0.851 +- 0.010 | 0.508 +- 0.003 |
| CLS2 + sleep decay: periodic gradual, total 10% [predict: fast] | 3 | 0.1972 +- 0.0002 | 0.9965 +- 0.0006 | 0.000 +- 0.000 | - | - | - | - | - |
| CLS2 + sleep decay: periodic gradual, total 10% [predict: max_conf] | 3 | 0.6313 +- 0.0025 | 0.4560 +- 0.0030 | 0.543 +- 0.003 | - | - | - | - | - |
| CLS2 + sleep decay: periodic gradual, total 10% | 3 | 0.8729 +- 0.0020 | 0.1498 +- 0.0030 | 0.850 +- 0.003 | 0.072 +- 0.002 | 3.27e+12 | 269,322 | 0.855 +- 0.013 | 0.507 +- 0.002 |
| CLS2 + sleep decay: periodic gradual, total 3% [predict: fast] | 3 | 0.1972 +- 0.0002 | 0.9965 +- 0.0006 | 0.000 +- 0.000 | - | - | - | - | - |
| CLS2 + sleep decay: periodic gradual, total 3% [predict: max_conf] | 3 | 0.6574 +- 0.0137 | 0.4233 +- 0.0165 | 0.576 +- 0.017 | - | - | - | - | - |
| CLS2 + sleep decay: periodic gradual, total 3% | 3 | 0.8750 +- 0.0087 | 0.1479 +- 0.0108 | 0.851 +- 0.011 | 0.074 +- 0.008 | 3.27e+12 | 269,322 | 0.860 +- 0.009 | 0.509 +- 0.001 |
| CLS2 + sleep decay: continuous weight decay 1e-4 [predict: fast] | 3 | 0.1972 +- 0.0002 | 0.9965 +- 0.0006 | 0.000 +- 0.000 | - | - | - | - | - |
| CLS2 + sleep decay: continuous weight decay 1e-4 [predict: max_conf] | 3 | 0.6510 +- 0.0071 | 0.4315 +- 0.0086 | 0.568 +- 0.009 | - | - | - | - | - |
| CLS2 + sleep decay: continuous weight decay 1e-4 | 3 | 0.8710 +- 0.0022 | 0.1523 +- 0.0043 | 0.847 +- 0.004 | 0.078 +- 0.001 | 3.27e+12 | 269,322 | 0.851 +- 0.012 | 0.507 +- 0.002 |
| CLS2 (fast width 256), no distillation [predict: fast] | 3 | 0.1965 +- 0.0005 | 0.9972 +- 0.0004 | 0.000 +- 0.000 | - | - | - | - | - |
| CLS2 (fast width 256), no distillation [predict: max_conf] | 3 | 0.6757 +- 0.0300 | 0.3995 +- 0.0374 | 0.599 +- 0.038 | - | - | - | - | - |
| CLS2 (fast width 256), no distillation | 3 | 0.8547 +- 0.0032 | 0.1718 +- 0.0041 | 0.827 +- 0.004 | 0.093 +- 0.002 | 1.31e+12 | 269,322 | 0.864 +- 0.007 | 0.510 +- 0.003 |
| CLS2 (fast 256) + global logit distillation [predict: fast] | 3 | 0.1966 +- 0.0005 | 0.9963 +- 0.0003 | 0.000 +- 0.000 | - | - | - | - | - |
| CLS2 (fast 256) + global logit distillation [predict: max_conf] | 3 | 0.6007 +- 0.0226 | 0.4918 +- 0.0277 | 0.506 +- 0.028 | - | - | - | - | - |
| CLS2 (fast 256) + global logit distillation | 3 | 0.7928 +- 0.0044 | 0.2509 +- 0.0049 | 0.748 +- 0.005 | 0.145 +- 0.005 | 1.31e+12 | 269,322 | 0.811 +- 0.009 | 0.519 +- 0.003 |
| CLS2 (fast 256) + local per-layer feature distillation [predict: fast] | 3 | 0.1966 +- 0.0005 | 0.9963 +- 0.0003 | 0.000 +- 0.000 | - | - | - | - | - |
| CLS2 (fast 256) + local per-layer feature distillation [predict: max_conf] | 3 | 0.6368 +- 0.0127 | 0.4468 +- 0.0151 | 0.552 +- 0.015 | - | - | - | - | - |
| CLS2 (fast 256) + local per-layer feature distillation | 3 | 0.8131 +- 0.0085 | 0.2209 +- 0.0108 | 0.778 +- 0.011 | 0.122 +- 0.012 | 1.31e+12 | 269,322 | 0.874 +- 0.007 | 0.508 +- 0.002 |
| CLS2 (fast 256) + local features + logits [predict: fast] | 3 | 0.1966 +- 0.0005 | 0.9963 +- 0.0003 | 0.000 +- 0.000 | - | - | - | - | - |
| CLS2 (fast 256) + local features + logits [predict: max_conf] | 3 | 0.5883 +- 0.0223 | 0.5056 +- 0.0276 | 0.492 +- 0.028 | - | - | - | - | - |
| CLS2 (fast 256) + local features + logits | 3 | 0.7394 +- 0.0025 | 0.3062 +- 0.0066 | 0.691 +- 0.007 | 0.196 +- 0.013 | 1.31e+12 | 269,322 | 0.813 +- 0.005 | 0.518 +- 0.002 |
| joint retraining (upper bound) | 3 | 0.9762 +- 0.0009 | 0.0086 +- 0.0031 | 0.991 +- 0.003 | 0.005 +- 0.002 | 8.79e+11 | 269,322 | 0.934 +- 0.005 | - |

Split MNIST 는 Class-IL (단일 헤드 10 출력, 5 태스크 x 2 클래스), Permuted 는 10 태스크. unknown-class AUROC: 각 시점에서 아직 안 배운 클래스의 테스트 표본을 확신도로 가려내는 성능 (0.5 = 못 가림).
