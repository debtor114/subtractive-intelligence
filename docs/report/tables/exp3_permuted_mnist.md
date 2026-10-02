| method | seeds | avg acc (final) | forgetting | retention | ECE (all classes) | train FLOPs | active params (predictor) | unknown-class AUROC (softmax) | unknown-class AUROC (fast/slow agreement) |
|---|---|---|---|---|---|---|---|---|---|
| fine-tune (no protection) | 3 | 0.4894 +- 0.0151 | 0.5384 +- 0.0166 | 0.447 +- 0.017 | 0.502 +- 0.043 | 2.90e+12 | 269,322 | - | - |
| experience replay (buffer 500) | 3 | 0.8739 +- 0.0103 | 0.1121 +- 0.0115 | 0.885 +- 0.012 | 0.187 +- 0.042 | 5.81e+12 | 269,322 | - | - |
| experience replay, 2x epochs | 3 | 0.8271 +- 0.0142 | 0.1646 +- 0.0157 | 0.831 +- 0.016 | 0.214 +- 0.017 | 1.16e+13 | 269,322 | - | - |
| ER, class-balanced batches (buffer 500) | 3 | 0.8828 +- 0.0037 | 0.0764 +- 0.0050 | 0.921 +- 0.005 | 0.140 +- 0.012 | 2.36e+12 | 269,322 | - | - |
| CLS + sleep (replay + distill) | 3 | 0.8780 +- 0.0038 | 0.0939 +- 0.0047 | 0.902 +- 0.005 | 0.110 +- 0.020 | 6.00e+12 | 269,322 | - | - |
| CLS + sleep (no distill) | 3 | 0.8856 +- 0.0006 | 0.0847 +- 0.0002 | 0.912 +- 0.000 | 0.125 +- 0.005 | 6.00e+12 | 269,322 | - | - |
| CLS + sleep + downscale 0.1 | 3 | 0.8269 +- 0.0064 | 0.1512 +- 0.0070 | 0.843 +- 0.007 | 0.151 +- 0.022 | 6.00e+12 | 269,322 | - | - |
| CLS + sleep + prune 5%/sleep (magnitude) | 3 | 0.8731 +- 0.0040 | 0.0973 +- 0.0042 | 0.899 +- 0.004 | 0.130 +- 0.022 | 6.00e+12 | 161,462 | - | - |
| CLS + sleep + prune 5%/sleep (activity) | 3 | 0.8148 +- 0.0026 | 0.1599 +- 0.0029 | 0.833 +- 0.003 | 0.195 +- 0.029 | 6.00e+12 | 161,462 | - | - |
| CLS + sleep + downscale + prune | 3 | 0.8366 +- 0.0076 | 0.1382 +- 0.0088 | 0.856 +- 0.009 | 0.120 +- 0.029 | 6.00e+12 | 161,462 | - | - |
| CLS2: balanced sleep replay, no distill, dense fast [predict: fast] | 3 | 0.4026 +- 0.0574 | 0.6295 +- 0.0630 | 0.351 +- 0.065 | - | - | - | - | - |
| CLS2: balanced sleep replay, no distill, dense fast [predict: max_conf] | 3 | 0.8577 +- 0.0064 | 0.1256 +- 0.0070 | 0.871 +- 0.007 | - | - | - | - | - |
| CLS2: balanced sleep replay, no distill, dense fast | 3 | 0.8903 +- 0.0020 | 0.0662 +- 0.0024 | 0.931 +- 0.003 | 0.109 +- 0.022 | 1.18e+13 | 269,322 | - | - |
| CLS2 + sparse fast (k-WTA 10%) [predict: fast] | 3 | 0.4661 +- 0.0150 | 0.5601 +- 0.0169 | 0.423 +- 0.017 | - | - | - | - | - |
| CLS2 + sparse fast (k-WTA 10%) [predict: max_conf] | 3 | 0.8708 +- 0.0145 | 0.1119 +- 0.0161 | 0.885 +- 0.017 | - | - | - | - | - |
| CLS2 + sparse fast (k-WTA 10%) | 3 | 0.8903 +- 0.0020 | 0.0662 +- 0.0024 | 0.931 +- 0.003 | 0.109 +- 0.022 | 1.18e+13 | 269,322 | - | - |
| CLS2 + sparse fast 10%, sleep 300 steps lr 1e-3 [predict: fast] | 3 | 0.4080 +- 0.0138 | 0.6248 +- 0.0162 | 0.357 +- 0.016 | - | - | - | - | - |
| CLS2 + sparse fast 10%, sleep 300 steps lr 1e-3 [predict: max_conf] | 3 | 0.8503 +- 0.0029 | 0.1330 +- 0.0034 | 0.863 +- 0.003 | - | - | - | - | - |
| CLS2 + sparse fast 10%, sleep 300 steps lr 1e-3 | 3 | 0.8851 +- 0.0027 | 0.0617 +- 0.0043 | 0.935 +- 0.004 | 0.097 +- 0.022 | 9.97e+12 | 269,322 | - | - |
| joint retraining (upper bound) | 3 | 0.9633 +- 0.0061 | 0.0034 +- 0.0005 | 0.998 +- 0.001 | 0.007 +- 0.002 | 8.87e+12 | 269,322 | - | - |

Split MNIST 는 Class-IL (단일 헤드 10 출력, 5 태스크 x 2 클래스), Permuted 는 10 태스크. unknown-class AUROC: 각 시점에서 아직 안 배운 클래스의 테스트 표본을 확신도로 가려내는 성능 (0.5 = 못 가림).
