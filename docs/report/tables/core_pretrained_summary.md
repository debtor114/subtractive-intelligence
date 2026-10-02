| density | arm | seeds | active conv weights | init acc | final acc | best acc | ECE | infer FLOPs | adaptation train FLOPs | samples to 90% |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pre-trained, dense fine-tune (no pruning) | 2 | 11,166,912 | 0.1085 +- 0.0170 | 0.9593 +- 0.0000 | 0.9598 +- 0.0000 | 0.018 +- 0.001 | 1.18e+09 | 1.78e+15 | 52515 +- 1566 |
| 0.05 | from scratch, dense small (width-scaled) | 2 | 562,759 | 0.1000 +- 0.0000 | 0.8484 +- 0.0004 | 0.8486 +- 0.0006 | 0.022 +- 0.001 | 7.19e+07 | 1.08e+14 | - |
| 0.05 | from scratch, prune-during (global magnitude) | 2 | 558,346 | 0.1000 +- 0.0000 | 0.8835 +- 0.0007 | 0.8838 +- 0.0010 | 0.031 +- 0.001 | 2.40e+08 | 7.72e+14 | - |
| 0.05 | pre-trained, one-shot prune then fine-tune | 2 | 558,346 | 0.1000 +- 0.0000 | 0.9306 +- 0.0012 | 0.9316 +- 0.0007 | 0.020 +- 0.001 | 2.01e+08 | 3.02e+14 | 165526 +- 4694 |
| 0.05 | pre-trained, random ERK mask + RigL regrow | 2 | 558,347 | 0.1000 +- 0.0000 | 0.8690 +- 0.0021 | 0.8692 +- 0.0020 | 0.023 +- 0.001 | 1.78e+08 | 2.68e+14 | - |
| 0.05 | pre-trained, magnitude mask + RigL regrow (keeps inherited structure) | 2 | 558,346 | 0.1000 +- 0.0000 | 0.9329 +- 0.0012 | 0.9336 +- 0.0015 | 0.018 +- 0.002 | 2.01e+08 | 3.02e+14 | 181334 +- 2491 |
| 0.05 | pre-trained, prune-during adaptation (ERK layer budget) | 2 | 558,347 | 0.1085 +- 0.0170 | 0.9343 +- 0.0005 | 0.9356 +- 0.0003 | 0.015 +- 0.001 | 1.78e+08 | 8.10e+14 | 52380 +- 1430 |
| 0.05 | pre-trained, prune-during adaptation (global magnitude) | 2 | 558,346 | 0.1085 +- 0.0170 | 0.9365 +- 0.0008 | 0.9377 +- 0.0005 | 0.017 +- 0.001 | 2.03e+08 | 7.26e+14 | 52276 +- 1327 |
| 0.02 | from scratch, dense small (width-scaled) | 2 | 226,242 | 0.0991 +- 0.0048 | 0.8240 +- 0.0006 | 0.8241 +- 0.0005 | 0.022 +- 0.001 | 3.29e+07 | 4.93e+13 | - |
| 0.02 | from scratch, prune-during (global magnitude) | 2 | 223,338 | 0.1000 +- 0.0000 | 0.8747 +- 0.0016 | 0.8752 +- 0.0011 | 0.031 +- 0.001 | 1.44e+08 | 6.94e+14 | - |
| 0.02 | pre-trained, one-shot prune then fine-tune | 2 | 223,338 | 0.1000 +- 0.0000 | 0.9029 +- 0.0010 | 0.9038 +- 0.0000 | 0.023 +- 0.000 | 1.23e+08 | 1.85e+14 | 401984 +- 12502 |
| 0.02 | pre-trained, random ERK mask + RigL regrow | 2 | 223,341 | 0.1000 +- 0.0000 | 0.8355 +- 0.0020 | 0.8355 +- 0.0019 | 0.025 +- 0.000 | 7.17e+07 | 1.08e+14 | - |
| 0.02 | pre-trained, magnitude mask + RigL regrow (keeps inherited structure) | 2 | 223,338 | 0.1000 +- 0.0000 | 0.9067 +- 0.0006 | 0.9073 +- 0.0010 | 0.025 +- 0.002 | 1.23e+08 | 1.85e+14 | 358505 +- 839 |
| 0.02 | pre-trained, prune-during adaptation (ERK layer budget) | 2 | 223,341 | 0.1085 +- 0.0170 | 0.8966 +- 0.0019 | 0.9292 +- 0.0008 | 0.020 +- 0.000 | 7.17e+07 | 7.10e+14 | 52567 +- 1617 |
| 0.02 | pre-trained, prune-during adaptation (global magnitude) | 2 | 223,338 | 0.1085 +- 0.0170 | 0.9098 +- 0.0008 | 0.9329 +- 0.0010 | 0.019 +- 0.001 | 1.25e+08 | 6.57e+14 | 52360 +- 1410 |
| 0.005 | from scratch, dense small (width-scaled) | 2 | 56,112 | 0.1000 +- 0.0000 | 0.7654 +- 0.0001 | 0.7664 +- 0.0011 | 0.022 +- 0.001 | 1.19e+07 | 1.78e+13 | - |
| 0.005 | from scratch, prune-during (global magnitude) | 2 | 55,835 | 0.1000 +- 0.0000 | 0.8380 +- 0.0037 | 0.8384 +- 0.0034 | 0.028 +- 0.002 | 5.39e+07 | 6.27e+14 | - |
| 0.005 | pre-trained, one-shot prune then fine-tune | 2 | 55,835 | 0.0884 +- 0.0116 | 0.8171 +- 0.0014 | 0.8184 +- 0.0019 | 0.019 +- 0.002 | 5.64e+07 | 8.45e+13 | - |
| 0.005 | pre-trained, random ERK mask + RigL regrow | 2 | 55,834 | 0.1000 +- 0.0000 | 0.7682 +- 0.0037 | 0.7690 +- 0.0044 | 0.023 +- 0.002 | 1.79e+07 | 2.69e+13 | - |
| 0.005 | pre-trained, magnitude mask + RigL regrow (keeps inherited structure) | 2 | 55,835 | 0.0882 +- 0.0118 | 0.8400 +- 0.0025 | 0.8416 +- 0.0030 | 0.024 +- 0.002 | 5.64e+07 | 8.45e+13 | - |
| 0.005 | pre-trained, prune-during adaptation (ERK layer budget) | 2 | 55,834 | 0.1085 +- 0.0170 | 0.7883 +- 0.0022 | 0.9314 +- 0.0006 | 0.020 +- 0.000 | 1.79e+07 | 6.59e+14 | 52422 +- 1473 |
| 0.005 | pre-trained, prune-during adaptation (global magnitude) | 2 | 55,835 | 0.1085 +- 0.0170 | 0.8087 +- 0.0012 | 0.9306 +- 0.0002 | 0.016 +- 0.000 | 5.68e+07 | 6.04e+14 | 52422 +- 1473 |
| 0.005 | pre-trained, prune-during adaptation, pruning ends at 50% of adaptation (global magnitude) | 2 | 55,835 | 0.1084 +- 0.0168 | 0.8365 +- 0.0013 | 0.9222 +- 0.0013 | 0.019 +- 0.002 | 5.64e+07 | 4.95e+14 | 43093 +- 5242 |

density 는 ImageNet 사전 학습 ResNet-18 의 conv 가중치(11,166,912) 대비 최종 활성 비율. 분류층(fc)은 모든 팔에서 밀집 유지. 입력은 128x128 로 키운 CIFAR-10, 10 에폭 적응. init acc 는 적응 전(새 분류층) 정확도. 적응 FLOPs 는 적응 10 에폭만 센 값이며 ImageNet 사전 학습 비용은 모든 pt_ 팔에 공통으로 들어 표에 넣지 않았다.
