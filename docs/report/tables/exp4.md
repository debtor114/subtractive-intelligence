| n_e | seeds | STDP acc | backprop MLP acc (same width, same samples) | SNN infer SOP/img | MLP infer FLOPs/img | SNN train ops (total) | MLP train FLOPs (total) | SNN samples to 80% | MLP samples to 80% | weights < 1% wmax (final) | train time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 1 | 0.3697 | 0.9474 | 1.08e+05 | 1.59e+05 | 1.15e+10 | 2.86e+10 | - | 4992 | 0.549 | 550 |
| 400 | 3 | 0.8108 +- 0.0028 | 0.9649 +- 0.0021 | 4.33e+05 | 6.35e+05 | 4.60e+10 | 1.14e+11 | 38189 +- 492 | 4992 +- 0 | 0.577 +- 0.000 | 561 +- 9 |
| 1600 | 1 | 0.7289 | 0.9682 | 1.74e+06 | 2.54e+06 | 1.83e+11 | 4.57e+11 | - | 4992 | 0.287 | 576 |
| control: MNIST class-mean background pixels | - | - | - | - | - | - | - | - | - | 0.498 (per class 0.42-0.70; zero in 99% of images 0.371) | - |

SOP = 사건 구동 시냅스 연산 (누산). 에너지 비교는 하지 않는다 (GPU 에서 실행). backprop MLP 는 784-n_e-10, Adam, 같은 표본을 한 번 훑음. 배경 픽셀 대조 (프로토타입 형성만으로 설명되는 희소화의 상한): MNIST 클래스 평균 이미지에서 최대값의 1% 미만인 픽셀 비율 0.498 (클래스별 0.422~0.698), 전체 평균 이미지 0.380, 학습 이미지의 99% 이상에서 0 인 픽셀 0.371.
