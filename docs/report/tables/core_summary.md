| density | arm | seeds | active weights | test acc | ECE | infer FLOPs | cum. train FLOPs | samples to target (97% MNIST / 85% CIFAR) |
|---|---|---|---|---|---|---|---|---|
| 1 | dense big (no pruning) | 3 | 1,861,632 | 0.9871 +- 0.0003 | 0.009 +- 0.000 | 3.72e+06 | 1.01e+13 | 87336 +- 7781 |
| 0.1 | dense small (additive) | 3 | 185,787 | 0.9836 +- 0.0003 | 0.009 +- 0.000 | 3.72e+05 | 1.00e+12 | 133096 +- 6988 |
| 0.1 | prune-after: one-shot magnitude (global) + fine-tune | 3 | 186,163 | 0.9869 +- 0.0003 | 0.008 +- 0.000 | 3.72e+05 | 1.06e+13 | 87336 +- 7781 |
| 0.1 | prune-after: gradual magnitude (global) during fine-tune | 3 | 186,163 | 0.9870 +- 0.0003 | 0.006 +- 0.000 | 3.72e+05 | 1.18e+13 | 87336 +- 7781 |
| 0.1 | prune-during: magnitude (uniform per layer) | 3 | 186,164 | 0.9867 +- 0.0003 | 0.007 +- 0.000 | 3.72e+05 | 3.33e+12 | 87336 +- 7781 |
| 0.1 | prune-during: magnitude (global) | 3 | 186,163 | 0.9867 +- 0.0002 | 0.008 +- 0.000 | 3.72e+05 | 3.33e+12 | 87336 +- 7781 |
| 0.1 | prune-during: activity, Hebbian (uniform per layer) | 3 | 186,164 | 0.9819 +- 0.0007 | 0.004 +- 0.000 | 3.72e+05 | 3.33e+12 | 87336 +- 7781 |
| 0.1 | prune-during: abs(w) x activity, Hebbian (uniform per layer) | 3 | 186,164 | 0.9857 +- 0.0004 | 0.006 +- 0.000 | 3.72e+05 | 3.33e+12 | 87336 +- 7781 |
| 0.1 | prune-during: synaptic drive (uniform per layer) | 3 | 186,164 | 0.9853 +- 0.0004 | 0.007 +- 0.000 | 3.72e+05 | 3.33e+12 | 87336 +- 7781 |
| 0.1 | prune-during: random (uniform per layer) | 3 | 186,164 | 0.9823 +- 0.0012 | 0.004 +- 0.000 | 3.72e+05 | 3.33e+12 | 87336 +- 7781 |
| 0.1 | SET (dynamic sparse training, random regrowth) | 3 | 186,164 | 0.9838 +- 0.0004 | 0.010 +- 0.001 | 3.72e+05 | 1.01e+12 | 138766 +- 2328 |
| 0.1 | RigL (dynamic sparse training, gradient regrowth) | 3 | 186,164 | 0.9853 +- 0.0011 | 0.008 +- 0.001 | 3.72e+05 | 1.01e+12 | 123519 +- 1928 |
| 0.1 | static random sparse | 3 | 186,164 | 0.9818 +- 0.0002 | 0.006 +- 0.000 | 3.72e+05 | 1.01e+12 | 206096 +- 4555 |
| 0.05 | dense small (additive) | 3 | 93,392 | 0.9801 +- 0.0003 | 0.009 +- 0.001 | 1.87e+05 | 5.04e+11 | 186445 +- 12401 |
| 0.05 | prune-after: one-shot magnitude (global) + fine-tune | 3 | 93,082 | 0.9873 +- 0.0003 | 0.006 +- 0.000 | 1.86e+05 | 1.03e+13 | 87336 +- 7781 |
| 0.05 | prune-after: gradual magnitude (global) during fine-tune | 3 | 93,082 | 0.9864 +- 0.0002 | 0.004 +- 0.000 | 1.86e+05 | 1.16e+13 | 87336 +- 7781 |
| 0.05 | prune-during: magnitude (uniform per layer) | 3 | 93,082 | 0.9868 +- 0.0006 | 0.006 +- 0.001 | 1.86e+05 | 2.96e+12 | 87336 +- 7781 |
| 0.05 | prune-during: magnitude (global) | 3 | 93,082 | 0.9872 +- 0.0003 | 0.006 +- 0.001 | 1.86e+05 | 2.96e+12 | 87336 +- 7781 |
| 0.05 | prune-during: activity, Hebbian (uniform per layer) | 3 | 93,082 | 0.9787 +- 0.0001 | 0.005 +- 0.001 | 1.86e+05 | 2.96e+12 | 87336 +- 7781 |
| 0.05 | prune-during: abs(w) x activity, Hebbian (uniform per layer) | 3 | 93,082 | 0.9842 +- 0.0004 | 0.004 +- 0.000 | 1.86e+05 | 2.96e+12 | 87336 +- 7781 |
| 0.05 | prune-during: synaptic drive (uniform per layer) | 3 | 93,082 | 0.9850 +- 0.0003 | 0.005 +- 0.000 | 1.86e+05 | 2.96e+12 | 87336 +- 7781 |
| 0.05 | prune-during: random (uniform per layer) | 3 | 93,082 | 0.9713 +- 0.0011 | 0.015 +- 0.000 | 1.86e+05 | 2.96e+12 | 87336 +- 7781 |
| 0.05 | SET (dynamic sparse training, random regrowth) | 3 | 93,082 | 0.9819 +- 0.0009 | 0.006 +- 0.001 | 1.86e+05 | 5.03e+11 | 209140 +- 5774 |
| 0.05 | RigL (dynamic sparse training, gradient regrowth) | 3 | 93,082 | 0.9822 +- 0.0004 | 0.008 +- 0.000 | 1.86e+05 | 5.03e+11 | 153687 +- 5905 |
| 0.05 | static random sparse | 3 | 93,082 | 0.9754 +- 0.0003 | 0.004 +- 0.001 | 1.86e+05 | 5.03e+11 | 378278 +- 18120 |
| 0.02 | dense small (additive) | 3 | 36,872 | 0.9730 +- 0.0010 | 0.006 +- 0.000 | 7.37e+04 | 1.99e+11 | 436301 +- 46700 |
| 0.02 | prune-after: one-shot magnitude (global) + fine-tune | 3 | 37,233 | 0.9838 +- 0.0008 | 0.003 +- 0.001 | 7.45e+04 | 1.02e+13 | 87336 +- 7781 |
| 0.02 | prune-after: gradual magnitude (global) during fine-tune | 3 | 37,233 | 0.9849 +- 0.0004 | 0.004 +- 0.000 | 7.45e+04 | 1.15e+13 | 87336 +- 7781 |
| 0.02 | prune-during: magnitude (uniform per layer) | 3 | 37,233 | 0.9855 +- 0.0007 | 0.003 +- 0.000 | 7.45e+04 | 2.73e+12 | 87336 +- 7781 |
| 0.02 | prune-during: magnitude (global) | 3 | 37,233 | 0.9865 +- 0.0002 | 0.003 +- 0.001 | 7.45e+04 | 2.73e+12 | 87336 +- 7781 |
| 0.02 | prune-during: activity, Hebbian (uniform per layer) | 3 | 37,233 | 0.9655 +- 0.0009 | 0.020 +- 0.001 | 7.45e+04 | 2.73e+12 | 87336 +- 7781 |
| 0.02 | prune-during: abs(w) x activity, Hebbian (uniform per layer) | 3 | 37,233 | 0.9805 +- 0.0004 | 0.005 +- 0.001 | 7.45e+04 | 2.73e+12 | 87336 +- 7781 |
| 0.02 | prune-during: synaptic drive (uniform per layer) | 3 | 37,233 | 0.9822 +- 0.0003 | 0.004 +- 0.000 | 7.45e+04 | 2.73e+12 | 87336 +- 7781 |
| 0.02 | prune-during: random (uniform per layer) | 3 | 37,233 | 0.9339 +- 0.0015 | 0.034 +- 0.002 | 7.45e+04 | 2.73e+12 | 87336 +- 7781 |
| 0.02 | SET (dynamic sparse training, random regrowth) | 3 | 37,233 | 0.9743 +- 0.0006 | 0.005 +- 0.000 | 7.45e+04 | 2.01e+11 | 467683 +- 20709 |
| 0.02 | RigL (dynamic sparse training, gradient regrowth) | 3 | 37,233 | 0.9785 +- 0.0004 | 0.005 +- 0.001 | 7.45e+04 | 2.01e+11 | 245749 +- 12815 |
| 0.02 | static random sparse | 3 | 37,233 | 0.9590 +- 0.0013 | 0.009 +- 0.001 | 7.45e+04 | 2.01e+11 | - |
| 0.01 | dense small (additive) | 3 | 18,791 | 0.9630 +- 0.0012 | 0.004 +- 0.001 | 3.76e+04 | 1.01e+11 | - |
| 0.01 | prune-after: one-shot magnitude (global) + fine-tune | 3 | 18,616 | 0.9752 +- 0.0020 | 0.006 +- 0.000 | 3.72e+04 | 1.01e+13 | 87336 +- 7781 |
| 0.01 | prune-after: gradual magnitude (global) during fine-tune | 2 | 18,616 | 0.9795 +- 0.0006 | 0.010 +- 0.000 | 3.72e+04 | 1.14e+13 | 84378 +- 8034 |
| 0.01 | prune-during: magnitude (uniform per layer) | 3 | 18,616 | 0.9823 +- 0.0001 | 0.007 +- 0.000 | 3.72e+04 | 2.66e+12 | 87336 +- 7781 |
| 0.01 | prune-during: magnitude (global) | 3 | 18,616 | 0.9844 +- 0.0003 | 0.004 +- 0.000 | 3.72e+04 | 2.66e+12 | 87336 +- 7781 |
| 0.01 | prune-during: activity, Hebbian (uniform per layer) | 3 | 18,616 | 0.9366 +- 0.0015 | 0.037 +- 0.001 | 3.72e+04 | 2.66e+12 | 87336 +- 7781 |
| 0.01 | prune-during: abs(w) x activity, Hebbian (uniform per layer) | 3 | 18,616 | 0.9725 +- 0.0008 | 0.012 +- 0.001 | 3.72e+04 | 2.66e+12 | 87336 +- 7781 |
| 0.01 | prune-during: synaptic drive (uniform per layer) | 3 | 18,616 | 0.9756 +- 0.0010 | 0.012 +- 0.000 | 3.72e+04 | 2.66e+12 | 87336 +- 7781 |
| 0.01 | prune-during: random (uniform per layer) | 3 | 18,616 | 0.8913 +- 0.0060 | 0.071 +- 0.003 | 3.72e+04 | 2.66e+12 | 87336 +- 7781 |
| 0.01 | SET (dynamic sparse training, random regrowth) | 3 | 18,616 | 0.9603 +- 0.0010 | 0.008 +- 0.001 | 3.72e+04 | 1.01e+11 | - |
| 0.01 | RigL (dynamic sparse training, gradient regrowth) | 3 | 18,616 | 0.9741 +- 0.0016 | 0.004 +- 0.001 | 3.72e+04 | 1.01e+11 | 444658 +- 67026 |
| 0.01 | static random sparse | 3 | 18,616 | 0.9259 +- 0.0034 | 0.015 +- 0.001 | 3.72e+04 | 1.01e+11 | - |
| 0.005 | dense small (additive) | 3 | 9,672 | 0.9407 +- 0.0010 | 0.009 +- 0.001 | 1.93e+04 | 5.22e+10 | - |
| 0.005 | prune-after: one-shot magnitude (global) + fine-tune | 3 | 9,308 | 0.9451 +- 0.0060 | 0.017 +- 0.006 | 1.86e+04 | 1.01e+13 | 87336 +- 7781 |
| 0.005 | prune-during: magnitude (uniform per layer) | 3 | 9,308 | 0.9704 +- 0.0005 | 0.015 +- 0.001 | 1.86e+04 | 2.62e+12 | 87336 +- 7781 |
| 0.005 | prune-during: magnitude (global) | 3 | 9,308 | 0.9781 +- 0.0002 | 0.006 +- 0.000 | 1.86e+04 | 2.62e+12 | 87336 +- 7781 |
| 0.005 | prune-during: activity, Hebbian (uniform per layer) | 3 | 9,322 | 0.8885 +- 0.0109 | 0.100 +- 0.037 | 1.86e+04 | 2.62e+12 | 87336 +- 7781 |
| 0.005 | prune-during: abs(w) x activity, Hebbian (uniform per layer) | 3 | 9,308 | 0.9522 +- 0.0013 | 0.038 +- 0.016 | 1.86e+04 | 2.62e+12 | 87336 +- 7781 |
| 0.005 | prune-during: synaptic drive (uniform per layer) | 3 | 9,308 | 0.9603 +- 0.0020 | 0.025 +- 0.000 | 1.86e+04 | 2.62e+12 | 87336 +- 7781 |
| 0.005 | prune-during: random (uniform per layer) | 3 | 9,308 | 0.7184 +- 0.0230 | 0.157 +- 0.017 | 1.86e+04 | 2.62e+12 | 87336 +- 7781 |
| 0.005 | SET (dynamic sparse training, random regrowth) | 3 | 9,308 | 0.9365 +- 0.0011 | 0.012 +- 0.001 | 1.86e+04 | 5.03e+10 | - |
| 0.005 | RigL (dynamic sparse training, gradient regrowth) | 3 | 9,308 | 0.9631 +- 0.0027 | 0.005 +- 0.001 | 1.86e+04 | 5.03e+10 | - |
| 0.005 | static random sparse | 3 | 9,308 | 0.7271 +- 0.0271 | 0.053 +- 0.012 | 1.86e+04 | 5.03e+10 | - |

density 는 과잉 초기화 망(784-1024-1024-10, 1,861,632 가중치) 대비 최종 활성 비율. dense small 은 같은 예산의 작은 망. 학습 FLOPs 는 매 스텝 실제 활성 연결 기준 누적 (순전파 x3).
