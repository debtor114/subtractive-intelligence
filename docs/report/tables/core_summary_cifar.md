| density | arm | seeds | active weights | test acc | ECE | infer FLOPs | cum. train FLOPs | samples to target (97% MNIST / 85% CIFAR) |
|---|---|---|---|---|---|---|---|---|
| 1 | dense big (no pruning) | 3 | 2,195,648 | 0.9133 +- 0.0010 | 0.031 +- 0.001 | 3.08e+08 | 9.23e+14 | 411367 +- 15611 |
| 0.1 | dense small (additive) | 3 | 219,735 | 0.8729 +- 0.0007 | 0.029 +- 0.001 | 3.10e+07 | 9.31e+13 | 632089 +- 19636 |
| 0.1 | dense small, shallow and wide (additive) | 3 | 219,918 | 0.8685 +- 0.0026 | 0.025 +- 0.002 | 8.09e+07 | 2.43e+14 | 676291 +- 26535 |
| 0.1 | prune-after: one-shot magnitude (global) + fine-tune | 3 | 219,565 | 0.9072 +- 0.0014 | 0.036 +- 0.001 | 8.19e+07 | 1.05e+15 | 411367 +- 15611 |
| 0.1 | prune-after: gradual magnitude (global) during fine-tune | 3 | 219,565 | 0.9051 +- 0.0023 | 0.036 +- 0.002 | 8.05e+07 | 1.15e+15 | 440414 +- 23340 |
| 0.1 | prune-during: magnitude (global) | 3 | 219,565 | 0.9028 +- 0.0029 | 0.033 +- 0.001 | 7.37e+07 | 4.19e+14 | 387708 +- 26819 |
| 0.1 | prune-during: magnitude (ERK per layer) | 3 | 219,565 | 0.8930 +- 0.0004 | 0.030 +- 0.001 | 2.69e+07 | 2.80e+14 | 374564 +- 22672 |
| 0.1 | prune-during: synaptic drive, per-neuron normalised (ERK per layer) | 3 | 219,565 | 0.8759 +- 0.0033 | 0.026 +- 0.003 | 2.69e+07 | 2.80e+14 | 633096 +- 45722 |
| 0.1 | RigL (dynamic sparse training, gradient regrowth) | 3 | 219,565 | 0.8759 +- 0.0005 | 0.026 +- 0.002 | 2.69e+07 | 8.07e+13 | 625325 +- 28326 |
| 0.03 | dense small (additive) | 3 | 67,905 | 0.8284 +- 0.0010 | 0.024 +- 0.001 | 9.72e+06 | 2.92e+13 | - |
| 0.03 | dense small, shallow and wide (additive) | 3 | 68,264 | 0.8391 +- 0.0008 | 0.019 +- 0.001 | 2.57e+07 | 7.72e+13 | - |
| 0.03 | prune-after: one-shot magnitude (global) + fine-tune | 3 | 65,869 | 0.8838 +- 0.0022 | 0.026 +- 0.001 | 3.58e+07 | 9.77e+14 | 411367 +- 15611 |
| 0.03 | prune-after: gradual magnitude (global) during fine-tune | 3 | 65,869 | 0.8797 +- 0.0011 | 0.022 +- 0.001 | 3.42e+07 | 1.11e+15 | 440414 +- 23340 |
| 0.03 | prune-after: one-shot magnitude (global) + fine-tune at lr 0.05 | 3 | 65,869 | 0.8837 +- 0.0012 | 0.031 +- 0.002 | 3.59e+07 | 9.77e+14 | 410359 +- 20729 |
| 0.03 | prune-after: gradual magnitude (global) during fine-tune at lr 0.05 | 3 | 65,869 | 0.8922 +- 0.0015 | 0.028 +- 0.003 | 3.09e+07 | 1.11e+15 | 404766 +- 38638 |
| 0.03 | prune-during: magnitude (global) | 3 | 65,869 | 0.8908 +- 0.0017 | 0.028 +- 0.002 | 2.98e+07 | 3.40e+14 | 391024 +- 6160 |
| 0.03 | prune-during: magnitude (ERK per layer) | 3 | 65,870 | 0.8692 +- 0.0026 | 0.023 +- 0.003 | 8.38e+06 | 2.39e+14 | 495322 +- 145564 |
| 0.03 | prune-during: synaptic drive, per-neuron normalised (ERK per layer) | 3 | 65,870 | 0.7928 +- 0.0048 | 0.020 +- 0.003 | 8.38e+06 | 2.39e+14 | - |
| 0.03 | RigL (dynamic sparse training, gradient regrowth) | 3 | 65,870 | 0.8353 +- 0.0019 | 0.018 +- 0.001 | 8.38e+06 | 2.52e+13 | - |
| 0.03 | RigL, 3x training length | 3 | 65,870 | 0.8702 +- 0.0022 | 0.025 +- 0.001 | 8.38e+06 | 7.55e+13 | 1909949 +- 39835 |
| 0.01 | dense small (additive) | 3 | 22,911 | 0.7709 +- 0.0036 | 0.020 +- 0.004 | 3.32e+06 | 9.97e+12 | - |
| 0.01 | dense small, shallow and wide (additive) | 3 | 22,074 | 0.7836 +- 0.0026 | 0.016 +- 0.002 | 8.53e+06 | 2.56e+13 | - |
| 0.01 | prune-after: one-shot magnitude (global) + fine-tune | 3 | 21,956 | 0.8051 +- 0.0010 | 0.015 +- 0.003 | 1.60e+07 | 9.47e+14 | 411367 +- 15611 |
| 0.01 | prune-after: gradual magnitude (global) during fine-tune | 3 | 21,956 | 0.8192 +- 0.0021 | 0.012 +- 0.002 | 1.47e+07 | 1.09e+15 | 440414 +- 23340 |
| 0.01 | prune-after: one-shot magnitude (global) + fine-tune at lr 0.05 | 3 | 21,956 | 0.8113 +- 0.0026 | 0.020 +- 0.001 | 1.60e+07 | 9.47e+14 | 410804 +- 17276 |
| 0.01 | prune-after: gradual magnitude (global) during fine-tune at lr 0.05 | 3 | 21,956 | 0.8481 +- 0.0018 | 0.021 +- 0.000 | 1.27e+07 | 1.09e+15 | 419317 +- 25110 |
| 0.01 | prune-during: magnitude (global) | 3 | 21,956 | 0.8570 +- 0.0020 | 0.023 +- 0.001 | 1.16e+07 | 3.09e+14 | 390029 +- 6850 |
| 0.01 | prune-during: magnitude (ERK per layer) | 3 | 21,957 | 0.8134 +- 0.0007 | 0.015 +- 0.001 | 2.78e+06 | 2.27e+14 | - |
| 0.01 | prune-during: synaptic drive, per-neuron normalised (ERK per layer) | 3 | 21,957 | 0.6004 +- 0.0141 | 0.013 +- 0.006 | 2.78e+06 | 2.27e+14 | - |
| 0.01 | RigL (dynamic sparse training, gradient regrowth) | 3 | 21,957 | 0.5392 +- 0.3110 | 0.008 +- 0.006 | 2.78e+06 | 8.35e+12 | - |
| 0.01 | RigL, 3x training length | 3 | 21,957 | 0.5666 +- 0.3299 | 0.010 +- 0.007 | 2.78e+06 | 2.50e+13 | - |

density 는 과잉 초기화 망(784-1024-1024-10, 1,861,632 가중치) 대비 최종 활성 비율. dense small 은 같은 예산의 작은 망. 학습 FLOPs 는 매 스텝 실제 활성 연결 기준 누적 (순전파 x3).
