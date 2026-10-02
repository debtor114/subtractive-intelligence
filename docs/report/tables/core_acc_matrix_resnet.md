| density | dense small (additive) | train-then-prune + finetune | prune-during: magnitude (layer) | prune-during: magnitude (global) | prune-during: magnitude (ERK layer budget) | prune-during: activity (layer) | prune-during: abs(w) x activity (layer) | prune-during: synaptic drive (layer) | prune-during: synaptic drive (ERK layer budget) | prune-during: synaptic drive, per-neuron normalized (ERK) | prune-during: random (layer) | SET (dynamic, random regrow) | RigL (dynamic, grad regrow) | static random sparse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.005 | 0.8131 +- 0.0026 | 0.8364 +- 0.0015 | 0.8311 +- 0.0006 | 0.8896 +- 0.0003 | 0.8783 +- 0.0022 | - | - | - | - | - | - | - | 0.8486 +- 0.0030 | - |
| 0.02 | 0.8647 +- 0.0030 | 0.9094 +- 0.0018 | - | 0.9146 +- 0.0015 | 0.9125 +- 0.0012 | - | - | - | - | - | - | - | 0.8950 +- 0.0018 | - |
| 0.05 | 0.8866 +- 0.0018 | 0.9227 +- 0.0006 | 0.9108 +- 0.0002 | 0.9196 +- 0.0023 | 0.9203 +- 0.0009 | - | - | - | - | - | - | - | 0.9097 +- 0.0016 | - |
| 0.2 | 0.9093 +- 0.0000 | - | 0.9179 +- 0.0009 | 0.9189 +- 0.0013 | - | - | - | - | - | - | - | - | - | - |
