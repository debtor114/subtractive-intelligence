| density | from scratch, dense small (width-scaled) | from scratch, prune-during (global magnitude) | pre-trained, one-shot prune then fine-tune | pre-trained, random ERK mask + RigL regrow | pre-trained, magnitude mask + RigL regrow (keeps inherited structure) | pre-trained, prune-during adaptation (ERK layer budget) | pre-trained, prune-during adaptation (global magnitude) | pre-trained, prune-during adaptation, pruning ends at 50% of adaptation (global magnitude) |
|---|---|---|---|---|---|---|---|---|
| 0.005 | 0.7654 +- 0.0001 | 0.8380 +- 0.0037 | 0.8171 +- 0.0014 | 0.7682 +- 0.0037 | 0.8400 +- 0.0025 | 0.7883 +- 0.0022 | 0.8087 +- 0.0012 | 0.8365 +- 0.0013 |
| 0.02 | 0.8240 +- 0.0006 | 0.8747 +- 0.0016 | 0.9029 +- 0.0010 | 0.8355 +- 0.0020 | 0.9061 | 0.8966 +- 0.0019 | 0.9098 +- 0.0008 | - |
| 0.05 | 0.8484 +- 0.0004 | 0.8835 +- 0.0007 | 0.9306 +- 0.0012 | 0.8690 +- 0.0021 | 0.9341 | 0.9343 +- 0.0005 | 0.9365 +- 0.0008 | - |
