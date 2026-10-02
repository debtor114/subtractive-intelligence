| mode | keep ratio | k (keys/query) | seeds | test acc | ECE | attn FLOPs/block (dense) | attn FLOPs/block (sparsity exploited) | total FLOPs ratio |
|---|---|---|---|---|---|---|---|---|
| full | 1 | 65 | 1 | 0.7940 | 0.035 | 3.24e+06 | 3.24e+06 | 1.000 |
| post | 0.25 | 17 | 1 | 0.8027 | 0.033 | 3.24e+06 | 2.05e+06 | 0.980 |
| pre | 0.25 | 17 | 1 | 0.8003 | 0.036 | 1.05e+06 | 1.05e+06 | 0.964 |

토큰 수 65. post = 사후 top-k 마스킹, pre = 저차원 라우터 사전 선택 (r=8). 총 FLOPs 비율은 full 대비, 어텐션 부분만 모드별 해석값으로 치환.
