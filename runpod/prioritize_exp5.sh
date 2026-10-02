#!/usr/bin/env bash
# 실험 5 의 결정적 비교 (d=0.05, dyn_local vs dyn_random, 시드 0) 가 먼저 나오도록 다른 갈래를 SIGSTOP 으로 잠시 멈추고,
# 두 결과 파일이 생기면 SIGCONT 로 이어서 돌린다. 작업 손실 없음 (GPU 메모리는 멈춘 동안도 잡혀 있음, 약 2GB/프로세스).
#   cd /workspace/subtractive-intelligence && nohup bash runpod/prioritize_exp5.sh > results/_pod_prioritize.log 2>&1 &
set -u
cd /workspace/subtractive-intelligence
PIDS="$(pgrep -f 'run_all_pretraine[d]') $(pgrep -f 'run_all_cifa[r]') $(pgrep -f 'arms kwta_i[n]')"
echo "pausing: $PIDS $(date)"
for p in $PIDS; do kill -STOP "$p" 2>/dev/null && echo "  stopped $p: $(ps -o state=,args= -p "$p" | cut -c1-80)"; done
until [ -f results/exp5_dynamic/d0.05/dyn_local/seed0.json ] && [ -f results/exp5_dynamic/d0.05/dyn_random/seed0.json ]; do sleep 60; done
echo "decisive pair done, resuming: $PIDS $(date)"
for p in $PIDS; do kill -CONT "$p" 2>/dev/null && echo "  resumed $p"; done
echo "RESUMED $(date)" > results/_pod_resumed.flag
