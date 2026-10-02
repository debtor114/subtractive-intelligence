#!/usr/bin/env bash
# 실험 5 연결 단위 두 팔만 다시 띄운다 (kthvalue 마스크 + 체크포인트 수정 후). 채널 단위/정적 0.2 는 start_exp5_now.sh 의 light 갈래가 계속 돈다.
# 순서: d=0.05 (s0, s1) -> 0.005 -> 0.2. 완료 표식: results/_pod_exp5_dyn.done
set -u
cd /workspace/subtractive-intelligence
source .venv_pod/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
launch() {
  local name=$1; shift
  ( "$@" > "results/_pod_${name}.log" 2>&1; echo "DONE exit=$? $(date)" > "results/_pod_${name}.done" ) &
  echo "launched $name pid $!"
}
launch exp5_local  python experiments/exp5_dynamic_pruning/run_all.py --arms dyn_local  --seeds 0 1 --densities 0.05 0.005 0.2 --skip_existing
launch exp5_random python experiments/exp5_dynamic_pruning/run_all.py --arms dyn_random --seeds 0 1 --densities 0.05 0.005 0.2 --skip_existing
wait
echo "EXP5_DYN_DONE $(date)" > results/_pod_exp5_dyn.done
