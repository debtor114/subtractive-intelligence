#!/usr/bin/env bash
# 실험 A 두 갈래만 다시 띄운다 (run_pretrained.py fc 보호 수정본 반영 후). run_parallel.sh 와 같은 스트림 이름을 쓴다.
set -u
cd /workspace/subtractive-intelligence
source .venv_pod/bin/activate
export PYTHONUNBUFFERED=1
launch() {
  local name=$1; shift
  ( "$@" > "results/_pod_${name}.log" 2>&1; echo "DONE exit=$? $(date)" > "results/_pod_${name}.done" ) &
  echo "launched $name pid $!"
}
launch expA_s0 python experiments/core_prune_during_learning/run_all_pretrained.py --densities 0.05 0.02 0.005 --seeds 0 --epochs 10 --skip_existing
launch expA_s1 python experiments/core_prune_during_learning/run_all_pretrained.py --densities 0.05 0.02 0.005 --seeds 1 --epochs 10 --skip_existing
wait
echo "EXPA_DONE $(date)" > results/_pod_expA_all.done
