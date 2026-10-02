#!/usr/bin/env bash
# 실험 5 를 앞 작업을 기다리지 않고 지금 띄운다 (queue_exp5.sh 대체). 결정적 비교가 먼저 나오도록 팔별 갈래:
#   local  : dyn_local  d=0.05 (s0, s1) -> 0.005 -> 0.2
#   random : dyn_random d=0.05 (s0, s1) -> 0.005 -> 0.2
#   light  : kwta_in 전부 -> 정적 대조군 밀도 0.2 (pd_mag_global, dense_small)
# 완료 표식: results/_pod_exp5_<stream>.done, 전부 끝나면 results/_pod_exp5.done
set -u
cd /workspace/subtractive-intelligence
source .venv_pod/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 표본별 가중치 텐서가 커서 캐시 할당기 조각화를 줄인다
launch() {
  local name=$1; shift
  ( "$@" > "results/_pod_${name}.log" 2>&1; echo "DONE exit=$? $(date)" > "results/_pod_${name}.done" ) &
  echo "launched $name pid $!"
}
launch exp5_local  python experiments/exp5_dynamic_pruning/run_all.py --arms dyn_local  --seeds 0 1 --densities 0.05 0.005 0.2 --skip_existing
launch exp5_random python experiments/exp5_dynamic_pruning/run_all.py --arms dyn_random --seeds 0 1 --densities 0.05 0.005 0.2 --skip_existing
launch exp5_light  bash -c "python experiments/exp5_dynamic_pruning/run_all.py --arms kwta_in --seeds 0 1 --densities 0.05 0.005 0.2 --skip_existing; python experiments/core_prune_during_learning/run_all_cifar.py --model resnet18 --densities 0.2 --seeds 0 1 --epochs 20 --arms pd_mag_global dense_small --skip_existing"
wait
echo "EXP5_DONE $(date)" > results/_pod_exp5.done
echo "##### exp5 done $(date)"
