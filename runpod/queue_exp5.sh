#!/usr/bin/env bash
# 실험 5 (추론 중 가지치기) 를 앞 작업이 모두 끝난 뒤 자동 시작하는 대기열. nohup 으로 띄운다:
#   cd /workspace/subtractive-intelligence && nohup bash runpod/queue_exp5.sh > results/_pod_queue_exp5.log 2>&1 &
# 완료 표식: results/_pod_exp5.done
set -u
cd /workspace/subtractive-intelligence
source .venv_pod/bin/activate
export PYTHONUNBUFFERED=1
echo "queue_exp5 waiting for _pod_all.done + _pod_fix_all.done $(date)"
until [ -f results/_pod_all.done ] && [ -f results/_pod_fix_all.done ]; do sleep 60; done
echo "##### exp5 start $(date)"
launch() {
  local name=$1; shift
  ( "$@" > "results/_pod_${name}.log" 2>&1; echo "DONE exit=$? $(date)" > "results/_pod_${name}.done" ) &
  echo "launched $name pid $!"
}
# 동적 팔: 시드별 두 갈래 (dyn 경로는 표본별 가중치라 메모리 약 16GB/프로세스)
launch exp5_s0 python experiments/exp5_dynamic_pruning/run_all.py --densities 0.05 0.005 0.2 --seeds 0 --arms dyn_local dyn_random kwta_in --skip_existing
launch exp5_s1 python experiments/exp5_dynamic_pruning/run_all.py --densities 0.05 0.005 0.2 --seeds 1 --arms dyn_local dyn_random kwta_in --skip_existing
# 정적 대조군 중 아직 없는 밀도 0.2 (학습 중 전역 크기 / dense small), 시드 0 1
launch static02 python experiments/core_prune_during_learning/run_all_cifar.py --model resnet18 --densities 0.2 --seeds 0 1 --epochs 20 --arms pd_mag_global dense_small --skip_existing
wait
echo "EXP5_DONE $(date)" > results/_pod_exp5.done
echo "##### exp5 done $(date)"
