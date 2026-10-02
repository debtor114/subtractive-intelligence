#!/usr/bin/env bash
# numpy 2 (trapz 제거) 수정 후 결과 저장 단계에서 죽었던 세 갈래를 다시 띄운다: 실험 A 시드 0/1, ResNet-18 시드 2.
# 완료 표식: results/_pod_fix_all.done (논문 2 두 갈래는 원래 실행기가 _pod_all.done 을 남긴다)
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
launch resnet_s2 python experiments/core_prune_during_learning/run_all_cifar.py --model resnet18 --densities 0.05 0.02 0.005 --seeds 2 --epochs 20 --arms dense_small pd_mag_global pd_mag_erk rigl ttp --skip_existing
wait
echo "FIX_DONE $(date)" > results/_pod_fix_all.done
