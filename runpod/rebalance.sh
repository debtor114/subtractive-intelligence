#!/usr/bin/env bash
# 로컬과 분담하기 위한 팟 쪽 재배치 (2026-10-02 새벽). 로컬이 ResNet 시드 2 d=0.005 와 dyn_local d=0.2 를 맡는다.
#  (a) ResNet 시드 2 갈래: 진행 중인 ttp d=0.02 가 끝나면 멈춘다 (d=0.005 는 로컬).
#  (b) dyn_local 갈래: 진행 중인 d=0.05 s1 이 끝나면 멈추고 d=0.005 만 다시 띄운다 (d=0.2 는 로컬).
# 완료 표식: results/_pod_dyn_local_0005.done
set -u
cd /workspace/subtractive-intelligence
source .venv_pod/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
(
  until [ -f results/core_resnet/d0.02/ttp/seed2.json ]; do sleep 60; done
  pkill -f 'run_all_cifar.py --model resnet18 --densities 0.05 0.02 0.005 --seeds [2]'
  echo "resnet_s2 stopped after ttp d0.02 $(date)"
) &
(
  until [ -f results/exp5_dynamic/d0.05/dyn_local/seed1.json ]; do sleep 60; done
  pkill -f 'arms dyn_loca[l] --seeds 0 1 --densities 0.05'
  sleep 5
  echo "dyn_local restarted for d=0.005 only $(date)"
  python experiments/exp5_dynamic_pruning/run_all.py --arms dyn_local --seeds 0 1 --densities 0.005 --skip_existing > results/_pod_exp5_local_0005.log 2>&1
  echo "DONE $(date)" > results/_pod_dyn_local_0005.done
) &
wait
echo "REBALANCE_DONE $(date)"
