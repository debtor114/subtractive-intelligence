#!/usr/bin/env bash
# 팟(A40 48GB) 에서 남은 실행을 5 갈래 병렬로. 각 갈래는 서로 다른 결과 파일을 쓰므로 충돌 없음.
#   cd /workspace/subtractive-intelligence && bash runpod/run_parallel.sh
# 로그: results/_pod_<stream>.log, 완료 표시: results/_pod_<stream>.done
set -u
cd /workspace/subtractive-intelligence
source .venv_pod/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p results
launch() {  # launch <name> <command...>
  local name=$1; shift
  ( "$@" > "results/_pod_${name}.log" 2>&1; echo "DONE exit=$? $(date)" > "results/_pod_${name}.done" ) &
  echo "launched $name pid $!"
}
# 실험 A (사전 학습 ResNet-18): 시드별 갈래
launch expA_s0 python experiments/core_prune_during_learning/run_all_pretrained.py --densities 0.05 0.02 0.005 --seeds 0 --epochs 10 --skip_existing
launch expA_s1 python experiments/core_prune_during_learning/run_all_pretrained.py --densities 0.05 0.02 0.005 --seeds 1 --epochs 10 --skip_existing
# 논문 2 토대: 데이터셋별 갈래
P2="cls2_decay_boundary cls2_decay_periodic cls2_decay_periodic_small cls2_decay_continuous cls2_kd_none_h256 cls2_kd_global cls2_kd_local cls2_kd_local_logits"
launch paper2_split python experiments/exp3_dual_learning/run_all.py --datasets split_mnist --seeds 0 1 2 --only $P2 --skip_existing
launch paper2_perm  python experiments/exp3_dual_learning/run_all.py --datasets permuted_mnist --seeds 0 1 2 --only $P2 --skip_existing
# ResNet-18 시드 2 (논문 표를 3 시드로)
launch resnet_s2 python experiments/core_prune_during_learning/run_all_cifar.py --model resnet18 --densities 0.05 0.02 0.005 --seeds 2 --epochs 20 --arms dense_small pd_mag_global pd_mag_erk rigl ttp --skip_existing
wait
echo "ALL_STREAMS_DONE $(date)" > results/_pod_all.done
