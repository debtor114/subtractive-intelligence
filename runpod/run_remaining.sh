#!/usr/bin/env bash
# 팟에서 남은 실행 전부 (완료분은 --skip_existing 으로 건너뜀). nohup 으로 띄우고 로그는 results/_pod.log.
#   cd /workspace/subtractive-intelligence && nohup bash runpod/run_remaining.sh > results/_pod.log 2>&1 &
set -u
cd /workspace/subtractive-intelligence
source .venv_pod/bin/activate
export PYTHONUNBUFFERED=1
echo "##### pod chain start $(date)"
# 1. 실험 A: 사전 학습 ResNet-18 -> CIFAR-10 적응하며 깎기 (38 런)
python experiments/core_prune_during_learning/run_all_pretrained.py --densities 0.05 0.02 0.005 --seeds 0 1 --epochs 10 --skip_existing
echo "##### exp A done $(date)"
# 2. 논문 2 토대: 수면 감쇠 스케줄 + 분산 증류 (48 런)
python experiments/exp3_dual_learning/run_all.py --datasets split_mnist permuted_mnist --seeds 0 1 2 \
  --only cls2_decay_boundary cls2_decay_periodic cls2_decay_periodic_small cls2_decay_continuous \
         cls2_kd_none_h256 cls2_kd_global cls2_kd_local cls2_kd_local_logits --skip_existing
echo "##### paper2 done $(date)"
# 3. 여유가 있으면: ResNet-18 시드 2 (논문 표를 3 시드로)
python experiments/core_prune_during_learning/run_all_cifar.py --model resnet18 --densities 0.05 0.02 0.005 --seeds 2 --epochs 20 \
  --arms dense_small pd_mag_global pd_mag_erk rigl ttp --skip_existing
echo "##### resnet seed2 done $(date)"
echo "POD_CHAIN_DONE $(date)"
