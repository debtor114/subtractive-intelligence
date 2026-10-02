#!/usr/bin/env bash
# 밤샘 순차 실행 체인. 한 번에 torch 프로세스 하나만 띄운다 (16GB RAM, 커밋 한도 때문).
# 사용: bash scripts/overnight.sh  (Git Bash). 로그: results/_overnight.log
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
LOG=results/_overnight.log
run() {
  echo "=== $(date +%T) START $*" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "=== $(date +%T) EXIT $? $*" >> "$LOG"
}
: >> "$LOG"
echo "##### overnight chain start $(date)" >> "$LOG"

# 1. 핵심 실험 나머지 밀도
run $PY -u experiments/core_prune_during_learning/run_all.py --densities 0.05 0.02 0.01 0.005 --seeds 0 1 2 --epochs 15 --skip_dense_big --skip_existing
# 2. STDP 본 실행 (400 뉴런, 60k)
run $PY -u experiments/exp4_stdp_temporal/run.py --n_e 400 --train 60000 --seed 0 --set intensity=32
# 3. 실험 1 MNIST 나머지 (pre 모드)
run $PY -u experiments/exp1_prerouting_attention/run_all.py --dataset mnist --seeds 0 1 2 --epochs 8 --skip_existing
# 4. 실험 3 permuted 나머지
run $PY -u experiments/exp3_dual_learning/run_all.py --datasets permuted_mnist --seeds 0 1 2 --skip_existing
# 5. 실험 2 MNIST (기준선 ViT 3 시드)
run $PY -u experiments/exp2_predictive_coding/run.py --ckpt "results/baseline_vit_mnist/seed0_*/model_final.pt" --dataset mnist --seed 0
run $PY -u experiments/exp2_predictive_coding/run.py --ckpt "results/baseline_vit_mnist/seed1_*/model_final.pt" --dataset mnist --seed 1
run $PY -u experiments/exp2_predictive_coding/run.py --ckpt "results/baseline_vit_mnist/seed2_*/model_final.pt" --dataset mnist --seed 2
# 6. CIFAR-10 기준선 (중단됐던 것) -> 실험 2 CIFAR
rm -rf results/baseline_vit_cifar10
run $PY -u scripts/train_baseline.py --config configs/vit_cifar10.yaml --set seed=0
run $PY -u experiments/exp2_predictive_coding/run.py --ckpt "results/baseline_vit_cifar10/seed0_*/model_final.pt" --dataset cifar10 --seed 0
# 7. 핵심 실험 추가 규칙 (시냅스 구동)
run $PY -u experiments/core_prune_during_learning/run_all.py --densities 0.1 0.05 0.02 0.01 0.005 --seeds 0 1 2 --epochs 15 --skip_dense_big --skip_existing --arms pd_drive_layer
# 8. 실험 1 CIFAR (30 에폭, 시드 1)
run $PY -u experiments/exp1_prerouting_attention/run_all.py --dataset cifar10 --seeds 0 --variants full post_0.25 pre_0.25 --epochs 30 --skip_existing
# 9. STDP 추가 시드 / 크기
run $PY -u experiments/exp4_stdp_temporal/run.py --n_e 400 --train 60000 --seed 1 --set intensity=32
run $PY -u experiments/exp4_stdp_temporal/run.py --n_e 100 --train 60000 --seed 0 --set intensity=32
run $PY -u experiments/exp4_stdp_temporal/run.py --n_e 400 --train 60000 --seed 2 --set intensity=32
run $PY -u experiments/exp4_stdp_temporal/run.py --n_e 1600 --train 60000 --seed 0 --set intensity=32
echo "##### OVERNIGHT_DONE $(date)" >> "$LOG"
