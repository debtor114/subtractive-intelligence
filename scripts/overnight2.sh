#!/usr/bin/env bash
# 2 차 체인: 실험 2 미세조정 변형 (학습 시 적응). overnight.sh 가 끝난 뒤 실행.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
LOG=results/_overnight.log
run() {
  echo "=== $(date +%T) START $*" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "=== $(date +%T) EXIT $? $*" >> "$LOG"
}
echo "##### overnight2 chain start $(date)" >> "$LOG"
run $PY -u experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_mnist/seed0_*/model_final.pt" --dataset mnist --score predicted --skip 0.5 --epochs 2 --seed 0
run $PY -u experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_mnist/seed0_*/model_final.pt" --dataset mnist --score random --skip 0.5 --epochs 2 --seed 0
run $PY -u experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_mnist/seed1_*/model_final.pt" --dataset mnist --score predicted --skip 0.5 --epochs 2 --seed 1
run $PY -u experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_mnist/seed1_*/model_final.pt" --dataset mnist --score random --skip 0.5 --epochs 2 --seed 1
echo "##### OVERNIGHT2_DONE $(date)" >> "$LOG"
# STDP seed 0 재실행 (평가 간격 5k 로 촘촘하게; 첫 실행은 20k 간격이었음)
run $PY -u experiments/exp4_stdp_temporal/run.py --n_e 400 --train 60000 --seed 0 --set intensity=32
echo "##### OVERNIGHT2_DONE (with stdp rerun) $(date)" >> "$LOG"
# CIFAR-10 미세조정 변형 (사후 적용이 크게 무너지므로 학습 시 적응 효과 확인)
run $PY -u experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_cifar10/seed0_*/model_final.pt" --dataset cifar10 --score predicted --skip 0.5 --epochs 3 --seed 0
run $PY -u experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_cifar10/seed0_*/model_final.pt" --dataset cifar10 --score random --skip 0.5 --epochs 3 --seed 0
echo "##### OVERNIGHT2_DONE (all) $(date)" >> "$LOG"
