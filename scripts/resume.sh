#!/usr/bin/env bash
# 밤샘 체인에서 메모리 보호기에 두 번 중단된 뒤 남은 단계. 사용자가 직접 실행한다.
#   bash scripts/resume.sh
# 실행 전 권장: 브라우저 등 큰 프로그램을 닫아 커밋 메모리를 비운다. 한 번에 torch 프로세스 하나만 뜬다.
# Claude Code 안에서 돌리려면 세션을 CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1 로 시작해야 보호기가 죽이지 않는다.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
LOG=results/_resume.log
run() {
  echo "=== $(date +%T) START $*" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "=== $(date +%T) EXIT $? $*" >> "$LOG"
}
echo "##### resume start $(date)" >> "$LOG"

# 실험 1 CIFAR 사전 라우팅 (고아 프로세스가 끝냈으면 --skip_existing 으로 건너뜀)
run $PY -u experiments/exp1_prerouting_attention/run_all.py --dataset cifar10 --seeds 0 --variants pre_0.25 --epochs 30 --skip_existing
# STDP 추가 시드 / 크기 (seed 0 은 평가 간격 5k 로 재실행)
run $PY -u experiments/exp4_stdp_temporal/run.py --n_e 400 --train 60000 --seed 0 --set intensity=32
run $PY -u experiments/exp4_stdp_temporal/run.py --n_e 400 --train 60000 --seed 1 --set intensity=32
run $PY -u experiments/exp4_stdp_temporal/run.py --n_e 400 --train 60000 --seed 2 --set intensity=32
run $PY -u experiments/exp4_stdp_temporal/run.py --n_e 100 --train 60000 --seed 0 --set intensity=32
run $PY -u experiments/exp4_stdp_temporal/run.py --n_e 1600 --train 60000 --seed 0 --set intensity=32
# 실험 2 미세조정 변형 (학습 시 적응)
run $PY -u experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_mnist/seed0_*/model_final.pt" --dataset mnist --score predicted --skip 0.5 --epochs 2 --seed 0
run $PY -u experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_mnist/seed0_*/model_final.pt" --dataset mnist --score random --skip 0.5 --epochs 2 --seed 0
run $PY -u experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_mnist/seed1_*/model_final.pt" --dataset mnist --score predicted --skip 0.5 --epochs 2 --seed 1
run $PY -u experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_mnist/seed1_*/model_final.pt" --dataset mnist --score random --skip 0.5 --epochs 2 --seed 1
run $PY -u experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_cifar10/seed0_*/model_final.pt" --dataset cifar10 --score predicted --skip 0.5 --epochs 3 --seed 0
run $PY -u experiments/exp2_predictive_coding/finetune.py --ckpt "results/baseline_vit_cifar10/seed0_*/model_final.pt" --dataset cifar10 --score random --skip 0.5 --epochs 3 --seed 0
# 집계
run $PY scripts/analyze.py
echo "##### RESUME_DONE $(date)" >> "$LOG"
