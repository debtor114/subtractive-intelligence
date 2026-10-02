@echo off
rem 팟과 분담: 남은 ResNet-18 시드 2 (밀도 0.005) 와 실험 5 dyn_local 밀도 0.2 를 로컬 3060 Ti 에서 순차 실행.
rem Claude Code 의 백그라운드 셸이 아니라 Start-Process 로 띄워야 메모리 보호기에 안 죽는다.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
echo ==== local remaining start %date% %time% >> results\_local_remaining.log
.venv\Scripts\python.exe -u experiments\core_prune_during_learning\run_all_cifar.py --model resnet18 --densities 0.005 --seeds 2 --epochs 20 --arms dense_small pd_mag_global pd_mag_erk rigl ttp --skip_existing >> results\_local_remaining.log 2>&1
echo ==== resnet seed2 d0.005 done %date% %time% >> results\_local_remaining.log
.venv\Scripts\python.exe -u experiments\exp5_dynamic_pruning\run_all.py --arms dyn_local --seeds 0 1 --densities 0.2 --skip_existing >> results\_local_remaining.log 2>&1
echo LOCAL_REMAINING_DONE %date% %time% >> results\_local_remaining.log
