@echo off
rem 14 차 리뷰 (선택): ResNet-18 0.5%% 에서 gradual prune-after 를 학습률 0.1 로 미세조정, 3 시드. 점진성 vs 학습 중 타이밍 분해를 세 모델로 완성한다.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\_review14.log
echo ==== review14 controls start %date% %time% > %LOG%
for %%s in (0 1 2) do %PY% -u experiments\core_prune_during_learning\run_cifar.py --model resnet18 --arm ttp_gradual --density 0.005 --seed %%s --set ft_lr=0.1 tag=ftlr10 >> %LOG% 2>&1
echo REVIEW14_CONTROLS_DONE %date% %time% >> %LOG%
