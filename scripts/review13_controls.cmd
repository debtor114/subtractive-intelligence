@echo off
rem 13 차 리뷰: ResNet-18 0.5%% 의 one-shot prune-after 를 높은 학습률로 미세조정 (0.1 = ResNet 학습률, 0.05 = 리뷰 제안값), 3 시드.
rem 표준 프로토콜은 0.01 (학습률의 1/10). 시드마다 두 학습률을 붙여 돌려 시드 0 결과가 먼저 나오게 한다.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\_review13.log
echo ==== review13 controls start %date% %time% > %LOG%
for %%s in (0 1 2) do (
  %PY% -u experiments\core_prune_during_learning\run_cifar.py --model resnet18 --arm ttp --density 0.005 --seed %%s --set ft_lr=0.1 tag=ftlr10 >> %LOG% 2>&1
  %PY% -u experiments\core_prune_during_learning\run_cifar.py --model resnet18 --arm ttp --density 0.005 --seed %%s --set ft_lr=0.05 tag=ftlr05 >> %LOG% 2>&1
)
echo REVIEW13_CONTROLS_DONE %date% %time% >> %LOG%
