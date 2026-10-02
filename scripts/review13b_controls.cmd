@echo off
rem 13 차 리뷰 후속: 0.5%% 에서 학습률 0.1 미세조정이 one-shot prune-after 를 2 점 올렸으므로 (83.6 -> 85.6), 같은 기준선을 2%% 와 5%% 에도 맞춘다. 3 시드.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\_review13b.log
echo ==== review13b controls start %date% %time% > %LOG%
for %%s in (0 1 2) do (
  %PY% -u experiments\core_prune_during_learning\run_cifar.py --model resnet18 --arm ttp --density 0.02 --seed %%s --set ft_lr=0.1 tag=ftlr10 >> %LOG% 2>&1
  %PY% -u experiments\core_prune_during_learning\run_cifar.py --model resnet18 --arm ttp --density 0.05 --seed %%s --set ft_lr=0.1 tag=ftlr10 >> %LOG% 2>&1
)
echo REVIEW13B_CONTROLS_DONE %date% %time% >> %LOG%
