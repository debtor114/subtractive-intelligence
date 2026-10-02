@echo off
rem 11 차 리뷰 대조군 (로컬 3060 Ti, 순차): CNN prune-after 를 원래 학습률 (0.05, cosine 재시작) 로 미세조정 (점진·one-shot, 1%% 와 3%%, 3 시드),
rem 사전학습 크기 마스크 RigL 의 5%%·2%% 두 번째 시드.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\_review11.log
echo ==== review11 controls start %date% %time% > %LOG%
for %%s in (0 1 2) do (
  for %%d in (0.01 0.03) do (
    %PY% -u experiments\core_prune_during_learning\run_cifar.py --model cnn --arm ttp_gradual --density %%d --seed %%s --set ft_lr=0.05 tag=ftlr05 >> %LOG% 2>&1
    %PY% -u experiments\core_prune_during_learning\run_cifar.py --model cnn --arm ttp --density %%d --seed %%s --set ft_lr=0.05 tag=ftlr05 >> %LOG% 2>&1
  )
)
%PY% -u experiments\core_prune_during_learning\run_pretrained.py --arm pt_rigl_mag --density 0.05 --seed 1 >> %LOG% 2>&1
%PY% -u experiments\core_prune_during_learning\run_pretrained.py --arm pt_rigl_mag --density 0.02 --seed 1 >> %LOG% 2>&1
echo REVIEW11_CONTROLS_DONE %date% %time% >> %LOG%
