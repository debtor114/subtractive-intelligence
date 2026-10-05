@echo off
rem 논문 2 탐색 밤샘 체인 1: 학습 단계 (E2 발달 순서 -> E4 타일 -> E3 두 물결/진도 -> X1 활동 희소성). 순차, 완료 런 건너뜀.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\p2\_chain1.log
echo ==== p2 chain1 start %date% %time% > %LOG%
%PY% -u experiments\p2\run_stage.py e2 e4 e3 x1 >> %LOG% 2>&1
echo P2_CHAIN1_DONE %date% %time% >> %LOG%
