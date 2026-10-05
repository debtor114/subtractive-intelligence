@echo off
rem 논문 2 탐색 밤샘 체인 2: 체인 1 (학습) 이 끝나길 기다린 뒤 GPU 벤치마크 (E1-micro, E1-real) -> X2 -> 집계.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\p2\_chain2.log
echo ==== p2 chain2 waiting for chain1 %date% %time% > %LOG%
:wait
findstr /c:"P2_CHAIN1_DONE" results\p2\_chain1.log >nul 2>&1
if errorlevel 1 (
  timeout /t 60 /nobreak >nul
  goto wait
)
echo ==== chain1 done, chain2 start %date% %time% >> %LOG%
%PY% -u experiments\p2\e1_micro.py >> %LOG% 2>&1
%PY% -u experiments\p2\e1_real.py >> %LOG% 2>&1
%PY% -u experiments\p2\x2_fastslow.py >> %LOG% 2>&1
%PY% -u experiments\p2\report_data.py >> %LOG% 2>&1
echo P2_CHAIN2_DONE %date% %time% >> %LOG%
