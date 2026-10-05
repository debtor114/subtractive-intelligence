@echo off
rem 논문 2 탐색 체인 3: 체인 2 가 끝나길 기다린 뒤 E4 추가 (층별 타일 예산) -> 집계 다시.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\p2\_chain3.log
echo ==== p2 chain3 waiting for chain2 %date% %time% > %LOG%
:wait
findstr /c:"P2_CHAIN2_DONE" results\p2\_chain2.log >nul 2>&1
if errorlevel 1 (
  timeout /t 60 /nobreak >nul
  goto wait
)
echo ==== chain3 start %date% %time% >> %LOG%
%PY% -u experiments\p2\run_stage.py e4pl >> %LOG% 2>&1
%PY% -u experiments\p2\report_data.py >> %LOG% 2>&1
echo P2_CHAIN3_DONE %date% %time% >> %LOG%
