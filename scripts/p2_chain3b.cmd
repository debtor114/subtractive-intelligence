@echo off
rem p2 chain3b: per-layer tile-budget controls (e4pl) then aggregation. No wait loop (chain2 already finished; findstr-based wait hung).
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\p2\_chain3b.log
echo ==== chain3b start %date% %time% > %LOG%
%PY% -u experiments\p2\run_stage.py e4pl >> %LOG% 2>&1
%PY% -u experiments\p2\report_data.py >> %LOG% 2>&1
echo P2_CHAIN3_DONE %date% %time% >> %LOG%
