@echo off
rem p3 chain 2: H2 v2 protocol (cosine lr, divergence retry at lr/3, lr/10). sequential, skips finished runs.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\p3\_chain2.log
echo ==== p3 chain2 start %date% %time% >> %LOG%
%PY% -u experiments\p3\run_stage.py h2v2 >> %LOG% 2>&1
echo P3_CHAIN2_DONE %date% %time% >> %LOG%
echo done > results\p3\_done2
