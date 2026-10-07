@echo off
rem p3 chain 3 (review-driven controls): H1b Wanda-faithful criteria -> degree-preserving random company -> 45-epoch convergence runs.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\p3\_chain3.log
echo ==== p3 chain3 start %date% %time% >> %LOG%
%PY% -u experiments\p3\run_stage.py h1b h2deg h2long >> %LOG% 2>&1
echo P3_CHAIN3_DONE %date% %time% >> %LOG%
echo done > results\p3\_done3
