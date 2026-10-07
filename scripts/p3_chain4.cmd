@echo off
rem p3 chain 4 (v3 protocol after code review): validation split, cosine lr selection (5 ep), backoff instead of restart, extra companies.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\p3\_chain4.log
echo ==== p3 chain4 start %date% %time% >> %LOG%
%PY% -u experiments\p3\run_stage.py h3lr h3 h3long >> %LOG% 2>&1
echo P3_CHAIN4_DONE %date% %time% >> %LOG%
echo done > results\p3\_done4
